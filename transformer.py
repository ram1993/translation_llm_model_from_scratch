import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu
import sentencepiece as spm
import os

# Download necessary NLTK data
# NLTK punkt is still needed for potential sentence splitting if not handled by datasets

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_HEADS = 8
NUM_LAYERS = 6
D_MODEL = 512
D_FF = 2048
DROPOUT = 0.1
MAX_LENGTH = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 3

# 1. Data Preprocessing
def load_data(max_length=MAX_LENGTH, limit=1000):
    """Loads and preprocesses the WMT14 English-to-French dataset using the datasets library."""
    try:
        # Load a subset of the training data
        dataset = load_dataset("wmt14", "fr-en", split=f'train[:{limit}]')
        # print(f"Loaded {len(dataset)} examples initially.") # Removed debug print

        en_sentences = []
        fr_sentences = []

        for example in dataset:
            en = example['translation']['en']
            fr = example['translation']['fr']
            # Basic check for sentence length (can be refined)
            if len(en.split()) <= max_length and len(fr.split()) <= max_length:
                en_sentences.append(en)
                fr_sentences.append(fr)

    except Exception as e:
        print(f"Error loading WMT14 data using datasets library: {e}")
        return [], []

    # print(f"Filtered down to {len(en_sentences)} English sentences.") # Removed debug print
    # print(f"Filtered down to {len(fr_sentences)} French sentences.") # Removed debug print
    return en_sentences, fr_sentences

def create_tokenizer(sentences, vocab_size=8000, lang='en'):
    """Creates a SentencePiece tokenizer."""
    spm_path = f'spm_{lang}.model'
    if not os.path.exists(spm_path):
        # Train SentencePiece model
        with open(f'temp_{lang}.txt', 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        spm.SentencePieceTrainer.train(
            f'--input=temp_{lang}.txt --model_prefix=spm_{lang} --vocab_size={vocab_size} --model_type=bpe'
        )
        os.remove(f'temp_{lang}.txt')

    sp = spm.SentencePieceProcessor(model_file=spm_path)
    return sp

def tokenize_and_pad(sentences, tokenizer, max_length=MAX_LENGTH):
    """Tokenizes and pads sentences."""
    tokens = [tokenizer.encode(sentence) for sentence in sentences]
    # Assuming PAD_ID = 0
    pad_id = 0
    tokens_padded = [
        t + [pad_id] * (max_length - len(t)) if len(t) < max_length else t[:max_length]
        for t in tokens
    ]
    return np.array(tokens_padded)

# 2. Model Definition
class EncoderLayer(nn.Module): # Renamed from TransformerLayer
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # Masked Self-Attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-Attention (Encoder-Decoder Attention)
        attn_output = self.cross_attn(x, memory, memory, src_mask) # Q from decoder, K/V from encoder memory
        x = self.norm2(x + self.dropout(attn_output))

        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, num_heads, num_layers, d_model, d_ff, dropout, pad_id=0):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        # src shape: [batch_size, src_len]
        src_mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)
        # src_mask shape: [batch_size, 1, 1, src_len]
        return src_mask

    def make_tgt_mask(self, tgt):
        # tgt shape: [batch_size, tgt_len]
        tgt_pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        # tgt_pad_mask shape: [batch_size, 1, 1, tgt_len]
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=DEVICE)).bool()
        # tgt_sub_mask shape: [tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        # tgt_mask shape: [batch_size, 1, tgt_len, tgt_len]
        return tgt_mask

    def encode(self, src, src_mask):
        src_embedded = self.dropout(self.pos_encoder(self.src_embedding(src) * np.sqrt(self.d_model)))
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        return enc_output

    def decode(self, tgt, memory, src_mask, tgt_mask):
        tgt_embedded = self.dropout(self.pos_encoder(self.tgt_embedding(tgt) * np.sqrt(self.d_model)))
        dec_output = tgt_embedded
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, memory, src_mask, tgt_mask)
        return dec_output

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# This class was renamed to EncoderLayer and moved above Transformer class

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 3. Training Loop
# Mask creation is now handled inside the Transformer model's forward pass
# def create_mask(src, tgt, src_pad=0, tgt_pad=0): ... (Removed)

def train(model, iterator, optimizer, criterion, clip=1):
    """Trains the model for one epoch."""
    model.train()
    epoch_loss = 0
    for i, (src, tgt) in enumerate(iterator):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE) # Shape: [batch_size, tgt_len]

        optimizer.zero_grad()

        # The target sequence for the decoder input needs to be shifted right (exclude <eos>)
        # The target sequence for the loss calculation needs to exclude <sos>
        tgt_input = tgt[:, :-1] # Shape: [batch_size, tgt_len - 1]
        tgt_output = tgt[:, 1:] # Shape: [batch_size, tgt_len - 1]

        # Forward pass through the model
        # Note: model.forward now handles mask creation internally
        output = model(src, tgt_input) # Shape: [batch_size, tgt_len - 1, tgt_vocab_size]

        # Reshape output to (batch_size * seq_len, vocab_size) and tgt to (batch_size * seq_len)
        # Reshape output to (batch_size * (tgt_len - 1), tgt_vocab_size)
        # Reshape tgt_output to (batch_size * (tgt_len - 1))
        output = output.contiguous().view(-1, output.shape[-1])
        tgt_output = tgt_output.contiguous().view(-1)

        loss = criterion(output, tgt_output)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    """Evaluates the model."""
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, tgt) in enumerate(iterator):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            # Prepare inputs/outputs similar to training loop
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Forward pass
            output = model(src, tgt_input) # Shape: [batch_size, tgt_len - 1, tgt_vocab_size]

            # Reshape for loss calculation
            output = output.contiguous().view(-1, output.shape[-1])
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, max_length=MAX_LENGTH):
    """Translates a sentence from English to French."""
    model.eval()
    with torch.no_grad():
        # Get BOS/EOS/PAD IDs from tokenizers
        src_bos_id = src_tokenizer.bos_id() if src_tokenizer.bos_id() is not None else 1 # Default to 1 if None
        src_eos_id = src_tokenizer.eos_id() if src_tokenizer.eos_id() is not None else 2 # Default to 2 if None
        tgt_bos_id = tgt_tokenizer.bos_id() if tgt_tokenizer.bos_id() is not None else 1 # Default to 1 if None
        tgt_eos_id = tgt_tokenizer.eos_id() if tgt_tokenizer.eos_id() is not None else 2 # Default to 2 if None
        pad_id = 0 # Assuming pad_id is 0

        tokens = src_tokenizer.encode(sentence)
        tokens = [src_bos_id] + tokens + [src_eos_id]
        src = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE) # Shape: [1, src_len]

        src_mask = model.make_src_mask(src) # Use model's mask function
        memory = model.encode(src, src_mask) # Shape: [1, src_len, d_model]

        # Start decoding with BOS token
        ys = torch.ones(1, 1).fill_(tgt_bos_id).type_as(src).to(DEVICE) # Shape: [1, 1]

        for i in range(max_length - 1):
            # Create target mask for current decoded sequence
            tgt_mask = model.make_tgt_mask(ys) # Shape: [1, 1, cur_len, cur_len]

            # Decode step
            out = model.decode(ys, memory, src_mask, tgt_mask) # Shape: [1, cur_len, d_model]

            # Get probabilities for the last token
            prob = model.fc(out[:, -1]) # Shape: [1, tgt_vocab_size]
            _, next_word = torch.max(prob, dim=1) # Shape: [1]
            next_word = next_word.item()

            # Append predicted token
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)

            # Stop if EOS token is generated
            if next_word == tgt_eos_id:
                break

        # Remove BOS token from the beginning
        translated_tokens = [token.item() for token in ys[0, 1:]]
        # Decode tokens back to string
        translated_sentence = tgt_tokenizer.decode(translated_tokens)
    return translated_sentence

# Main function
def main():
    # 1. Data Preprocessing
    en_sentences, fr_sentences = load_data()
    en_tokenizer = create_tokenizer(en_sentences, lang='en')
    fr_tokenizer = create_tokenizer(fr_sentences, lang='fr')
    en_padded = tokenize_and_pad(en_sentences, en_tokenizer)
    fr_padded = tokenize_and_pad(fr_sentences, fr_tokenizer)

    # 2. Model Definition
    # Get vocab size directly from the tokenizer
    src_vocab_size = en_tokenizer.get_piece_size()
    tgt_vocab_size = fr_tokenizer.get_piece_size()
    model = Transformer(src_vocab_size, tgt_vocab_size, NUM_HEADS, NUM_LAYERS, D_MODEL, D_FF, DROPOUT).to(DEVICE)

    # 3. Training
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pad_id = 0 # Define pad_id, should match tokenizer/padding
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    train_iterator = [(torch.LongTensor(en_padded[i:i+BATCH_SIZE]), torch.LongTensor(fr_padded[i:i+BATCH_SIZE]))
                      for i in range(0, len(en_padded), BATCH_SIZE)]

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}")

    # 4. Translation
    example_sentence = "Hello, how are you?"
    translated_sentence = translate_sentence(model, example_sentence, en_tokenizer, fr_tokenizer)
    print(f"Original sentence: {example_sentence}")
    print(f"Translated sentence: {translated_sentence}")

if __name__ == "__main__":
    main()