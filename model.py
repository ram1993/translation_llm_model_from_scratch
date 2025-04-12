import torch
import torch.nn as nn
import numpy as np

# Constants (Consider moving to a config file or main.py)
# Determine the best available device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

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
        # x shape: [seq_len, batch_size, d_model] (if batch_first=False)
        # or [batch_size, seq_len, d_model] (if batch_first=True)
        # Assuming batch_first=False convention often used in older PyTorch tutorials
        # If using batch_first=True, need to adjust slicing and addition
        # Let's assume input x is [seq_len, batch_size, d_model] for this PE implementation
        # If input is [batch_size, seq_len, d_model], transpose before adding PE
        # x = x.transpose(0, 1) # Uncomment if batch_first=True
        x = x + self.pe[:x.size(0), :]
        # x = x.transpose(0, 1) # Uncomment if batch_first=True
        return self.dropout(x)

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
        # Q, K, V shape: [batch_size, num_heads, seq_len, d_k]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        # attn_scores shape: [batch_size, num_heads, seq_len_q, seq_len_k]
        if mask is not None:
            # Mask shape should be broadcastable to attn_scores shape
            # e.g., [batch_size, 1, 1, seq_len_k] for src_mask
            # or [batch_size, 1, seq_len_q, seq_len_k] for tgt_mask
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        # output shape: [batch_size, num_heads, seq_len_q, d_k]
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_length, d_model = x.size()
        # output shape: [batch_size, num_heads, seq_len, d_k]
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # x shape: [batch_size, num_heads, seq_len, d_k]
        batch_size, _, seq_length, d_k = x.size()
        # output shape: [batch_size, seq_len, d_model]
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Input Q, K, V shape: [batch_size, seq_len, d_model]
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        # Q, K, V shape: [batch_size, num_heads, seq_len, d_k]

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # attn_output shape: [batch_size, num_heads, seq_len_q, d_k]
        output = self.W_o(self.combine_heads(attn_output))
        # output shape: [batch_size, seq_len_q, d_model]
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

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x shape: [batch_size, seq_len, d_model]
        # mask shape: [batch_size, 1, 1, seq_len]
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
        # x shape: [batch_size, tgt_len, d_model]
        # memory shape: [batch_size, src_len, d_model]
        # src_mask shape: [batch_size, 1, 1, src_len]
        # tgt_mask shape: [batch_size, 1, tgt_len, tgt_len]

        # Masked Self-Attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-Attention (Encoder-Decoder Attention)
        # Q from decoder (x), K/V from encoder memory
        attn_output = self.cross_attn(x, memory, memory, src_mask)
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
        # Note: PositionalEncoding expects input shape [seq_len, batch_size, d_model] by default
        # If using batch_first=True elsewhere, ensure PE handles it or transpose inputs/outputs
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
        # Use global DEVICE constant
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=DEVICE)).bool()
        # tgt_sub_mask shape: [tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        # tgt_mask shape: [batch_size, 1, tgt_len, tgt_len]
        return tgt_mask

    def encode(self, src, src_mask):
        # src shape: [batch_size, src_len]
        # src_mask shape: [batch_size, 1, 1, src_len]
        # Embedding output: [batch_size, src_len, d_model]
        src_embedded = self.src_embedding(src) * np.sqrt(self.d_model)
        # PositionalEncoding expects [seq_len, batch_size, d_model], so transpose
        src_embedded = self.dropout(self.pos_encoder(src_embedded.transpose(0, 1)))
        # Transpose back to [batch_size, seq_len, d_model] for encoder layers
        enc_output = src_embedded.transpose(0, 1)
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        # enc_output shape: [batch_size, src_len, d_model]
        return enc_output

    def decode(self, tgt, memory, src_mask, tgt_mask):
        # tgt shape: [batch_size, tgt_len]
        # memory shape: [batch_size, src_len, d_model]
        # src_mask shape: [batch_size, 1, 1, src_len]
        # tgt_mask shape: [batch_size, 1, tgt_len, tgt_len]
        # Embedding output: [batch_size, tgt_len, d_model]
        tgt_embedded = self.tgt_embedding(tgt) * np.sqrt(self.d_model)
        # PositionalEncoding expects [seq_len, batch_size, d_model], so transpose
        tgt_embedded = self.dropout(self.pos_encoder(tgt_embedded.transpose(0, 1)))
        # Transpose back to [batch_size, seq_len, d_model] for decoder layers
        dec_output = tgt_embedded.transpose(0, 1)
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, memory, src_mask, tgt_mask)
        # dec_output shape: [batch_size, tgt_len, d_model]
        return dec_output

    def forward(self, src, tgt):
        # src shape: [batch_size, src_len]
        # tgt shape: [batch_size, tgt_len]
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        # output shape: [batch_size, tgt_len, tgt_vocab_size]
        return output