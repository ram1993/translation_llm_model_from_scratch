import numpy as np
import nltk
from datasets import load_dataset
import sentencepiece as spm
import os

# Download necessary NLTK data (might be better in main script or setup)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_data(max_length, limit=1000):
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
            # Using split() might not be robust for all languages/tokenization schemes
            if len(en.split()) <= max_length and len(fr.split()) <= max_length:
                en_sentences.append(en)
                fr_sentences.append(fr)

    except Exception as e:
        print(f"Error loading WMT14 data using datasets library: {e}")
        # Consider raising the exception or handling it more robustly
        return [], []

    # print(f"Filtered down to {len(en_sentences)} English sentences.") # Removed debug print
    # print(f"Filtered down to {len(fr_sentences)} French sentences.") # Removed debug print
    return en_sentences, fr_sentences

def create_tokenizer(sentences, vocab_size=8000, lang='en', model_prefix='spm'):
    """Creates or loads a SentencePiece tokenizer."""
    spm_path = f'{model_prefix}_{lang}.model'
    if not os.path.exists(spm_path):
        print(f"Training SentencePiece tokenizer for {lang}...")
        # Train SentencePiece model
        temp_file = f'temp_{lang}.txt'
        with open(temp_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        try:
            spm.SentencePieceTrainer.train(
                f'--input={temp_file} --model_prefix={model_prefix}_{lang} --vocab_size={vocab_size} '
                f'--model_type=bpe --character_coverage=1.0 --input_sentence_size=1000000 ' # Limit input sentences for faster training
                f'--shuffle_input_sentence=true '
                f'--bos_id=1 --eos_id=2 --unk_id=3 --pad_id=0' # Explicitly set special token IDs
            )
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        print(f"Tokenizer training complete for {lang}.")

    sp = spm.SentencePieceProcessor(model_file=spm_path)
    # Ensure special token IDs are consistent
    assert sp.pad_id() == 0, "PAD ID mismatch"
    assert sp.bos_id() == 1, "BOS ID mismatch"
    assert sp.eos_id() == 2, "EOS ID mismatch"
    assert sp.unk_id() == 3, "UNK ID mismatch"
    return sp

def tokenize_and_pad(sentences, tokenizer, max_length):
    """Tokenizes and pads sentences using the provided tokenizer."""
    pad_id = tokenizer.pad_id() # Get pad_id from tokenizer
    tokens = [tokenizer.encode(sentence) for sentence in sentences]
    tokens_padded = [
        t + [pad_id] * (max_length - len(t)) if len(t) < max_length else t[:max_length]
        for t in tokens
    ]
    return np.array(tokens_padded)