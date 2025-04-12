import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Import from our modules
from model import Transformer
from data_utils import load_data, create_tokenizer, tokenize_and_pad
from train_utils import train # Removed evaluate import as it's not used in this main flow
from translate import translate_sentence

# Constants (Configuration)
# Determine the best available device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
NUM_HEADS = 8
NUM_LAYERS = 6 # Standard Transformer Base model has 6 layers
D_MODEL = 512 # Standard Transformer Base model has d_model=512
D_FF = 2048 # Standard Transformer Base model has d_ff=2048
DROPOUT = 0.1
MAX_LENGTH = 50 # Maximum sequence length
BATCH_SIZE = 32 # Adjusted batch size
LEARNING_RATE = 0.0001
NUM_EPOCHS = 52 # Keep low for demonstration
DATA_LIMIT = 5000 # Increase data limit slightly for better tokenization/training
VOCAB_SIZE = 10000 # Increased vocab size
CLIP = 1.0 # Gradient clipping value
PAD_ID = 0 # Ensure this matches tokenizer settings

# Main execution function
def run():
    print(f"Using device: {DEVICE}")

    # 1. Data Preprocessing
    print("Loading and preprocessing data...")
    en_sentences, fr_sentences = load_data(max_length=MAX_LENGTH, limit=DATA_LIMIT)
    if not en_sentences or not fr_sentences:
        print("Failed to load data. Exiting.")
        return

    print(f"Loaded {len(en_sentences)} sentence pairs.")

    print("Creating/loading tokenizers...")
    # Ensure model_prefix is consistent if reusing tokenizers
    en_tokenizer = create_tokenizer(en_sentences, vocab_size=VOCAB_SIZE, lang='en', model_prefix='spm_wmt')
    fr_tokenizer = create_tokenizer(fr_sentences, vocab_size=VOCAB_SIZE, lang='fr', model_prefix='spm_wmt')
    print("Tokenizers ready.")

    print("Tokenizing and padding data...")
    en_padded = tokenize_and_pad(en_sentences, en_tokenizer, max_length=MAX_LENGTH)
    fr_padded = tokenize_and_pad(fr_sentences, fr_tokenizer, max_length=MAX_LENGTH)
    print("Data tokenized and padded.")

    # 2. Model Definition
    print("Defining model...")
    src_vocab_size = en_tokenizer.get_piece_size()
    tgt_vocab_size = fr_tokenizer.get_piece_size()
    print(f"Source Vocab Size: {src_vocab_size}, Target Vocab Size: {tgt_vocab_size}")

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        d_ff=D_FF,
        dropout=DROPOUT,
        pad_id=PAD_ID # Pass pad_id to model
    ).to(DEVICE)
    print("Model defined.")

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {total_params:,} trainable parameters')

    # 3. Training Setup (Optimizer, Criterion)
    print("Setting up optimizer and loss function...")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    print("Optimizer and loss function ready.")

    # --- Checkpoint Loading ---
    model_save_path = 'transformer_checkpoint.pt' # Changed extension for clarity
    start_epoch = 0
    if os.path.exists(model_save_path):
        print(f"Loading checkpoint from {model_save_path}...")
        try:
            checkpoint = torch.load(model_save_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
            print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0 # Reset epoch if loading fails
    else:
        print("No checkpoint found. Starting training from scratch.")

    # (Optimizer and Criterion setup moved before checkpoint loading)

    print("Creating data iterator...")
    # Convert numpy arrays to tensors before creating the iterator
    en_tensor = torch.LongTensor(en_padded)
    fr_tensor = torch.LongTensor(fr_padded)
    dataset = list(zip(en_tensor, fr_tensor)) # Create list of tuples

    # Simple batching (consider DataLoader for more features)
    train_iterator = [
        (dataset[i][0].unsqueeze(0), dataset[i][1].unsqueeze(0)) # Process one sentence pair at a time for simplicity
        if BATCH_SIZE == 1 else
        (torch.stack([item[0] for item in dataset[i:min(i + BATCH_SIZE, len(dataset))]]),
         torch.stack([item[1] for item in dataset[i:min(i + BATCH_SIZE, len(dataset))]]))
        for i in range(0, len(dataset), BATCH_SIZE)
    ]
    print(f"Data iterator created with {len(train_iterator)} batches.")


    print(f"Starting training from epoch {start_epoch} up to {NUM_EPOCHS}...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        # Add evaluation step here if needed using evaluate() from train_utils
        print(f"Epoch: {epoch+1:02}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}")

        # --- Periodic Checkpointing ---
        # Save every 10 epochs (and on the last epoch)
        # Save every 10 epochs (and on the last epoch)
        # Note: epoch is 0-based, epoch+1 is the actual epoch number completed
        current_epoch_num = epoch + 1
        if current_epoch_num % 10 == 0 or current_epoch_num == NUM_EPOCHS:
            print(f"Saving checkpoint at epoch {current_epoch_num} to {model_save_path}...")
            checkpoint = {
                'epoch': epoch, # Save the completed epoch index (0-based)
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, model_save_path)
            print("Checkpoint saved.")

            # 4. Translation Example
            print("\nPerforming translation example...")
            example_sentence = "This is a test sentence."
            # Ensure max_length is passed to translate_sentence
            translated_sentence = translate_sentence(model, example_sentence, en_tokenizer, fr_tokenizer, max_length=MAX_LENGTH)
            print(f"Original sentence (en): {example_sentence}")
            print(f"Translated sentence (fr): {translated_sentence}")

    print("Training complete.")
    # Final save after loop is removed as checkpointing happens within the loop

    # 4. Translation Example
    print("\nPerforming translation example...")
    example_sentence = "This is a test sentence."
    # Ensure max_length is passed to translate_sentence
    translated_sentence = translate_sentence(model, example_sentence, en_tokenizer, fr_tokenizer, max_length=MAX_LENGTH)
    print(f"Original sentence (en): {example_sentence}")
    print(f"Translated sentence (fr): {translated_sentence}")

if __name__ == "__main__":
    run()