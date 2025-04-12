import torch

# Constants (Consider moving to a config file or main.py)
# Determine the best available device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, max_length):
    """Translates a sentence using the trained model."""
    model.eval()
    with torch.no_grad():
        # Get BOS/EOS/PAD IDs from tokenizers
        # Use defaults if methods don't exist or return None
        src_bos_id = getattr(src_tokenizer, 'bos_id', lambda: 1)() or 1
        src_eos_id = getattr(src_tokenizer, 'eos_id', lambda: 2)() or 2
        tgt_bos_id = getattr(tgt_tokenizer, 'bos_id', lambda: 1)() or 1
        tgt_eos_id = getattr(tgt_tokenizer, 'eos_id', lambda: 2)() or 2
        # pad_id = getattr(src_tokenizer, 'pad_id', lambda: 0)() or 0 # Assuming pad_id is 0 in model

        tokens = src_tokenizer.encode(sentence)
        tokens = [src_bos_id] + tokens + [src_eos_id]
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE) # Shape: [1, src_len]

        src_mask = model.make_src_mask(src_tensor) # Use model's mask function
        memory = model.encode(src_tensor, src_mask) # Shape: [1, src_len, d_model]

        # Start decoding with BOS token
        ys = torch.ones(1, 1).fill_(tgt_bos_id).type_as(src_tensor).to(DEVICE) # Shape: [1, 1]

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
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor).fill_(next_word)], dim=1)

            # Stop if EOS token is generated
            if next_word == tgt_eos_id:
                break

        # Remove BOS token from the beginning
        translated_tokens = ys[0, 1:].tolist() # Convert to list of ints
        # Decode tokens back to string
        translated_sentence = tgt_tokenizer.decode(translated_tokens)

    return translated_sentence