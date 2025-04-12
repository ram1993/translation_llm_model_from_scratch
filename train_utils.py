import torch
import torch.nn as nn

# Constants (Consider moving to a config file or main.py)
# Determine the best available device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

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

        # Reshape output to (batch_size * (tgt_len - 1), tgt_vocab_size)
        # Reshape tgt_output to (batch_size * (tgt_len - 1))
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        tgt_output = tgt_output.contiguous().view(-1)

        loss = criterion(output, tgt_output)
        loss.backward()

        # Clip gradients to prevent exploding gradients
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
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)