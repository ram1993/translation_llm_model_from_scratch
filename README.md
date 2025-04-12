# Transformer Machine Translation (English to French)

This project implements a Transformer model using PyTorch for translating English text to French.

## Prerequisites

*   Python 3.x
*   PyTorch
*   SentencePiece
    *   Run `pip install torch sentencepiece` (preferably in a virtual environment).

## Usage

### Training

(Verify arguments in `main.py` or `train_utils.py`)

```bash
# Example command (adjust paths and arguments as needed)
python main.py --mode train --data_path ./data --save_path ./checkpoints
```

### Translation

(Verify arguments in `translate.py`)

```bash
# Example command (adjust paths and arguments as needed)
python translate.py --model_path transformer_checkpoint.pt --sp_en_model_path spm_en.model --sp_fr_model_path spm_fr.model --text "Your English text here"
```

## Notes

*   Requires trained SentencePiece models (`.model`, `.vocab`) for tokenization.
*   Requires a trained model checkpoint (`.pt`) for translation.
*   The `.gitignore` file excludes model files and common Python artifacts.