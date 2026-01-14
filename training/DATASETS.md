# Dataset Download Guide

## Quick Start - Datasets Ready

✅ **TinyStories** - Downloaded and ready! (21.5 MB)
- Location: `data/tinystories/train.txt`
- Perfect for story generation training
- Train with: `python train.py --dataset data/tinystories/train.txt --n_layer 6 --n_head 6 --n_embd 384`

## OpenWebText - Manual Download Instructions

Due to the large size and recent API changes, OpenWebText requires manual download. Here are the best options:

### Option 1: Small Sample (Recommended for Testing)
Download a preprocessed sample (~100MB):
```bash
# Create directory
mkdir -p data/openwebtext

# Download from a mirror (you'll need to find a current one)
# Or use Wikipedia text as alternative
```

### Option 2: Use Wikipedia Instead (Easier Alternative)
Wikipedia dumps are easier to download and work great for training:

1. Visit: https://dumps.wikimedia.org/enwiki/latest/
2. Download: `enwiki-latest-pages-articles.xml.bz2` (~20GB compressed, ~80GB uncompressed)
3. Use WikiExtractor to process it

Or download a smaller sample:
```python
# Add this to your prepare_datasets.py
pip install wikipedia-api
```

### Option 3: Alternative Datasets (Good Substitutes)

**BookCorpus Alternative - Project Gutenberg:**
```bash
# Download from Project Gutenberg
# 1. Visit: https://www.gutenberg.org/
# 2. Download plain text books
# 3. Concatenate into one file
```

**News Dataset - CNN/DailyMail:**
```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
```

## Quick Training Commands

### Shakespeare (Already Complete)
```bash
python train.py --dataset data/shakespeare.txt --n_layer 6 --n_head 6 --n_embd 384 --batch_size 64 --epochs 10
```

### TinyStories (Ready to Go!)
```bash
# Small model for quick training
python train.py --dataset data/tinystories/train.txt \
    --n_layer 4 --n_head 4 --n_embd 256 \
    --batch_size 64 --epochs 5

# Medium model for better results
python train.py --dataset data/tinystories/train.txt \
    --n_layer 8 --n_head 8 --n_embd 512 \
    --batch_size 48 --epochs 10

# Large model (uses more GPU memory)
python train.py --dataset data/tinystories/train.txt \
    --n_layer 12 --n_head 12 --n_embd 768 \
    --batch_size 32 --epochs 15
```

### When You Get OpenWebText
```bash
python train.py --dataset data/openwebtext/train.txt \
    --n_layer 12 --n_head 12 --n_embd 768 \
    --batch_size 32 --epochs 5
```

## Next Steps

1. ✅ Shakespeare training (in progress/completed)
2. ✅ TinyStories downloaded and ready
3. ⬜ Train on TinyStories with different model sizes
4. ⬜ Manually download OpenWebText or alternative dataset
5. ⬜ Set up distributed training when A2000 arrives

## Recommended Training Sequence

Start with these in order:

1. **Shakespeare** (Quick test - 10 min)
   - Verifies setup works
   - Small dataset for fast iteration

2. **TinyStories** (2-4 hours)
   - Medium-sized dataset
   - Great for story generation
   - Good quality with reasonable training time

3. **Larger Dataset** (24-48 hours)
   - OpenWebText, Wikipedia, or BookCorpus
   - Better general language understanding
   - Requires longer training

## Alternative: Use Pre-trained Models

Instead of training from scratch, consider:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Download GPT-2
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Fine-tune on your data
```

This is much faster and often gives better results!
