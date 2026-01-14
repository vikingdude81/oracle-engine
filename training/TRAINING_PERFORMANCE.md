# Training Performance Guide - Optimized for Dual GPU Setup

## What Changed?

### 🚀 Major Optimizations:
1. **Reduced recursive steps**: 16 → 3 (default)
   - Paper uses 16, but 3 is much faster for training
   - Can increase later for better quality if needed
   
2. **Parallel data loading**: 0 → 8 workers
   - Uses your 32 CPU threads to load data
   - GPUs never wait for data
   
3. **Pin memory**: Enabled for faster GPU transfers

4. **Timing**: Added epoch timing to track progress

## Expected Training Times

### TinyStories (22.5 MB)

**Before (Single GPU, 16 recursive steps):**
- ~12 hours per epoch ❌

**After (Single GPU, 3 recursive steps):**
- ~20-30 minutes per epoch ✅

**After (Dual GPU, 3 recursive steps):**
- ~12-18 minutes per epoch ✅✅
- Effective batch size: 64 (32 per GPU)

### Shakespeare (1.1 MB)

**Single GPU:**
- ~5-10 minutes per epoch

**Dual GPU:**
- Not worth it (overhead > benefit)

## Commands

### Single GPU Training (Recommended for Shakespeare & TinyStories)
```bash
# Activate virtual environment first
.venv\Scripts\Activate.ps1

# Train TinyStories
python trm_train.py --dataset data/tinystories/train.txt --epochs 10 --max_recursive_steps 3
```

### Dual GPU Training (Best for TinyStories & OpenWebText)
```bash
# Activate virtual environment first
.venv\Scripts\Activate.ps1

# Train TinyStories on both GPUs
torchrun --nproc_per_node=2 distributed_trm_train.py ^
    --dataset data/tinystories/train.txt ^
    --epochs 10 ^
    --max_recursive_steps 3 ^
    --batch_size 32

# Note: Effective batch size = 64 (32 per GPU × 2 GPUs)
```

### Resume Training from Epoch 1
```bash
# TODO: Need to add --resume flag to training scripts
# For now, you can manually adjust the training script to load checkpoint
```

## Testing Different Recursive Steps

```bash
# Fast training (3 steps) - Default
python trm_train.py --dataset data/tinystories/train.txt --max_recursive_steps 3

# Medium quality (8 steps) - 2.7x slower
python trm_train.py --dataset data/tinystories/train.txt --max_recursive_steps 8

# Paper quality (16 steps) - 5.3x slower
python trm_train.py --dataset data/tinystories/train.txt --max_recursive_steps 16
```

## Generate Text from Models

```bash
# Generate from TinyStories epoch 1
python trm_generate.py --checkpoint checkpoints/tinystories_epoch1.pt --prompt "Once upon a time"

# Generate from Shakespeare
python generate.py --checkpoint checkpoints/shakespeare_final.pt --prompt "ROMEO:"
```

## Performance Tips

1. **Start with 3 recursive steps** - Get quick results
2. **Use single GPU for small datasets** (Shakespeare, small experiments)
3. **Use dual GPU for larger datasets** (TinyStories with 10+ epochs, OpenWebText)
4. **Monitor GPU usage**: `nvidia-smi -l 1` (updates every second)
5. **Watch for bottlenecks**: If GPU utilization < 80%, increase batch size

## Your System Specs
- **CPUs**: 32 threads (Threadripper PRO 5955WX)
- **RAM**: 128 GB
- **GPU 1**: RTX 4070 Ti (12 GB VRAM)
- **GPU 2**: RTX A2000 (12 GB VRAM)
- **Total VRAM**: 24 GB

## Next Steps

1. Continue training TinyStories from epoch 1
2. Test generation quality
3. If quality is good, continue with more epochs
4. If quality needs improvement, try more recursive steps
