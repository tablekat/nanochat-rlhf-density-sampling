# RunPod Setup Guide for NanoChat RLHF Density Sampling

This guide provides step-by-step instructions for running the full KAT speedrun pipeline on RunPod.

## Quick Start (30 seconds)

```bash
# SSH into RunPod
git clone https://github.com/tablekat/nanochat-rlhf-density-sampling.git
cd nanochat-rlhf-density-sampling

# Run the full pipeline with GPU auto-detection
bash kat_speedrun.sh
```

The script will:

- ✅ Automatically detect available GPUs
- ✅ Run all 8 stages (tokenizer → pretraining → SFT → GRPO → eval)
- ✅ Handle GPU unavailability gracefully
- ✅ Generate a diversity report

## Prerequisites

### 1. Check Your RunPod Instance

```bash
# SSH into your RunPod instance
# Check GPU count
nvidia-smi
# Should show output like:
# GPU 0: H100 80GB
# GPU 1: H100 80GB
# ...
# (Note the exact count - important for next steps)

# Verify CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### 2. Install System Dependencies

RunPod usually has Python 3.10+ and CUDA 11.8/12.0 pre-installed. If not:

```bash
sudo apt-get update -y
sudo apt-get install -y \
    git \
    python3-venv \
    python3-pip \
    screen \
    build-essential \
    curl

# Install Rust (needed for tokenizer compilation)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

### 3. Verify CUDA Compatibility

```bash
# Check CUDA version
nvcc --version

# If CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If CUDA 12.0+:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Running the Full Pipeline

### Option 1: Simple (Recommended)

```bash
bash kat_speedrun.sh
```

This will:

- Auto-detect GPU count
- Use all available GPUs (up to 8)
- Run full pipeline
- Generate report

**Expected time**: 1-2 days on 8xH100

### Option 2: With Screen Session (Recommended for SSH)

```bash
# Create a persistent screen session
screen -L -Logfile kat_speedrun.log -S kat_speedrun bash kat_speedrun.sh

# Detach: Ctrl+A, then D
# Reattach: screen -r kat_speedrun
# View logs: tail -f kat_speedrun.log
```

### Option 3: With Custom GPU Count

If you want to force a specific number of GPUs:

```bash
export NPROC_PER_NODE=4  # Use 4 GPUs instead of auto-detected
bash kat_speedrun.sh
```

### Option 4: With Weights & Biases Logging

```bash
export WANDB_RUN=my_experiment_name
bash kat_speedrun.sh
```

## Troubleshooting

### Problem: CUDA Error - "device(s) is/are busy or unavailable"

**Cause**: Another process is using the GPUs or the GPU count mismatch

**Solution**:

```bash
# 1. Check what's using the GPUs
nvidia-smi

# 2. Kill any hanging processes
pkill -f python
pkill -f torchrun

# 3. Reset CUDA
python3 -c "import torch; torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()"

# 4. Restart with fewer GPUs
export NPROC_PER_NODE=4
bash kat_speedrun.sh
```

### Problem: CUDA Error - "invalid device ordinal"

**Cause**: Requesting more GPUs than available

**Example**: RunPod has 4 GPUs but script tries to use 7

**Solution**:

```bash
# Run auto-detection (already built-in to kat_speedrun.sh)
bash kat_speedrun.sh

# Or manually check and set:
python3 -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}')"
export NPROC_PER_NODE=4  # Use actual count
bash kat_speedrun.sh
```

### Problem: Out of Memory (OOM)

**Cause**: Batch size too large for available GPU memory

**Solution**:

```bash
# Option 1: Use fewer GPUs
export NPROC_PER_NODE=4

# Option 2: Reduce batch size (in training scripts)
# Edit the script to add: --device_batch_size=2

# Option 3: Mix both approaches
export NPROC_PER_NODE=4
bash kat_speedrun.sh
```

### Problem: "RuntimeError: NCCL operation timed out"

**Cause**: Network communication slow between ranks

**Solution**:

```bash
# Increase timeout
export NCCL_TIMEOUT=600  # 10 minutes
bash kat_speedrun.sh
```

### Problem: Tokenizer Training Fails

**Cause**: Rust/Maturin not installed or old version

**Solution**:

```bash
# Reinstall Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Rebuild
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Problem: "ModuleNotFoundError" or "No module named"

**Cause**: Python environment not activated or dependencies missing

**Solution**:

```bash
# Ensure we're in repo root
cd /path/to/nanochat-rlhf-density-sampling

# Recreate environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install datasets ujson tqdm nltk numpy scipy scikit-learn
pip install sentence-transformers umap-learn
pip install fastapi uvicorn pydantic tensorboard regex python-multipart
pip install transformers huggingface-hub

# Or use uv (faster, recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra gpu
source .venv/bin/activate
```

## Understanding the Pipeline Output

### Stage Progress

```
[1/6] Setting up Python environment...
✓ Environment ready

[2/6] Training Tokenizer...
✓ Tokenizer trained

[3/6] Pretraining base model (depth=20)...
✓ Base model pretrained

[4/6] Mid-training...
✓ Mid-training complete

[5/6] Supervised Fine-Tuning...
✓ SFT complete

[6/6] Preparing pairwise preference data for GRPO...
  [6a/6] Downloading preference pairs...
  [6b/6] Deduplicating prompts...
  [6c/6] Training Reward Model...
✓ Preference data pipeline complete

MAIN EXPERIMENT: GRPO with Density-Aware Sampling
✓ GRPO with density sampling complete

BASELINE: GRPO without Density Sampling (uniform sampling)
✓ GRPO baseline complete

EVALUATION: Testing Hypothesis
Report saved to: .cache/diversity_report.md
```

### GPU Detection

```
[GPU Detection] Detecting available GPUs...
✓ Found 8 GPU(s), using 7 GPUs (can override with NPROC_PER_NODE env var)
  Using NPROC_PER_NODE=7 (from environment or detection)
```

### Key Outputs

After completion:

- `outs/grpo_density/ckpt.pt` — Model trained with density sampling
- `outs/grpo_baseline/ckpt.pt` — Baseline model without density sampling
- `.cache/diversity_report.md` — Full evaluation report
- `~/.cache/nanochat/` — Tokenizer, pretrained models, data

## Checking Progress

While running:

```bash
# In another terminal, check GPU usage
watch -n 1 nvidia-smi

# View logs
tail -f kat_speedrun.log

# Check processes
ps aux | grep python

# View output directory
ls -lh outs/
ls -lh ~/.cache/nanochat/
```

## Intermediate Checkpoints

If the script crashes, you can resume from checkpoints:

```bash
# Resume from SFT stage (skip tokenizer, pretraining, mid-training)
# Just comment out earlier stages or delete their output files to skip

# Delete completed stages to re-run:
rm -f outs/base/ckpt.pt  # Redo pretraining
rm -f outs/mid/ckpt.pt   # Redo mid-training
rm -f outs/sft/ckpt.pt   # Redo SFT

# Then run normally
bash kat_speedrun.sh
```

## Performance Notes

### Expected Times (8xH100)

| Stage                        | Duration        |
| ---------------------------- | --------------- |
| Tokenizer training           | 30 min - 1 hour |
| Base pretraining             | 2-4 hours       |
| Mid-training                 | 1-2 hours       |
| SFT                          | 2-4 hours       |
| Preference data download     | 20-40 min       |
| Reward model training        | 2-4 hours       |
| GRPO with density (5k steps) | 4-8 hours       |
| GRPO baseline (5k steps)     | 3-6 hours       |
| **Total**                    | **~1-2 days**   |

### Memory Requirements

| Stage       | GPU Memory        |
| ----------- | ----------------- |
| Pretraining | ~60-70 GB per GPU |
| SFT         | ~60-70 GB per GPU |
| GRPO        | ~60-70 GB per GPU |

**Tip**: If you get OOM errors, reduce `device_batch_size` or `--nproc_per_node`

## Advanced Configuration

### Environment Variables

```bash
# GPU/Compute
export NPROC_PER_NODE=8           # Number of processes per node
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specific GPUs to use
export CUDA_LAUNCH_BLOCKING=1     # Debug CUDA errors
export TORCH_USE_CUDA_DSA=1       # Enable device-side assertions
export NCCL_TIMEOUT=600           # Increase NCCL timeout (seconds)

# Memory
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Data
export NANOCHAT_BASE_DIR="/custom/cache/path"

# Logging
export WANDB_RUN="experiment_name"
export OMP_NUM_THREADS=1          # Single-threaded OpenMP
```

### Single GPU Testing

```bash
# Test on 1 GPU before running full pipeline
export NPROC_PER_NODE=1

# Quick test of SFT
python -m scripts.chat_sft --device_batch_size=2 --eval_every=10

# Then run full pipeline with more GPUs
export NPROC_PER_NODE=8
bash kat_speedrun.sh
```

## Monitoring & Debugging

### Watch GPU Usage in Real-Time

```bash
# Terminal 1: Watch GPU status
watch -n 1 nvidia-smi

# Terminal 2: Watch training logs
tail -f kat_speedrun.log | grep -E "Step|Loss|Rank"
```

### Debug CUDA Issues

```bash
# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Re-run with debug output
bash kat_speedrun.sh 2>&1 | tee debug_kat_speedrun.log
```

### Check Disk Space

```bash
# Monitor disk usage
df -h
du -sh ~/.cache/nanochat/
du -sh ./outs/

# Clear old cache if needed
rm -rf ~/.cache/nanochat/tok/
```

## Post-Run Analysis

### View Results

```bash
# Check final report
cat .cache/diversity_report.md

# List generated models
ls -lh outs/grpo_*/ckpt.pt

# Test density model interactively
python -m scripts.chat_cli --ckpt_path outs/grpo_density/ckpt.pt

# Start web UI
python -m scripts.chat_web --ckpt_path outs/grpo_density/ckpt.pt --port 3000
# Then visit: http://<your-runpod-url>:3000
```

## Support & Additional Resources

### Common Issues by Stage

**Tokenizer**: Usually works fine. If fails, check Rust install
**Pretraining**: Most GPU intensive. May need to reduce batch size
**SFT**: Typically stable. Check if base checkpoint exists
**GRPO**: Requires reward model. Ensure RM training completed
**Eval**: Usually fast. Check if model checkpoints exist

### Getting Help

1. Check logs: `cat kat_speedrun.log | tail -100`
2. Review error: Look for keywords like "CUDA", "OOM", "timeout"
3. Check this guide's troubleshooting section
4. Check original repo issues/discussions
5. Try with fewer GPUs: `export NPROC_PER_NODE=4`

## Cleanup

```bash
# Save important files before cleanup
cp .cache/diversity_report.md ~/diversity_report_backup.md

# Remove all outputs
rm -rf outs/
rm -rf ~/.cache/nanochat/

# Reset to fresh state (keeps environment)
bash kat_speedrun.sh  # Restart fresh
```

---

**Last Updated**: October 2025
**Tested On**: RunPod with H100 GPUs
**AutoDetect**: ✅ Enabled by default in kat_speedrun.sh
