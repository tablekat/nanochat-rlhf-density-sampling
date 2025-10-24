# ⚡ Quick Start - RunPod GPU Fix

## 🚀 Start Here (Just Copy & Paste)

```bash
# 1. Check your GPUs
bash fix_gpu.sh

# 2. Run the full pipeline (auto-detects GPUs)
bash kat_speedrun.sh

# Expected output:
# [GPU Detection] Detecting available GPUs...
# ✓ Found X GPU(s), using all of them
# ✓ Environment ready
# [... training starts ...]
```

**That's it!** The script will auto-detect your GPUs and run everything.

---

## ⚠️ If You Get Errors

### Error: "CUDA device(s) is/are busy or unavailable"

```bash
# Kill any hanging processes
pkill -f python
pkill -f torchrun

# Reset CUDA cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Try again
bash kat_speedrun.sh
```

### Error: "invalid device ordinal"

```bash
# This means you're trying to use more GPUs than available
# The fixed script handles this automatically now, but if it persists:

# Check how many GPUs you have
bash fix_gpu.sh

# Use fewer GPUs explicitly
export NPROC_PER_NODE=4
bash kat_speedrun.sh
```

### Error: Out of Memory (OOM)

```bash
# Use fewer GPUs or reduce batch size
export NPROC_PER_NODE=4
bash kat_speedrun.sh

# Or for more control, edit device_batch_size in scripts
# (see RUNPOD_SETUP.md for details)
```

---

## 📊 Monitor While Running

```bash
# Terminal 1: Watch GPUs
watch -n 1 nvidia-smi

# Terminal 2: Watch logs
tail -f kat_speedrun.log | grep -E "GPU|CUDA|Step|Loss"

# Terminal 3: Check processes
ps aux | grep python
```

---

## 🔧 Common Commands

```bash
# Diagnose GPU issues
bash fix_gpu.sh

# Run with custom GPU count (e.g., 4 GPUs)
export NPROC_PER_NODE=4
bash kat_speedrun.sh

# Run with logging to file
screen -L -Logfile kat_speedrun.log -S kat_speedrun bash kat_speedrun.sh
# Detach: Ctrl+A then D
# Reattach: screen -r kat_speedrun

# View final results
cat .cache/diversity_report.md
```

---

## ✅ What Was Fixed

| Issue              | Before              | After                        |
| ------------------ | ------------------- | ---------------------------- |
| GPU Count Mismatch | ❌ Hardcoded to 7   | ✅ Auto-detects              |
| Too Few GPUs       | ❌ Crashes          | ✅ Works with any number     |
| GPU Busy           | ❌ Crashes          | ✅ Attempts recovery         |
| Manual Override    | ❌ Must edit script | ✅ `export NPROC_PER_NODE=X` |

---

## 📚 Need More Help?

- **Detailed setup**: See `RUNPOD_SETUP.md`
- **GPU diagnostics**: Run `bash fix_gpu.sh`
- **Full summary**: See `RUNPOD_GPU_FIX_SUMMARY.md`

---

## 🎯 Expected Timeline

| Stage            | Time            |
| ---------------- | --------------- |
| Tokenizer        | 30 min - 1 hour |
| Base Pretraining | 2-4 hours       |
| SFT              | 2-4 hours       |
| GRPO (both runs) | 8-16 hours      |
| **Total**        | **~1-2 days**   |

---

**Everything should just work now.** If you hit any issues, reference the troubleshooting above or check `RUNPOD_SETUP.md` for the comprehensive guide.

Good luck! 🚀
