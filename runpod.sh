#!/bin/bash

# ============================================================================
# RunPod Setup Script for nanochat-rlhf-density-sampling
# ============================================================================
# Run this inside a screen session after cloning the repo
# Usage:
#   screen -S speedrun
#   bash runpod.sh
#   bash kat_speedrun.sh
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Global Configuration - Save to /workspace for persistent storage
# ============================================================================
# export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/workspace/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

echo "================================================================================"
echo "RunPod Setup for nanochat-rlhf-density-sampling"
echo "================================================================================"
echo "Start time: $(date)"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo ""

# ============================================================================
# System Package Installation (from README)
# ============================================================================
echo "[1/5] Installing system packages..."
sudo apt-get update -y && sudo apt-get install -y git python3-venv screen
echo "✓ System packages installed"
echo ""

# ============================================================================
# Python Virtual Environment Setup (from README)
# ============================================================================
echo "[2/5] Setting up Python virtual environment..."
python3 -m venv .venv && source .venv/bin/activate
echo "✓ Virtual environment created and activated"
echo ""

# ============================================================================
# Python Package Installation (from README - exactly as specified)
# ============================================================================
echo "[3/5] Installing Python packages..."
pip install --upgrade pip setuptools wheel && \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
pip install datasets ujson tqdm nltk numpy scipy scikit-learn && \
pip install sentence-transformers umap-learn && \
pip install fastapi uvicorn pydantic tensorboard regex python-multipart && \
pip install transformers huggingface-hub
echo "✓ Python packages installed"
echo ""

# ============================================================================
# Verify Package Installation (from README)
# ============================================================================
echo "[4/5] Verifying package installation..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from sentence_transformers import SentenceTransformer; print('✓ sentence-transformers')"
python -c "import umap; print('✓ umap')"
python -c "from fastapi import FastAPI; print('✓ fastapi')"
echo ""

# ============================================================================
# GPU Health Check
# ============================================================================
echo "[5/5] GPU Health Check..."
python - <<'PY'
import torch
print("count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
   try:
       torch.cuda.set_device(i); torch.cuda.synchronize()
       print(f"GPU {i}: OK")
   except Exception as e:
       print(f"GPU {i}: FAIL -> {e}")
PY
echo ""

# ============================================================================
# Ready for training
# ============================================================================
echo "================================================================================"
echo "✓ Setup Complete!"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Configuration:"
echo "  • NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "  • Python venv: .venv"
echo ""
echo "⚠️  IMPORTANT: To activate the environment in your current shell, run:"
echo ""
echo "   source .venv/bin/activate"
echo "   export NANOCHAT_BASE_DIR=/workspace/nanochat"
echo ""
echo "Then you can run:"
echo "  bash kat_speedrun.sh"
echo ""
echo "================================================================================"
