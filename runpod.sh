#!/bin/bash

# ============================================================================
# RunPod Setup Script for nanochat-rlhf-density-sampling
# ============================================================================
# This script matches the README setup flow:
# 1. Install system packages
# 2. Clone the repo
# 3. Create a screen session named "speedrun"
# 4. Inside the screen: set up venv and install packages
# After completion, you'll be inside the screen session ready to run kat_speedrun.sh
# ============================================================================

set -e  # Exit on error

# ============================================================================
# System Package Installation (from README)
# ============================================================================
echo "================================================================================"
echo "RunPod Setup for nanochat-rlhf-density-sampling"
echo "================================================================================"
echo "Installing system packages..."
sudo apt-get update -y && sudo apt-get install -y git python3-venv screen
echo "✓ System packages installed"
echo ""

# ============================================================================
# Clone the Repository
# ============================================================================
echo "Cloning repository..."
git clone https://github.com/tablekat/nanochat-rlhf-density-sampling.git
cd nanochat-rlhf-density-sampling
echo "✓ Repository cloned"
echo ""

# ============================================================================
# Create screen session and run setup inside it
# ============================================================================
# Create a script that will run inside the screen session
SETUP_SCRIPT=$(cat <<'SETUP_EOF'
#!/bin/bash
set -e

# ============================================================================
# Global Configuration - Save to /workspace for persistent storage
# ============================================================================
# export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/workspace/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

echo "================================================================================"
echo "Inside screen session: speedrun"
echo "================================================================================"
echo "Start time: $(date)"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo ""

# ============================================================================
# Python Virtual Environment Setup (from README)
# ============================================================================
echo "[1/5] Setting up Python virtual environment..."
python3 -m venv .venv && source .venv/bin/activate
echo "✓ Virtual environment created and activated"
echo ""

# ============================================================================
# Python Package Installation (from README - exactly as specified)
# ============================================================================
echo "[2/5] Installing Python packages..."
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
echo "[3/5] Verifying package installation..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from sentence_transformers import SentenceTransformer; print('✓ sentence-transformers')"
python -c "import umap; print('✓ umap')"
python -c "from fastapi import FastAPI; print('✓ fastapi')"
echo ""

# ============================================================================
# GPU Health Check
# ============================================================================
echo "[4/5] GPU Health Check..."
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
echo "[5/5] Setup Complete!"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Configuration:"
echo "  • NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "  • Python venv: .venv (currently activated)"
echo "  • Current directory: $(pwd)"
echo ""
echo "You are now inside the screen session 'speedrun'"
echo ""
echo "Next step:"
echo "  bash kat_speedrun.sh"
echo ""
echo "Screen commands:"
echo "  • Detach from screen: Ctrl-A then D"
echo "  • Reattach to screen: screen -r speedrun"
echo ""
echo "================================================================================"
SETUP_EOF
)

# Write the setup script to a temp file
echo "$SETUP_SCRIPT" > /tmp/runpod_setup.sh
chmod +x /tmp/runpod_setup.sh

# Launch screen session with the setup script
screen -S speedrun -d -m bash /tmp/runpod_setup.sh

# Give screen a moment to start
sleep 2

# Attach to the screen session
screen -r speedrun
