#!/bin/bash

# KAT Speedrun: Density-Aware GRPO with Hypothesis Testing
# 
# Complete pipeline testing mode collapse reduction via density-aware sampling.
# This script trains two models:
#   1. GRPO with density-aware sampling (main experiment)
#   2. GRPO without density sampling (control/baseline)
# Then evaluates both on diversity metrics.
#
# Expected runtime: ~1-2 days on 8xH100 GPU node at $3/GPU/hour
#
# Usage:
#   bash kat_speedrun.sh
#   WANDB_RUN=density_experiment bash kat_speedrun.sh
#   screen -L -Logfile kat_speedrun.log -S kat_speedrun bash kat_speedrun.sh

set -e  # Exit on error

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# wandb logging setup (optional)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=density_experiment
fi

echo "================================================================================"
echo "KAT Speedrun: Density-Aware GRPO + Hypothesis Testing"
echo "================================================================================"
echo "Start time: $(date)"
echo "WANDB_RUN: $WANDB_RUN"
echo ""

# =============================================================================
# GPU Detection and Setup
echo "[GPU Detection] Detecting available GPUs..."
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$NUM_GPUS" = "0" ]; then
    echo "⚠️  No GPUs detected! Running on CPU (very slow)"
    NPROC_PER_NODE=1
else
    NPROC_PER_NODE=$NUM_GPUS
    echo "✓ Found $NUM_GPUS GPU(s), using all of them"
fi

# Allow override via environment variable
if [ ! -z "$NPROC_PER_NODE" ]; then
    echo "  Using NPROC_PER_NODE=$NPROC_PER_NODE (from environment or detection)"
fi

echo ""

# =============================================================================
# Python venv setup with uv
echo "[1/6] Setting up Python environment..."
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
echo "✓ Environment ready"
echo ""

# =============================================================================
# Stage 1: Train Tokenizer (if needed)
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "[2/6] Training Tokenizer..."
    
    # Install Rust / Cargo
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    
    # Build rustbpe tokenizer
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
    
    # Download pretraining data
    python -m nanochat.dataset -n 8
    python -m nanochat.dataset -n 240 &
    DATASET_DOWNLOAD_PID=$!
    
    # Train tokenizer
    python -m scripts.tok_train --max_chars=2000000000
    python -m scripts.tok_eval
    
    echo "✓ Tokenizer trained"
else
    echo "[2/6] Tokenizer already exists, skipping..."
fi
echo ""

# Wait for dataset download to complete (started during tokenizer training)
if [ ! -z "$DATASET_DOWNLOAD_PID" ]; then
    echo "Waiting for dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID
fi
echo ""

# =============================================================================
# Stage 2: Pretrain Base Model
if [ ! -f "$NANOCHAT_BASE_DIR/base_checkpoints/d20/model_000000.pt" ]; then
    echo "[3/6] Pretraining base model (depth=20)..."
    
    # Download eval bundle if needed
    if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
        EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
        curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
        unzip -q eval_bundle.zip
        rm eval_bundle.zip
        mv eval_bundle $NANOCHAT_BASE_DIR
    fi
    
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval
    echo "✓ Base model pretrained"
else
    echo "[3/6] Base model already exists, skipping..."
fi
echo ""

# =============================================================================
# Stage 3: Mid-training (conversation special tokens, etc.)
if [ ! -f "$NANOCHAT_BASE_DIR/mid_checkpoints/d20/model_000000.pt" ]; then
    echo "[4/6] Mid-training..."
    
    # Download identity conversations
    if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
        curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    fi
    
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
    echo "✓ Mid-training complete"
else
    echo "[4/6] Mid-training checkpoint already exists, skipping..."
fi
echo ""

# =============================================================================
# Stage 4: Supervised Fine-Tuning (SFT)
if [ ! -f "$NANOCHAT_BASE_DIR/chatsft_checkpoints/d20/model_000000.pt" ]; then
    echo "[5/6] Supervised Fine-Tuning..."
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
    echo "✓ SFT complete"
else
    echo "[5/6] SFT checkpoint already exists, skipping..."
fi
echo ""

# =============================================================================
# Stage 5: Pairwise Preference Data Pipeline
echo "[6/8] Preparing pairwise preference data for GRPO..."

echo "  [6a/8] Downloading preference pairs..."
python -m scripts.kat_download_pairs --only hh  # Start with just HH for speed

echo "  [6b/8] Deduplicating prompts..."
python -m scripts.kat_make_prompts

echo "  [6c/8] Training Reward Model..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.kat_train_rm -- --max_steps=1000

echo "✓ Preference data pipeline complete"
echo ""

# =============================================================================
# Stage 6: Offline Embeddings Computation (NEW!)
echo "[7/8] Computing Offline Embeddings for Density-Aware Sampling..."
echo ""
echo "  This precomputes prompt embeddings and density weights."
echo "  ⚡ GRPO training will start instantly (<1s) instead of waiting 3+ minutes"
echo ""

EMBEDDINGS_DIR="$NANOCHAT_BASE_DIR/data/embeddings_offline"

# Check if embeddings already exist
if [ -f "$EMBEDDINGS_DIR/density_weights.npy" ]; then
    echo "  ✓ Embeddings already precomputed, skipping..."
else
    echo "  Computing embeddings (~5-10 minutes)..."
    python -m scripts.kat_compute_embeddings_offline \
        --base_model_source base \
        --batch_size 8 \
        --k 10 \
        --output_dir $EMBEDDINGS_DIR
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Embeddings computed successfully"
    else
        echo "  ⚠️  Embedding computation failed, will fall back to online during training"
    fi
fi
echo ""

# =============================================================================
# Stage 7: GRPO Training - Main Experiment (with density sampling + offline embeddings)
echo "================================================================"
echo "MAIN EXPERIMENT: GRPO with Density-Aware Sampling"
echo "  Using offline precomputed embeddings (instant startup!)"
echo "================================================================"
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.kat_train_grpo \
    --use_precomputed_embeddings \
    --embeddings_dir $EMBEDDINGS_DIR \
    --max_steps=5000 \
    --learning_rate=1e-5 \
    --beta=0.1 \
    --density_aware=True \
    --density_k=10 \
    --out_dir outs/grpo_density

echo "✓ GRPO with density sampling complete"
echo ""

# =============================================================================
# Stage 8: GRPO Training - Baseline (without density sampling + offline embeddings)
echo "================================================================"
echo "BASELINE: GRPO without Density Sampling (uniform sampling)"
echo "  Using offline precomputed embeddings (instant startup!)"
echo "================================================================"
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.kat_train_grpo \
    --use_precomputed_embeddings \
    --embeddings_dir $EMBEDDINGS_DIR \
    --max_steps=5000 \
    --learning_rate=1e-5 \
    --beta=0.1 \
    --density_aware=False \
    --out_dir outs/grpo_baseline

echo "✓ GRPO baseline complete"
echo ""

# =============================================================================
# Stage 9: Evaluation - Diversity Metrics
echo "================================================================"
echo "EVALUATION: Testing Hypothesis"
echo "================================================================"
echo ""
echo "Evaluating outputs for mode collapse indicators..."
python -m scripts.kat_eval_diversity \
    --density_model_path outs/grpo_density/ckpt.pt \
    --baseline_model_path outs/grpo_baseline/ckpt.pt \
    --output_report .cache/diversity_report.md

echo ""
echo "Report saved to: .cache/diversity_report.md"
echo ""

# =============================================================================
# Final Summary
echo "================================================================================"
echo "KAT Speedrun Complete!"
echo "End time: $(date)"
echo "================================================================================"
echo ""
echo "Pipeline stages:"
echo "  ✓ [1/8] Environment setup"
echo "  ✓ [2/8] Tokenizer training"
echo "  ✓ [3/8] Base model pretraining"
echo "  ✓ [4/8] Mid-training"
echo "  ✓ [5/8] Supervised Fine-Tuning"
echo "  ✓ [6/8] Preference data pipeline"
echo "  ✓ [7/8] Offline embeddings computation"
echo "  ✓ [8/8] GRPO training + evaluation"
echo ""
echo "Key outputs:"
echo "  ✓ outs/grpo_density/ckpt.pt       (Main experiment with density sampling)"
echo "  ✓ outs/grpo_baseline/ckpt.pt      (Baseline without density sampling)"
echo "  ✓ .cache/diversity_report.md      (Diversity evaluation results)"
echo ""
echo "Offline embeddings:"
echo "  ✓ $EMBEDDINGS_DIR/embeddings.npy"
echo "  ✓ $EMBEDDINGS_DIR/density_weights.npy"
echo "  ✓ $EMBEDDINGS_DIR/embeddings_metadata.json"
echo ""
echo "Training startup times:"
echo "  Before: 240s (4 minutes) - compute embeddings online"
echo "  After:  <1s (instant!)   - load precomputed embeddings"
echo ""
echo "Next steps:"
echo "  1. Review .cache/diversity_report.md for hypothesis validation"
echo "  2. Interactive chat: python -m scripts.chat_cli --ckpt_path outs/grpo_density/ckpt.pt"
echo "  3. Web UI: python -m scripts.chat_web --ckpt_path outs/grpo_density/ckpt.pt"
echo ""
