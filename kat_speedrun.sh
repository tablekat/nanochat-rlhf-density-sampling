#!/bin/bash

# KAT Speedrun: Density-Aware GRPO with Hypothesis Testing
# 
# Complete pipeline for comparing reward models trained with different sampling strategies.
# This script trains two reward models:
#   1. Regular RM with uniform sampling (baseline)
#   2. Density-Aware RM with inverse density sampling (commented out - for later experiments)
#
# Then runs GRPO training separately with each RM, both using uniform sampling during GRPO.
# This enables testing the hypothesis that density-aware RM training improves policy learning.
#
# On first run: Only regular RM + GRPO (density sections commented out)
# When ready to test: Uncomment sections to enable density-aware RM and comparisons
#
# Expected runtime: ~1-2 days on 8xH100 GPU node at $3/GPU/hour
#
# Usage:
#   bash kat_speedrun.sh
#   WANDB_RUN=density_experiment bash kat_speedrun.sh
#   screen -L -Logfile kat_speedrun.log -S kat_speedrun bash kat_speedrun.sh

set -e  # Exit on error

export OMP_NUM_THREADS=1
# export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_BASE_DIR="/workspace/nanochat"
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
echo "[2/6] Training Tokenizer..."

# Install Rust / Cargo (idempotent)
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
echo ""

# Wait for dataset download to complete (started during tokenizer training)
if [ ! -z "$DATASET_DOWNLOAD_PID" ]; then
    echo "Waiting for dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID
fi
echo ""

# =============================================================================
# Stage 2: Pretrain Base Model
echo "[3/6] Pretraining base model (depth=20)..."

# Download eval bundle if needed
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    echo "  Downloading evaluation bundle..."
    EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# Training scripts
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval
echo "✓ Base model pretrained"
echo ""

# =============================================================================
# Stage 3: Mid-training (conversation special tokens, etc.)
echo "[4/6] Mid-training..."

# Download identity conversations
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "  Downloading identity conversations..."
    curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
echo "✓ Mid-training complete"
echo ""

# =============================================================================
# Stage 4: Supervised Fine-Tuning (SFT)
echo "[5/6] Supervised Fine-Tuning..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
echo "✓ SFT complete"
echo ""

# =============================================================================
# Stage 5: Pairwise Preference Data Pipeline
echo "[6/9] Preparing pairwise preference data for GRPO..."

echo "  [6a/9] Downloading preference pairs..."
python -m scripts.kat_download_pairs --only hh  # Start with just HH for speed

echo "  [6b/9] Making conversation prefix-only set..."
python -m scripts.kat_make_prefixes

echo ""
echo "  [6c/9] Training Reward Models..."
echo ""

# =============================================================================
# RM Option 1: Regular Reward Model (uniform sampling - baseline)
# Output: $NANOCHAT_BASE_DIR/rm_checkpoints/uniform/d20/model_*.pt
echo "  Training RM #1: Regular (uniform sampling)..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.kat_train_rm \
    -- \
    --rm_source=rm \
    --max_steps=1000 \
    --run=${WANDB_RUN}_rm_uniform
echo "  ✓ RM #1 (uniform) trained"
echo ""

echo "  Training RM #2: Dual Likert scorer (uniform sampling)..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.kat_train_rm_dual \
    -- \
    --rm_source=rm \
    --max_steps=1000 \
    --run=${WANDB_RUN}_rm_dual
echo "  ✓ RM #2 (dual) trained"
echo ""

# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rm

echo "✓ Reward Models training complete"
echo ""

# =============================================================================
# Stage 6 & RM #2: Density-Aware Setup (commented out for first run)
# 
# To enable density-aware experiments, uncomment the sections below IN THIS ORDER:
#   1. First uncomment Stage 6 (Offline Embeddings)
#   2. Then uncomment RM #2 (which depends on Stage 6)
#   3. Then uncomment Stage 8 (GRPO with density RM)
# ===

# Stage 6: Offline Embeddings Computation
# Output: $NANOCHAT_BASE_DIR/data/embeddings_offline/density_weights.npy
# echo "[7/9] Computing Offline Embeddings for Density-Aware Sampling..."
# echo ""
# echo "  This precomputes prompt embeddings and density weights."
# echo "  Required for RM #2 (density-aware) training."
# echo ""
# 
# EMBEDDINGS_DIR="$NANOCHAT_BASE_DIR/data/embeddings_offline"
# 
# echo "  Computing embeddings (~5-10 minutes)..."
# python -m scripts.kat_compute_embeddings_offline \
#     --base_model_source base \
#     --batch_size 8 \
#     --k 10 \
#     --output_dir $EMBEDDINGS_DIR
# echo "  ✓ Embeddings computed successfully"
# echo ""
# 
# # RM Option 2: Density-Aware Reward Model (depends on Stage 6 embeddings above)
# # Output: $NANOCHAT_BASE_DIR/rm_checkpoints/density/d20/model_*.pt
# echo "  Training RM #2: Density-Aware (inverse density sampling)..."
# EMBEDDINGS_DIR="$NANOCHAT_BASE_DIR/data/embeddings_offline"
# 
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.kat_train_rm \
#     --rm_source rm_density \
#     --max_steps=1000
# echo "  ✓ RM #2 (density) trained"
# echo ""

echo ""

# =============================================================================
# Stage 7: GRPO Training - with Regular RM (uniform sampling on both RM and GRPO)
# Output: $NANOCHAT_BASE_DIR/grpo_checkpoints/uniform/d20/model_*.pt
echo "================================================================"
echo "EXPERIMENT 1: GRPO with Regular RM (uniform sampling)"
echo "  Policy learns from RM trained with uniform sampling"
echo "  Policy training also uses uniform sampling"
echo "================================================================"
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.kat_train_grpo \
    -- \
    --rm_source=rm \
    --grpo_source=grpo \
    --max_steps=5000 \
    --learning_rate=1e-5 \
    --beta=0.1 \
    --run=${WANDB_RUN}_grpo_uniform

echo "✓ GRPO with regular RM complete"
echo ""

# =============================================================================
# Stage 8: GRPO Training - with Density-Aware RM (commented out for first run)
# Output: $NANOCHAT_BASE_DIR/grpo_checkpoints/density/d20/model_*.pt
# Uncomment this section to compare with density-aware RM
# ===
# echo "================================================================"
# echo "EXPERIMENT 2: GRPO with Density-Aware RM (uniform GRPO sampling)"
# echo "  Policy learns from RM trained with density-aware sampling"
# echo "  Policy training uses uniform sampling (no density weighting in GRPO)"
# echo "================================================================"
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.kat_train_grpo \
#     --rm_source rm_density \
#     --grpo_source grpo_density \
#     --max_steps=5000 \
#     --learning_rate=1e-5 \
#     --beta=0.1 \
#     --run=${WANDB_RUN}_grpo_density
# 
# echo "✓ GRPO with density RM complete"
# echo ""
# ===

# =============================================================================
# Stage 9: Evaluation - Diversity Metrics
echo "================================================================"
echo "EVALUATION: Testing Output Quality"
echo "================================================================"
echo ""
echo "Evaluating reward models and diversity..."
python -m scripts.kat_eval_rm \
    --rm-sources rm${ENABLE_DENSITY_RM:+,rm_density} \
    --max-examples 5000 \
    --output-json $NANOCHAT_BASE_DIR/metrics/rm_eval.json || true

python -m scripts.kat_eval_diversity \
    --density_model_source grpo${ENABLE_DENSITY_RM:+_density} \
    --baseline_model_source grpo \
    --output_report $NANOCHAT_BASE_DIR/metrics/diversity_report.md || true

echo ""
echo "RM metrics: $NANOCHAT_BASE_DIR/metrics/rm_eval.json"
echo "Diversity report: $NANOCHAT_BASE_DIR/metrics/diversity_report.md"
echo ""

# =============================================================================
# Final Summary
echo "================================================================================"
echo "KAT Speedrun Complete!"
echo "End time: $(date)"
echo "================================================================================"
echo ""
echo "Pipeline stages:"
echo "  ✓ [1/9] Environment setup"
echo "  ✓ [2/9] Tokenizer training"
echo "  ✓ [3/9] Base model pretraining"
echo "  ✓ [4/9] Mid-training"
echo "  ✓ [5/9] Supervised Fine-Tuning"
echo "  ✓ [6/9] Preference data pipeline (2 RMs: regular + density-aware)"
echo "  ✓ [7/9] GRPO training with regular RM (uniform sampling)"
echo "  ✓ [8/9] GRPO training with density-aware RM (commented out)"
echo "  ✓ [9/9] Diversity evaluation"
echo ""
echo "Key outputs:"
echo "  ✓ \$NANOCHAT_BASE_DIR/rm_checkpoints/uniform/d20/"
echo "      (Regular RM with uniform sampling, timestamped)"
echo "  ✓ \$NANOCHAT_BASE_DIR/grpo_checkpoints/uniform/d20/"
echo "      (GRPO with regular RM, timestamped)"
echo "  ✓ $NANOCHAT_BASE_DIR/metrics/rm_eval.json"
echo "      (Reward model accuracy & margin metrics)"
echo "  ✓ $NANOCHAT_BASE_DIR/metrics/diversity_report.md"
echo "      (Diversity evaluation comparing policies)"
echo ""
echo "To enable density-based RM and comparison:"
echo "  1. Uncomment Stage 6 (Offline Embeddings) in this script"
echo "  2. Uncomment RM #2 section (Density-Aware RM training)"
echo "  3. Uncomment Stage 8 (GRPO with Density-Aware RM)"
echo "  4. Uncomment evaluation section"
echo "  5. Re-run: bash kat_speedrun.sh"
echo ""
echo "Next steps:"
echo "  1. Review diversity metrics and training logs"
echo "  2. Interactive chat: python -m scripts.chat_cli --ckpt_path \$NANOCHAT_BASE_DIR/grpo_checkpoints/uniform/d20/model_*.pt"
echo "  3. Web UI: python -m scripts.chat_web --ckpt_path \$NANOCHAT_BASE_DIR/grpo_checkpoints/uniform/d20/model_*.pt"
echo ""
echo "================================================================================"
echo "Optional: 3D Embedding Visualization"
echo "================================================================================"
echo ""
echo "To visualize the embedding space in 3D:"
echo ""
echo "  # 1. Generate 3D embedding visualization"
echo "  python -m scripts.kat_viz_embeddings"
echo ""
echo "  # 2. Start the web server with visualization"
echo "  python -m scripts.chat_web"
echo ""
echo "  # 3. Open browser to:"
echo "  http://localhost:8000/viz"
echo ""
echo "This shows:"
echo "  - All 28k unique prefixes in 3D space (UMAP projection)"
echo "  - Point colors by source dataset (hh-rlhf/ultrafeedback/stack-exchange)"
echo "  - Point sizes by inverse density (rare prompts = bigger)"
echo "  - Interactive: Hover to see prefix, click to select"
echo ""
