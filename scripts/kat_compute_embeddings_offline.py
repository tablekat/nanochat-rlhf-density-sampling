#!/usr/bin/env python3
"""
Offline Embedding Computation for Density-Aware GRPO

This script precomputes embeddings for all prompts and calculates local density
weights. This is run once as a preprocessing step and outputs files that can be
loaded quickly during training.

Usage:
    python -m scripts.kat_compute_embeddings_offline --base_model_source base --output_dir data/
    python -m scripts.kat_compute_embeddings_offline --base_model_source base --k 10 --batch_size 16

Output files (in output_dir):
    - embeddings.npy          (28k x 50304 float32 array)
    - embeddings_metadata.json (statistics and parameters)
    - density_weights.npy      (28k float32 array of density weights)
    - prompts_list.json        (list of prompts in same order as embeddings)

These files are then loaded by kat_train_grpo.py with --use_precomputed_embeddings flag.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from nanochat.checkpoint_manager import load_model, get_base_dir
from nanochat.tokenizer import get_tokenizer


def load_prompts(prompts_path):
    """Load prompt list from JSONL file."""
    prompts = []
    with open(prompts_path, 'r') as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                if 'prompt' not in obj:
                    print(f"⚠️  Line {line_no}: missing 'prompt' field, skipping")
                    continue
                prompts.append(obj['prompt'])
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_no}: invalid JSON ({e}), skipping")
                continue
    
    if len(prompts) == 0:
        raise RuntimeError(f"No valid prompts found in {prompts_path}")
    
    return prompts


def compute_embeddings(prompts, base_model, tokenizer, device, batch_size=8):
    """
    Compute embeddings for all prompts using the base model.
    
    Args:
        prompts: List of prompt strings
        base_model: GPT model loaded on device
        tokenizer: Tokenizer instance
        device: Device to run on (cuda/cpu)
        batch_size: Number of prompts per batch
    
    Returns:
        embeddings: (n_prompts, vocab_size) float32 numpy array
    """
    print(f"Computing embeddings for {len(prompts)} prompts...")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    
    embeddings = []
    base_model.eval()
    
    with torch.no_grad():
        for batch_idx in range(0, len(prompts), batch_size):
            if batch_idx % (batch_size * 10) == 0:
                pct = 100.0 * batch_idx / len(prompts)
                print(f"  [{pct:5.1f}%] Processing batch {batch_idx // batch_size}...", flush=True)
            
            batch_prompts = prompts[batch_idx:batch_idx + batch_size]
            
            # Tokenize
            try:
                encoded = [tokenizer.encode(p) for p in batch_prompts]
            except Exception as e:
                print(f"⚠️  Error encoding batch starting at {batch_idx}: {e}")
                continue
            
            # Pad to same length within batch
            max_len = max(len(e) for e in encoded) if encoded else 1
            max_len = min(max_len, 512)  # Limit to 512 tokens
            
            pad_token_id = tokenizer.encode_special("<|assistant_end|>")[0] if tokenizer.encode_special("<|assistant_end|>") else 0
            
            input_ids = []
            for e in encoded:
                padded = e + [pad_token_id] * (max_len - len(e))
                input_ids.append(padded[:512])
            
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
            
            # Forward pass - get logits
            try:
                outputs = base_model(input_tensor)  # [batch, seq_len, vocab_size]
            except Exception as e:
                print(f"⚠️  Error in forward pass for batch starting at {batch_idx}: {e}")
                continue
            
            # Average pool over sequence dimension
            batch_emb = outputs.mean(dim=1)  # [batch, vocab_size]
            embeddings.append(batch_emb.cpu().numpy())
    
    # Concatenate all batches
    if not embeddings:
        raise RuntimeError("No embeddings computed! Check input data and model.")
    
    embeddings = np.concatenate(embeddings, axis=0)  # [n_prompts, vocab_size]
    
    # L2 normalize to unit vectors
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    print(f"✓ Computed {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    
    return embeddings


def compute_density_weights(embeddings, k=10):
    """
    Compute local density weights using k-NN.
    
    Args:
        embeddings: (n_prompts, embedding_dim) array
        k: Number of nearest neighbors
    
    Returns:
        weights: (n_prompts,) array of density weights summing to 1
    """
    n_prompts = embeddings.shape[0]
    
    if n_prompts < k + 1:
        print(f"⚠️  Only {n_prompts} prompts but k={k}, using uniform weights")
        return np.ones(n_prompts) / n_prompts
    
    print(f"Computing local density using k-NN (k={k})...")
    
    # Build k-NN index
    nbrs = NearestNeighbors(n_neighbors=k + 1)  # +1 for self
    nbrs.fit(embeddings)
    
    # Get distances to k nearest neighbors
    distances, indices = nbrs.kneighbors(embeddings)
    
    # Local density = average distance to k nearest neighbors (exclude self at column 0)
    local_densities = np.mean(distances[:, 1:], axis=1)
    
    # Avoid division by zero
    local_densities = np.maximum(local_densities, 1e-6)
    
    # Inverse weight: rare prompts (large distances) get high weights
    weights = 1.0 / local_densities
    
    # Normalize to valid probability distribution
    weights = weights / weights.sum()
    
    print(f"✓ Density weights computed")
    print(f"  Min weight: {weights.min():.6f}")
    print(f"  Max weight: {weights.max():.6f}")
    print(f"  Mean weight: {weights.mean():.6f}")
    print(f"  Std weight: {weights.std():.6f}")
    
    return weights


def save_embeddings(embeddings, weights, prompts, output_dir, model_config):
    """Save embeddings, weights, and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    weights_path = os.path.join(output_dir, "density_weights.npy")
    prompts_path = os.path.join(output_dir, "prompts_list.json")
    metadata_path = os.path.join(output_dir, "embeddings_metadata.json")
    
    print(f"\nSaving outputs to {output_dir}...")
    
    # Save embeddings
    np.save(embeddings_path, embeddings.astype(np.float32))
    print(f"✓ Saved embeddings ({embeddings.shape}) to {embeddings_path}")
    
    # Save weights
    np.save(weights_path, weights.astype(np.float32))
    print(f"✓ Saved density weights ({weights.shape}) to {weights_path}")
    
    # Save prompts list
    with open(prompts_path, 'w') as f:
        json.dump(prompts, f)
    print(f"✓ Saved {len(prompts)} prompts to {prompts_path}")
    
    # Save metadata
    metadata = {
        "n_prompts": len(prompts),
        "embedding_dim": embeddings.shape[1],
        "embedding_dtype": "float32",
        "weights_dtype": "float32",
        "model_vocab_size": model_config.get("vocab_size", 50304),
        "model_n_embd": model_config.get("n_embd", 768),
        "normalization": "L2 unit vectors",
        "weights_min": float(weights.min()),
        "weights_max": float(weights.max()),
        "weights_mean": float(weights.mean()),
        "weights_std": float(weights.std()),
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")
    
    print(f"\n✅ All files saved successfully")
    print(f"\nTo use these embeddings in training:")
    print(f"  torchrun --nproc_per_node=8 -m scripts.kat_train_grpo \\")
    print(f"    --use_precomputed_embeddings \\")
    print(f"    --embeddings_dir {output_dir} \\")
    print(f"    --max_steps 5000")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute embeddings and density weights for GRPO training"
    )
    parser.add_argument(
        "--base_model_source",
        default="base",
        help="Model checkpoint source (base|mid|sft|grpo|rm)",
    )
    parser.add_argument(
        "--prompts_path",
        default=None,
        help="Path to prompts_all.jsonl (auto-detected if not provided)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for embeddings (auto-detected if not provided)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors for density computation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing prompts through model",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda|cpu)",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = get_base_dir()
    
    if args.prompts_path is None:
        args.prompts_path = os.path.join(base_dir, "data", "prompts_all.jsonl")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(base_dir, "data", "embeddings_offline")
    
    # Validate inputs
    if not os.path.exists(args.prompts_path):
        print(f"❌ Error: Prompts file not found: {args.prompts_path}")
        print(f"   Run kat_make_prompts.py first to generate prompts_all.jsonl")
        sys.exit(1)
    
    print("=" * 70)
    print("OFFLINE EMBEDDING COMPUTATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Prompts file:      {args.prompts_path}")
    print(f"  Output directory:  {args.output_dir}")
    print(f"  Model source:      {args.base_model_source}")
    print(f"  k (neighbors):     {args.k}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Device:            {args.device}")
    print()
    
    # Load components
    print("Loading model and tokenizer...")
    try:
        model, tokenizer, meta = load_model(
            source=args.base_model_source,
            device=args.device,
            phase="eval",
        )
        model.eval()
        print("✓ Model and tokenizer loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Load prompts
    print("Loading prompts...")
    try:
        prompts = load_prompts(args.prompts_path)
        print(f"✓ Loaded {len(prompts)} prompts")
    except Exception as e:
        print(f"❌ Error loading prompts: {e}")
        sys.exit(1)
    
    # Compute embeddings
    print()
    try:
        embeddings = compute_embeddings(
            prompts,
            model,
            tokenizer,
            device=args.device,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"❌ Error computing embeddings: {e}")
        sys.exit(1)
    
    # Compute density weights
    print()
    try:
        weights = compute_density_weights(embeddings, k=args.k)
    except Exception as e:
        print(f"❌ Error computing density weights: {e}")
        sys.exit(1)
    
    # Save outputs
    try:
        model_config = {
            "vocab_size": meta.get("vocab_size", 50304) if meta else 50304,
            "n_embd": meta.get("n_embd", 768) if meta else 768,
        }
        save_embeddings(embeddings, weights, prompts, args.output_dir, model_config)
    except Exception as e:
        print(f"❌ Error saving outputs: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ EMBEDDING COMPUTATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
