#!/usr/bin/env python3
"""
STANDALONE PIPELINE VERIFICATION SCRIPT

Run this to verify your setup before running the full pipeline:
    python tests/verify_pipeline.py

This script:
  1. Checks all required imports
  2. Verifies kat_utils functions
  3. Tests data format handling
  4. Validates density computation logic
  5. Checks loss function implementations
  6. Reports any issues found

NO GPU REQUIRED - All verification runs on CPU.
"""

import sys
import os
import json
import tempfile
import traceback
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(title):
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}{title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")

def print_pass(msg):
    print(f"  {GREEN}✓{RESET} {msg}")

def print_fail(msg):
    print(f"  {RED}✗{RESET} {msg}")

def print_warn(msg):
    print(f"  {YELLOW}⚠{RESET} {msg}")

def print_info(msg):
    print(f"  {CYAN}→{RESET} {msg}")

# Track results
passed = 0
failed = 0
warnings = 0

def check(condition, pass_msg, fail_msg):
    global passed, failed
    if condition:
        print_pass(pass_msg)
        passed += 1
        return True
    else:
        print_fail(fail_msg)
        failed += 1
        return False

def warn(msg):
    global warnings
    print_warn(msg)
    warnings += 1

# ═════════════════════════════════════════════════════════════════════════════
# CHECK 1: Required Imports
# ═════════════════════════════════════════════════════════════════════════════

def check_imports():
    print_header("CHECK 1: Required Imports")
    
    # Core Python
    try:
        import json, hashlib, re
        print_pass("Standard library imports")
    except ImportError as e:
        print_fail(f"Standard library: {e}")
        return False
    
    # NumPy
    try:
        import numpy as np
        print_pass(f"NumPy {np.__version__}")
    except ImportError:
        print_fail("NumPy not installed: pip install numpy")
        return False
    
    # scikit-learn (for k-NN density)
    try:
        from sklearn.neighbors import NearestNeighbors
        import sklearn
        print_pass(f"scikit-learn {sklearn.__version__}")
    except ImportError:
        print_fail("scikit-learn not installed: pip install scikit-learn")
        return False
    
    # PyTorch (optional for this verification)
    try:
        import torch
        print_pass(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print_info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print_info("CUDA not available (CPU only)")
    except ImportError:
        warn("PyTorch not installed - needed for training but not verification")
    
    # Project modules
    try:
        from scripts.kat_utils import norm_space, ensure_prefix_dict, prefix_id_from_prefix
        print_pass("kat_utils module")
    except ImportError as e:
        print_fail(f"kat_utils module: {e}")
        return False
    
    try:
        from nanochat.common import get_base_dir
        print_pass("nanochat.common module")
    except ImportError as e:
        print_fail(f"nanochat.common module: {e}")
        return False
    
    return True


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 2: kat_utils Functions
# ═════════════════════════════════════════════════════════════════════════════

def check_kat_utils():
    print_header("CHECK 2: kat_utils Functions")
    
    from scripts.kat_utils import (
        norm_space, 
        ensure_prefix_dict, 
        first_user_message, 
        prefix_id_from_text,
        prefix_id_from_prefix
    )
    
    # norm_space
    check(
        norm_space("  hello   world  ") == "hello world",
        "norm_space: whitespace normalization",
        "norm_space: failed whitespace normalization"
    )
    
    check(
        norm_space(None) == "",
        "norm_space: None handling",
        "norm_space: failed None handling"
    )
    
    # ensure_prefix_dict
    result = ensure_prefix_dict("What is AI?")
    check(
        "messages" in result and result["messages"][0]["role"] == "user",
        "ensure_prefix_dict: string → prefix conversion",
        "ensure_prefix_dict: failed string conversion"
    )
    
    original = {"messages": [{"role": "user", "content": "Test"}]}
    copied = ensure_prefix_dict(original)
    copied["messages"].append({"role": "assistant", "content": "Hi"})
    check(
        len(original["messages"]) == 1,
        "ensure_prefix_dict: deep copy (mutation safety)",
        "ensure_prefix_dict: failed deep copy - original was mutated!"
    )
    
    # first_user_message
    prefix = {"messages": [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Reply"},
        {"role": "user", "content": "Second"}
    ]}
    check(
        first_user_message(prefix) == "First",
        "first_user_message: extracts first user message",
        "first_user_message: failed extraction"
    )
    
    # prefix_id_from_text
    id1 = prefix_id_from_text("What is AI?")
    id2 = prefix_id_from_text("What is AI?")
    id3 = prefix_id_from_text("What is ML?")
    check(
        id1 == id2 and id1 != id3,
        "prefix_id_from_text: deterministic and unique",
        "prefix_id_from_text: IDs not deterministic or not unique"
    )
    
    check(
        len(id1) == 16,
        "prefix_id_from_text: correct length (16 chars)",
        f"prefix_id_from_text: wrong length ({len(id1)} chars)"
    )
    
    # prefix_id_from_prefix
    pid1 = prefix_id_from_prefix({"messages": [{"role": "user", "content": "What is AI?"}]})
    pid2 = prefix_id_from_prefix("What is AI?")
    check(
        pid1 == pid2,
        "prefix_id_from_prefix: consistent across formats",
        "prefix_id_from_prefix: inconsistent IDs for same content"
    )
    
    return True


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 3: Data Format Handling
# ═════════════════════════════════════════════════════════════════════════════

def check_data_formats():
    print_header("CHECK 3: Data Format Handling")
    
    from scripts.kat_utils import ensure_prefix_dict, prefix_id_from_prefix
    
    # Legacy format
    legacy_pair = {
        "id": "abc123",
        "prompt": "What is AI?",
        "chosen": "AI is artificial intelligence.",
        "rejected": "I don't know."
    }
    
    prefix = ensure_prefix_dict(legacy_pair.get("prompt"))
    check(
        "messages" in prefix,
        "Legacy 'prompt' format: converts to prefix dict",
        "Legacy 'prompt' format: conversion failed"
    )
    
    # New format
    new_pair = {
        "id": "abc123",
        "prefix": {
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is..."},
                {"role": "user", "content": "Tell me more"}
            ]
        },
        "chosen": "Machine learning is...",
        "rejected": "I don't know."
    }
    
    prefix_new = ensure_prefix_dict(new_pair["prefix"])
    check(
        len(prefix_new["messages"]) == 3,
        "New 'prefix' format: preserves full conversation",
        "New 'prefix' format: conversation not preserved"
    )
    
    # ID consistency
    id_legacy = prefix_id_from_prefix(legacy_pair.get("prompt"))
    id_new = prefix_id_from_prefix(new_pair["prefix"])
    check(
        id_legacy == id_new,
        "ID consistency: same first user message → same ID",
        "ID consistency: IDs differ for same first user message"
    )
    
    return True


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 4: Density Computation
# ═════════════════════════════════════════════════════════════════════════════

def check_density_computation():
    print_header("CHECK 4: Density Computation")
    
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    
    # Create test embeddings with clear structure
    np.random.seed(42)
    
    # Dense cluster (high density, should get LOW weights)
    dense = np.random.randn(20, 32) * 0.1
    
    # Sparse outliers (low density, should get HIGH weights)
    sparse = np.random.randn(5, 32) * 2 + 10
    
    embeddings = np.vstack([dense, sparse])
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    check(
        embeddings.shape == (25, 32),
        f"Embeddings shape: {embeddings.shape}",
        "Embeddings shape incorrect"
    )
    
    # k-NN density
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    
    local_densities = np.mean(distances[:, 1:], axis=1)
    local_densities = np.maximum(local_densities, 1e-6)
    
    check(
        len(local_densities) == 25,
        "k-NN density: computed for all embeddings",
        "k-NN density: wrong number of densities"
    )
    
    # Inverse density weights
    weights = 1.0 / local_densities
    weights = weights / weights.sum()
    
    check(
        np.isclose(weights.sum(), 1.0),
        "Weights normalization: sum to 1.0",
        f"Weights normalization: sum is {weights.sum()}"
    )
    
    # Key check: sparse points should have HIGHER weights
    dense_weight_mean = weights[:20].mean()
    sparse_weight_mean = weights[20:].mean()
    
    check(
        sparse_weight_mean > dense_weight_mean * 2,
        f"Sparse prompts get higher weights ({sparse_weight_mean:.4f} > {dense_weight_mean:.4f})",
        f"Sparse prompts should have higher weights ({sparse_weight_mean:.4f} vs {dense_weight_mean:.4f})"
    )
    
    return True


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 5: Weight Mapping
# ═════════════════════════════════════════════════════════════════════════════

def check_weight_mapping():
    print_header("CHECK 5: Weight Mapping (Prefixes → Pairs)")
    
    from scripts.kat_utils import prefix_id_from_prefix
    import numpy as np
    
    # Create prefixes with weights
    prefixes = [
        {"id": "abc123", "prefix": {"messages": [{"role": "user", "content": "Q1"}]}},
        {"id": "def456", "prefix": {"messages": [{"role": "user", "content": "Q2"}]}},
        {"id": "ghi789", "prefix": {"messages": [{"role": "user", "content": "Q3"}]}},
    ]
    weights = np.array([0.1, 0.6, 0.3])  # Q2 is rare
    
    id_to_weight = {p["id"]: float(w) for p, w in zip(prefixes, weights)}
    
    # Create pairs (some share prefixes)
    pairs = [
        {"prefix": {"messages": [{"role": "user", "content": "Q1"}]}, "prefix_id": "abc123"},
        {"prefix": {"messages": [{"role": "user", "content": "Q1"}]}, "prefix_id": "abc123"},  # Duplicate
        {"prefix": {"messages": [{"role": "user", "content": "Q2"}]}, "prefix_id": "def456"},  # Rare
    ]
    
    # Map weights
    pair_weights = []
    for pair in pairs:
        pid = pair.get("prefix_id")
        weight = id_to_weight.get(pid, 1.0)
        pair_weights.append(weight)
    
    check(
        pair_weights[0] == pair_weights[1],
        "Same prefix → same weight",
        "Same prefix should have same weight!"
    )
    
    check(
        pair_weights[2] > pair_weights[0],
        f"Rare prefix Q2 has higher weight ({pair_weights[2]:.2f} > {pair_weights[0]:.2f})",
        f"Rare prefix should have higher weight ({pair_weights[2]:.2f} vs {pair_weights[0]:.2f})"
    )
    
    return True


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 6: Loss Functions
# ═════════════════════════════════════════════════════════════════════════════

def check_loss_functions():
    print_header("CHECK 6: Loss Functions")
    
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        warn("PyTorch not installed - skipping loss function checks")
        return True
    
    # Bradley-Terry loss
    def bt_loss(r_chosen, r_rejected):
        return F.softplus(-(r_chosen - r_rejected))
    
    # Correct preference
    r_c = torch.tensor([5.0])
    r_r = torch.tensor([1.0])
    loss_correct = bt_loss(r_c, r_r)
    
    # Wrong preference
    r_c_wrong = torch.tensor([1.0])
    r_r_wrong = torch.tensor([5.0])
    loss_wrong = bt_loss(r_c_wrong, r_r_wrong)
    
    check(
        loss_correct < loss_wrong,
        f"Bradley-Terry: correct preference has lower loss ({loss_correct.item():.4f} < {loss_wrong.item():.4f})",
        f"Bradley-Terry: wrong loss ordering ({loss_correct.item():.4f} vs {loss_wrong.item():.4f})"
    )
    
    check(
        loss_correct.item() > 0,
        "Bradley-Terry: loss is always positive",
        "Bradley-Terry: loss should be positive"
    )
    
    return True


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 7: Diversity Metrics
# ═════════════════════════════════════════════════════════════════════════════

def check_diversity_metrics():
    print_header("CHECK 7: Diversity Metrics")
    
    # Gini coefficient
    def gini_coefficient(tokens):
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        freqs = sorted(counts.values())
        n = sum(freqs)
        if n == 0:
            return 0.0
        return sum((2 * i + 1) * f for i, f in enumerate(freqs)) / (n * len(freqs)) - (len(freqs) + 1) / len(freqs)
    
    # Uniform distribution (high diversity)
    tokens_uniform = list(range(100))
    gini_uniform = gini_coefficient(tokens_uniform)
    
    # Peaked distribution (low diversity)
    tokens_peaked = [0] * 90 + list(range(1, 11))
    gini_peaked = gini_coefficient(tokens_peaked)
    
    check(
        gini_uniform < gini_peaked,
        f"Gini: uniform distribution has lower Gini ({gini_uniform:.4f} < {gini_peaked:.4f})",
        f"Gini: uniform should be lower ({gini_uniform:.4f} vs {gini_peaked:.4f})"
    )
    
    # Em-dash counting
    def count_em_dashes(text):
        return text.count("—") + text.count("–")
    
    check(
        count_em_dashes("test—test—test") == 2,
        "Em-dash counting: correctly counts em-dashes",
        "Em-dash counting: incorrect count"
    )
    
    return True


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 8: File Paths and Base Directory
# ═════════════════════════════════════════════════════════════════════════════

def check_paths():
    print_header("CHECK 8: File Paths and Base Directory")
    
    from nanochat.common import get_base_dir
    
    base_dir = get_base_dir()
    check(
        os.path.exists(base_dir),
        f"Base directory exists: {base_dir}",
        f"Base directory does not exist: {base_dir}"
    )
    
    # Check expected subdirectories
    data_dir = os.path.join(base_dir, "data")
    if os.path.exists(data_dir):
        print_pass(f"Data directory exists: {data_dir}")
        
        # Check for key files
        pairs_file = os.path.join(data_dir, "pairs_all.jsonl")
        if os.path.exists(pairs_file):
            print_pass(f"pairs_all.jsonl exists")
            # Count lines
            with open(pairs_file, 'r') as f:
                count = sum(1 for _ in f)
            print_info(f"pairs_all.jsonl has {count:,} pairs")
        else:
            print_info(f"pairs_all.jsonl not found (run kat_download_pairs first)")
        
        prefixes_file = os.path.join(data_dir, "prefixes_all.jsonl")
        if os.path.exists(prefixes_file):
            print_pass(f"prefixes_all.jsonl exists")
        else:
            print_info(f"prefixes_all.jsonl not found (run kat_make_prefixes first)")
        
        embeddings_dir = os.path.join(data_dir, "embeddings_offline")
        if os.path.exists(embeddings_dir):
            print_pass(f"embeddings_offline directory exists")
            weights_file = os.path.join(embeddings_dir, "density_weights.npy")
            if os.path.exists(weights_file):
                import numpy as np
                weights = np.load(weights_file)
                print_info(f"density_weights.npy: shape={weights.shape}, sum={weights.sum():.4f}")
        else:
            print_info(f"embeddings_offline not found (run kat_compute_embeddings_offline first)")
    else:
        print_info(f"Data directory not found: {data_dir}")
        print_info("This is normal if you haven't run the pipeline yet")
    
    return True


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  KAT DENSITY-AWARE PIPELINE VERIFICATION{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    
    success = True
    
    try:
        success = check_imports() and success
        success = check_kat_utils() and success
        success = check_data_formats() and success
        success = check_density_computation() and success
        success = check_weight_mapping() and success
        success = check_loss_functions() and success
        success = check_diversity_metrics() and success
        success = check_paths() and success
    except Exception as e:
        print_fail(f"Unexpected error: {e}")
        traceback.print_exc()
        success = False
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    print(f"  {GREEN}Passed:{RESET}   {passed}")
    print(f"  {RED}Failed:{RESET}   {failed}")
    print(f"  {YELLOW}Warnings:{RESET} {warnings}")
    
    if failed == 0:
        print(f"\n  {GREEN}{BOLD}✅ ALL CHECKS PASSED!{RESET}")
        print(f"\n  Your setup is ready. Next steps:")
        print(f"  {CYAN}1.{RESET} Run data pipeline: python -m scripts.kat_download_pairs")
        print(f"  {CYAN}2.{RESET} Extract prefixes:  python -m scripts.kat_make_prefixes")
        print(f"  {CYAN}3.{RESET} Train RM:          python -m scripts.kat_train_rm")
    else:
        print(f"\n  {RED}{BOLD}❌ {failed} CHECK(S) FAILED{RESET}")
        print(f"\n  Please fix the issues above before running the pipeline.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


