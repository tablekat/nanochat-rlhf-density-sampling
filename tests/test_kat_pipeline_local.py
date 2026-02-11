#!/usr/bin/env python3
"""
COMPREHENSIVE LOCAL VERIFICATION TESTS FOR KAT DENSITY-AWARE PIPELINE

Run on Windows without GPU:
    python -m pytest tests/test_kat_pipeline_local.py -v -s

    # Or run individual test classes:
    python -m pytest tests/test_kat_pipeline_local.py::TestKatUtils -v -s
    python -m pytest tests/test_kat_pipeline_local.py::TestPrefixFormats -v -s
    python -m pytest tests/test_kat_pipeline_local.py::TestDensityPipeline -v -s
    python -m pytest tests/test_kat_pipeline_local.py::TestEndToEndDataFlow -v -s

These tests verify:
  ✓ kat_utils.py - Core utility functions
  ✓ Prefix format handling (legacy prompt vs new conversation format)
  ✓ Density weight computation and mapping
  ✓ End-to-end data flow from pairs → prefixes → weights → training samples
  ✓ HH-RLHF conversation parsing
  ✓ Gini coefficient and diversity metrics
  ✓ Truncation logic for sequences
  ✓ Bradley-Terry loss computation

NO GPU REQUIRED - All tests run on CPU with mock data.
"""

import pytest
import sys
import os
import json
import tempfile
import numpy as np
import hashlib
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: kat_utils.py - Core Utility Functions
# ═════════════════════════════════════════════════════════════════════════════

class TestKatUtils:
    """Test core utility functions from scripts/kat_utils.py"""
    
    def test_norm_space(self):
        """Test whitespace normalization"""
        from scripts.kat_utils import norm_space
        
        # Basic whitespace collapse
        assert norm_space("  hello   world  ") == "hello world"
        assert norm_space("hello\n\nworld") == "hello world"
        assert norm_space("hello\t\tworld") == "hello world"
        
        # None handling
        assert norm_space(None) == ""
        
        # Already clean
        assert norm_space("hello world") == "hello world"
    
    def test_ensure_prefix_dict_with_messages(self):
        """Test ensure_prefix_dict with proper conversation format"""
        from scripts.kat_utils import ensure_prefix_dict
        
        # Already in correct format
        prefix = {"messages": [{"role": "user", "content": "Hello"}]}
        result = ensure_prefix_dict(prefix)
        assert result == prefix
        assert result is not prefix  # Should be a deep copy
        
        # Verify deep copy (mutation safety)
        result["messages"].append({"role": "assistant", "content": "Hi"})
        assert len(prefix["messages"]) == 1  # Original unchanged
    
    def test_ensure_prefix_dict_with_string(self):
        """Test ensure_prefix_dict converts string to conversation"""
        from scripts.kat_utils import ensure_prefix_dict
        
        result = ensure_prefix_dict("What is AI?")
        assert "messages" in result
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "What is AI?"
    
    def test_ensure_prefix_dict_with_none(self):
        """Test ensure_prefix_dict handles None"""
        from scripts.kat_utils import ensure_prefix_dict
        
        result = ensure_prefix_dict(None)
        assert "messages" in result
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == ""
    
    def test_first_user_message(self):
        """Test extracting first user message from prefix"""
        from scripts.kat_utils import first_user_message
        
        # Single user message
        prefix = {"messages": [{"role": "user", "content": "First question"}]}
        assert first_user_message(prefix) == "First question"
        
        # Multi-turn conversation
        prefix = {
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Reply"},
                {"role": "user", "content": "Second"}
            ]
        }
        assert first_user_message(prefix) == "First"
        
        # No user message
        prefix = {"messages": [{"role": "assistant", "content": "Hello"}]}
        assert first_user_message(prefix) == ""
        
        # Empty messages
        prefix = {"messages": []}
        assert first_user_message(prefix) == ""
    
    def test_prefix_id_from_text(self):
        """Test deterministic prefix ID generation"""
        from scripts.kat_utils import prefix_id_from_text
        
        # Same text -> same ID
        id1 = prefix_id_from_text("What is AI?")
        id2 = prefix_id_from_text("What is AI?")
        assert id1 == id2
        assert len(id1) == 16  # MD5 first 16 chars
        
        # Different text -> different ID
        id3 = prefix_id_from_text("What is ML?")
        assert id1 != id3
        
        # Whitespace normalization
        id4 = prefix_id_from_text("  What   is   AI?  ")
        assert id1 == id4  # Should match after normalization
        
        # Empty text -> None
        assert prefix_id_from_text("") is None
        assert prefix_id_from_text("   ") is None
    
    def test_prefix_id_from_prefix(self):
        """Test prefix ID from full prefix object"""
        from scripts.kat_utils import prefix_id_from_prefix
        
        # Dict with messages
        prefix1 = {"messages": [{"role": "user", "content": "What is AI?"}]}
        id1 = prefix_id_from_prefix(prefix1)
        
        # String (legacy format)
        id2 = prefix_id_from_prefix("What is AI?")
        
        assert id1 == id2  # Same first user message -> same ID
        
        # Multi-turn: only first user message matters for ID
        prefix_multi = {
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Follow-up"}
            ]
        }
        id3 = prefix_id_from_prefix(prefix_multi)
        assert id1 == id3  # Same first user message


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Prefix Format Handling
# ═════════════════════════════════════════════════════════════════════════════

class TestPrefixFormats:
    """Test handling of both legacy and new prefix formats"""
    
    def test_legacy_prompt_format(self):
        """Test legacy format with 'prompt' field"""
        legacy_pair = {
            "id": "abc123",
            "prompt": "What is AI?",
            "chosen": "AI is artificial intelligence.",
            "rejected": "I don't know."
        }
        
        from scripts.kat_utils import ensure_prefix_dict, prefix_id_from_prefix
        
        # Should convert prompt string to prefix dict
        prefix = ensure_prefix_dict(legacy_pair.get("prompt"))
        assert "messages" in prefix
        assert prefix["messages"][0]["content"] == "What is AI?"
        
        # ID should be consistent
        pid = prefix_id_from_prefix(prefix)
        assert pid == prefix_id_from_prefix("What is AI?")
    
    def test_new_prefix_format(self):
        """Test new format with 'prefix' conversation object"""
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
        
        from scripts.kat_utils import ensure_prefix_dict, first_user_message, prefix_id_from_prefix
        
        prefix = ensure_prefix_dict(new_pair["prefix"])
        assert len(prefix["messages"]) == 3
        
        # First user message for dedup
        first = first_user_message(prefix)
        assert first == "What is AI?"
        
        # ID based on first user message
        pid = prefix_id_from_prefix(prefix)
        assert pid == prefix_id_from_prefix("What is AI?")
    
    def test_hh_conversation_parsing(self):
        """Test HH-RLHF conversation string parsing"""
        # Example from kat_download_pairs.py
        text = """Human: What kind of noises did dinosaurs make?
Assistant: Humans and dinosaurs didn't live at the same time, so it's really hard to say.
Human: yes they did
Assistant: to guess, and that would probably require lots of reading and a certain amount of imagination."""
        
        def parse_conversation(text: str) -> list:
            exchanges = []
            lines = text.split("\n")
            current_role = None
            current_text = []
            
            for line in lines:
                line = line.rstrip()
                if line.startswith("Human:"):
                    if current_role == "A" and current_text:
                        exchanges.append(("A", " ".join(current_text).strip()))
                        current_text = []
                    current_role = "H"
                    current_text.append(line[6:].strip())
                elif line.startswith("Assistant:"):
                    if current_role == "H" and current_text:
                        exchanges.append(("H", " ".join(current_text).strip()))
                        current_text = []
                    current_role = "A"
                    current_text.append(line[10:].strip())
                elif current_role and line.strip():
                    current_text.append(line.strip())
            
            if current_role and current_text:
                exchanges.append((current_role, " ".join(current_text).strip()))
            return exchanges
        
        exchanges = parse_conversation(text)
        
        # Verify correct parsing
        assert len(exchanges) == 4
        assert exchanges[0][0] == "H"  # Human first
        assert exchanges[1][0] == "A"  # Then assistant
        assert "dinosaurs" in exchanges[0][1].lower()


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Density Pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestDensityPipeline:
    """Test density weight computation and mapping"""
    
    def test_density_weight_computation(self):
        """Test k-NN density weight computation"""
        from sklearn.neighbors import NearestNeighbors
        
        # Create synthetic embeddings with clear clusters
        np.random.seed(42)
        
        # Dense cluster (high similarity, low weight expected)
        dense_cluster = np.random.randn(20, 10) * 0.1
        
        # Sparse outliers (low similarity, high weight expected)
        sparse_points = np.random.randn(5, 10) * 2 + 10
        
        embeddings = np.vstack([dense_cluster, sparse_points])
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        
        # Local density = mean distance to k neighbors (exclude self)
        local_densities = np.mean(distances[:, 1:], axis=1)
        local_densities = np.maximum(local_densities, 1e-6)
        
        # Inverse density weights
        weights = 1.0 / local_densities
        weights = weights / weights.sum()
        
        # Dense cluster should have lower weights (higher density)
        dense_weight_mean = weights[:20].mean()
        sparse_weight_mean = weights[20:].mean()
        
        # Sparse points should have HIGHER weights (lower density, rarer)
        assert sparse_weight_mean > dense_weight_mean, \
            f"Sparse points should have higher weights: {sparse_weight_mean} vs {dense_weight_mean}"
    
    def test_weight_mapping_to_pairs(self):
        """Test mapping density weights from prefixes to pairs"""
        # Create synthetic prefixes with weights
        prefixes = [
            {"id": "abc123", "prefix": {"messages": [{"role": "user", "content": "Q1"}]}},
            {"id": "def456", "prefix": {"messages": [{"role": "user", "content": "Q2"}]}},
            {"id": "ghi789", "prefix": {"messages": [{"role": "user", "content": "Q3"}]}},
        ]
        
        # Weights for each prefix (indexed by id)
        weights = np.array([0.1, 0.5, 0.4])  # Q2 is rare (high weight)
        
        # Create ID -> weight mapping
        id_to_weight = {p["id"]: float(w) for p, w in zip(prefixes, weights)}
        
        # Create pairs that reference these prefixes
        pairs = [
            {"prefix_id": "abc123", "chosen": "A1", "rejected": "R1"},  # Q1, common
            {"prefix_id": "abc123", "chosen": "A2", "rejected": "R2"},  # Q1, common (duplicate)
            {"prefix_id": "def456", "chosen": "A3", "rejected": "R3"},  # Q2, rare
        ]
        
        # Map weights to pairs
        pair_weights = []
        for pair in pairs:
            pid = pair.get("prefix_id")
            weight = id_to_weight.get(pid, 1.0)
            pair_weights.append(weight)
        
        # Verify weights
        assert pair_weights[0] == 0.1  # Q1
        assert pair_weights[1] == 0.1  # Q1 (same prefix)
        assert pair_weights[2] == 0.5  # Q2 (rare, higher weight)
        
        # Pairs with same prefix should have same weight
        assert pair_weights[0] == pair_weights[1]
    
    def test_weighted_sampling(self):
        """Test that weighted sampling favors rare prompts"""
        np.random.seed(42)
        
        # Simulate weights: some prompts are 10x more rare
        n_samples = 1000
        weights = np.ones(100)
        weights[:10] = 10.0  # First 10 prompts are rare (10x weight)
        weights = weights / weights.sum()
        
        # Sample with replacement
        samples = np.random.choice(100, size=n_samples, p=weights, replace=True)
        
        # Count how often rare prompts (0-9) are sampled
        rare_count = np.sum(samples < 10)
        common_count = np.sum(samples >= 10)
        
        # Rare prompts should be sampled ~50% (10 prompts with 10x weight each)
        # vs common prompts ~50% (90 prompts with 1x weight each)
        rare_ratio = rare_count / n_samples
        assert 0.4 < rare_ratio < 0.6, f"Rare prompts should be ~50% of samples, got {rare_ratio:.2%}"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: End-to-End Data Flow
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEndDataFlow:
    """Test complete data flow from pairs to training"""
    
    def test_pairs_to_prefixes_dedup(self):
        """Test deduplication from pairs to unique prefixes"""
        from scripts.kat_utils import prefix_id_from_prefix, ensure_prefix_dict, first_user_message, norm_space
        
        pairs = [
            # Same first user message (should dedup)
            {"prefix": {"messages": [{"role": "user", "content": "What is AI?"}]}, "chosen": "C1", "rejected": "R1"},
            {"prefix": {"messages": [{"role": "user", "content": "What is AI?"}]}, "chosen": "C2", "rejected": "R2"},
            # Different first user message
            {"prefix": {"messages": [{"role": "user", "content": "What is ML?"}]}, "chosen": "C3", "rejected": "R3"},
            # Same first user message, different continuation
            {"prefix": {"messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is..."},
                {"role": "user", "content": "More?"}
            ]}, "chosen": "C4", "rejected": "R4"},
        ]
        
        seen = set()
        unique_prefixes = []
        
        for pair in pairs:
            prefix = ensure_prefix_dict(pair["prefix"])
            first_msg = norm_space(first_user_message(prefix))
            pid = prefix_id_from_prefix(prefix)
            
            if pid not in seen:
                seen.add(pid)
                unique_prefixes.append({"id": pid, "prefix": prefix})
        
        # Should have 2 unique prefixes (by first user message)
        assert len(unique_prefixes) == 2
        
        # All 4 pairs should map to one of 2 prefix IDs
        pair_ids = [prefix_id_from_prefix(p["prefix"]) for p in pairs]
        unique_ids = set(pair_ids)
        assert len(unique_ids) == 2
    
    def test_full_pipeline_simulation(self):
        """Simulate full pipeline: pairs → prefixes → embeddings → weights → training"""
        from scripts.kat_utils import prefix_id_from_prefix, ensure_prefix_dict
        
        # Stage 1: Create synthetic pairs (simulating kat_download_pairs)
        pairs = []
        prompts = ["What is AI?", "How does ML work?", "Explain deep learning"]
        
        for i in range(30):
            # First two prompts appear 10x each (common)
            # Last prompt appears 10x (also common but different topic)
            prompt_idx = i % 3
            pair = {
                "id": f"pair_{i}",
                "prefix": {"messages": [{"role": "user", "content": prompts[prompt_idx]}]},
                "chosen": f"Good answer {i}",
                "rejected": f"Bad answer {i}",
            }
            pairs.append(pair)
        
        # Stage 2: Extract unique prefixes (simulating kat_make_prefixes)
        seen = set()
        prefixes = []
        for pair in pairs:
            pid = prefix_id_from_prefix(pair["prefix"])
            if pid not in seen:
                seen.add(pid)
                prefixes.append({"id": pid, "prefix": pair["prefix"]})
        
        assert len(prefixes) == 3  # 3 unique prompts
        
        # Stage 3: Compute embeddings (simulating kat_compute_embeddings_offline)
        # Mock: just use random vectors
        np.random.seed(42)
        embeddings = np.random.randn(len(prefixes), 64)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Stage 4: Compute density weights
        from sklearn.neighbors import NearestNeighbors
        k = min(2, len(prefixes) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        local_densities = np.mean(distances[:, 1:], axis=1)
        local_densities = np.maximum(local_densities, 1e-6)
        weights = 1.0 / local_densities
        weights = weights / weights.sum()
        
        assert len(weights) == len(prefixes)
        assert np.isclose(weights.sum(), 1.0)
        
        # Stage 5: Map weights to pairs (simulating kat_train_rm)
        id_to_weight = {p["id"]: float(w) for p, w in zip(prefixes, weights)}
        
        pair_weights = []
        for pair in pairs:
            pid = prefix_id_from_prefix(pair["prefix"])
            weight = id_to_weight.get(pid, 1.0)
            pair_weights.append(weight)
        
        assert len(pair_weights) == len(pairs)
        
        # Pairs with same prefix should have same weight
        prompt_0_indices = [i for i, p in enumerate(pairs) if "AI" in p["prefix"]["messages"][0]["content"]]
        prompt_0_weights = [pair_weights[i] for i in prompt_0_indices]
        assert len(set(np.round(prompt_0_weights, 6))) == 1, "Same prefix should have same weight"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Diversity Metrics
# ═════════════════════════════════════════════════════════════════════════════

class TestDiversityMetrics:
    """Test diversity evaluation metrics"""
    
    def test_gini_coefficient(self):
        """Test Gini coefficient computation"""
        def gini_coefficient(tokens):
            if not tokens:
                return 0.0
            counts = Counter(tokens)
            freqs = sorted(counts.values())
            n = sum(freqs)
            if n == 0:
                return 0.0
            return sum((2 * i + 1) * f for i, f in enumerate(freqs)) / (n * len(freqs)) - (len(freqs) + 1) / len(freqs)
        
        # Uniform distribution (high diversity, low Gini)
        tokens_uniform = list(range(100))  # 100 unique tokens
        gini_uniform = gini_coefficient(tokens_uniform)
        
        # Peaked distribution (low diversity, high Gini)
        tokens_peaked = [0] * 90 + list(range(1, 11))  # Mostly token 0
        gini_peaked = gini_coefficient(tokens_peaked)
        
        # Gini should be lower for more uniform/diverse distribution
        assert gini_uniform < gini_peaked, \
            f"Uniform distribution should have lower Gini: {gini_uniform} vs {gini_peaked}"
    
    def test_em_dash_counting(self):
        """Test em-dash frequency counting"""
        def count_em_dashes(text):
            return text.count("—") + text.count("–")
        
        text_with_dashes = "This is—a test—with em-dashes—everywhere"
        text_without = "This is a test with no em dashes anywhere"
        
        assert count_em_dashes(text_with_dashes) == 3
        assert count_em_dashes(text_without) == 0
    
    def test_vocabulary_ratio(self):
        """Test vocabulary diversity ratio"""
        def vocabulary_ratio(tokens):
            if not tokens:
                return 0.0
            return len(set(tokens)) / len(tokens)
        
        # High diversity
        tokens_diverse = list(range(100))  # All unique
        assert vocabulary_ratio(tokens_diverse) == 1.0
        
        # Low diversity
        tokens_repetitive = [0] * 100  # All same
        assert vocabulary_ratio(tokens_repetitive) == 0.01
        
        # Mixed
        tokens_mixed = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # 3 unique in 10 tokens
        assert vocabulary_ratio(tokens_mixed) == 0.3


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Truncation and Sequence Building
# ═════════════════════════════════════════════════════════════════════════════

class TestSequenceBuilding:
    """Test sequence truncation and building logic"""
    
    def test_truncate_two(self):
        """Test truncation logic from kat_train_grpo.py"""
        def truncate_two(p, r, max_len, min_prompt):
            if len(p) + len(r) <= max_len:
                return p, r
            resp_budget = max_len - min(len(p), min_prompt)
            resp_budget = max(resp_budget, 1)
            r = r[:resp_budget]
            over = (len(p) + len(r)) - max_len
            if over > 0:
                p = p[over:]
            return p, r
        
        # Case 1: Both fit
        p, r = truncate_two([1]*50, [2]*50, max_len=200, min_prompt=30)
        assert len(p) == 50
        assert len(r) == 50
        
        # Case 2: Need truncation
        p, r = truncate_two([1]*100, [2]*100, max_len=50, min_prompt=10)
        assert len(p) + len(r) <= 50
        assert len(r) >= 1  # Response always has at least 1 token
        
        # Case 3: Very long prompt
        p, r = truncate_two([1]*500, [2]*50, max_len=100, min_prompt=30)
        assert len(p) + len(r) <= 100
        assert len(p) >= 1  # Prompt is trimmed from left
    
    def test_sequence_padding(self):
        """Test sequence padding to fixed length"""
        def pad_sequences(seqs, max_len, pad_id):
            seqs = [seq[:max_len] for seq in seqs]
            padded = [seq + [pad_id] * (max_len - len(seq)) for seq in seqs]
            return padded
        
        seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
        padded = pad_sequences(seqs, max_len=6, pad_id=0)
        
        assert len(padded[0]) == 6
        assert len(padded[1]) == 6
        assert len(padded[2]) == 6
        
        assert padded[0] == [1, 2, 3, 0, 0, 0]
        assert padded[1] == [4, 5, 0, 0, 0, 0]
        assert padded[2] == [6, 7, 8, 9, 10, 0]  # Truncated


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7: Loss Functions
# ═════════════════════════════════════════════════════════════════════════════

class TestLossFunctions:
    """Test loss function implementations"""
    
    def test_bradley_terry_loss(self):
        """Test Bradley-Terry preference loss"""
        import torch
        import torch.nn.functional as F
        
        def bt_loss(reward_chosen, reward_rejected):
            return F.softplus(-(reward_chosen - reward_rejected))
        
        # Chosen > rejected: low loss
        r_c = torch.tensor([5.0])
        r_r = torch.tensor([1.0])
        loss_correct = bt_loss(r_c, r_r)
        
        # Chosen < rejected: high loss
        r_c_wrong = torch.tensor([1.0])
        r_r_wrong = torch.tensor([5.0])
        loss_wrong = bt_loss(r_c_wrong, r_r_wrong)
        
        assert loss_correct < loss_wrong, "Correct preference should have lower loss"
        assert loss_correct.item() > 0  # Always positive
    
    def test_weight_application(self):
        """Test per-example weight application"""
        import torch
        
        def apply_weights(loss_per_ex, weights, mode):
            if mode == "mean":
                wn = weights / (weights.mean() + 1e-12)
            elif mode == "sum":
                wn = weights * (weights.numel() / (weights.sum() + 1e-12))
            else:
                wn = weights
            return (wn * loss_per_ex).mean()
        
        loss = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([0.5, 1.0, 0.5])  # Middle example has higher weight
        
        # Without weights
        unweighted = loss.mean()
        
        # With weights (mode=mean normalizes to mean=1)
        weighted = apply_weights(loss, weights, "mean")
        
        # The weighted loss should be different (middle example emphasized)
        assert not torch.isclose(unweighted, weighted), "Weights should change loss"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 8: File Format Verification
# ═════════════════════════════════════════════════════════════════════════════

class TestFileFormats:
    """Test file format reading and writing"""
    
    def test_pairs_jsonl_format(self):
        """Test pairs_all.jsonl format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            pairs = [
                {
                    "id": "uuid1",
                    "prefix": {"messages": [{"role": "user", "content": "Q1"}]},
                    "chosen": "Good answer",
                    "rejected": "Bad answer",
                    "src": "hh-rlhf"
                },
                {
                    "id": "uuid2",
                    "prefix": {"messages": [{"role": "user", "content": "Q2"}]},
                    "chosen": "Response A",
                    "rejected": "Response B",
                    "src": "ultrafeedback"
                }
            ]
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
            fname = f.name
        
        try:
            # Read back and verify
            loaded = []
            with open(fname, 'r') as f:
                for line in f:
                    loaded.append(json.loads(line))
            
            assert len(loaded) == 2
            assert "prefix" in loaded[0]
            assert "messages" in loaded[0]["prefix"]
            assert loaded[0]["src"] == "hh-rlhf"
        finally:
            os.unlink(fname)
    
    def test_density_weights_format(self):
        """Test density_weights.npy format"""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            weights = np.array([0.1, 0.5, 0.3, 0.1], dtype=np.float32)
            np.save(f.name, weights)
            fname = f.name
        
        try:
            loaded = np.load(fname)
            assert loaded.dtype == np.float32
            assert loaded.shape == (4,)
            assert np.isclose(loaded.sum(), 1.0)
        finally:
            os.unlink(fname)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 9: Integration with Mock Tokenizer
# ═════════════════════════════════════════════════════════════════════════════

class TestMockTokenizer:
    """Test tokenizer-related functions with mock tokenizer"""
    
    def test_render_prefix_for_completion(self):
        """Test render_prefix_for_completion with mock tokenizer"""
        from scripts.kat_utils import render_prefix_for_completion
        
        # Create mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                # Simple: each character is a token
                return [ord(c) for c in text[:10]]
            
            def encode_special(self, token):
                special_map = {
                    "<|bos|>": 1,
                    "<|user_start|>": 2,
                    "<|user_end|>": 3,
                    "<|assistant_start|>": 4,
                    "<|assistant_end|>": 5,
                }
                return special_map.get(token, 0)
            
            def get_bos_token_id(self):
                return 1
            
            def render_conversation(self, conversation):
                # Simplified rendering
                ids = [self.get_bos_token_id()]
                for msg in conversation.get("messages", []):
                    if msg["role"] == "user":
                        ids.append(self.encode_special("<|user_start|>"))
                        ids.extend(self.encode(msg["content"]))
                        ids.append(self.encode_special("<|user_end|>"))
                    elif msg["role"] == "assistant":
                        ids.append(self.encode_special("<|assistant_start|>"))
                        ids.extend(self.encode(msg["content"]))
                        ids.append(self.encode_special("<|assistant_end|>"))
                mask = [0] * len(ids)
                return ids, mask
            
            def render_for_completion(self, conversation):
                ids, _ = self.render_conversation(conversation)
                ids.append(self.encode_special("<|assistant_start|>"))
                return ids
        
        tokenizer = MockTokenizer()
        
        # Test with user-ending prefix
        prefix = {"messages": [{"role": "user", "content": "Hello"}]}
        ids = render_prefix_for_completion(tokenizer, prefix)
        
        # Should end with assistant_start token (4)
        assert ids[-1] == 4, f"Should end with assistant_start, got {ids[-1]}"


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


