#!/usr/bin/env python3
"""
MATHEMATICAL CORRECTNESS TESTS FOR TRAINING

Run: python -m pytest tests/test_training_math.py -v -s

These tests verify the mathematical correctness of:
  1. Bradley-Terry loss
  2. KL divergence
  3. Density weighting
  4. GRPO advantage computation
  5. Weight normalization
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Only import torch if available
torch_available = False
try:
    import torch
    import torch.nn.functional as F
    torch_available = True
except ImportError:
    pass


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Bradley-Terry Loss Mathematical Correctness
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestBradleyTerryLoss:
    """Verify Bradley-Terry loss is mathematically correct."""
    
    def test_bt_loss_formula(self):
        """
        Bradley-Terry: P(chosen > rejected) = σ(r_c - r_r)
        Loss = -log σ(r_c - r_r) = log(1 + exp(-(r_c - r_r))) = softplus(-(r_c - r_r))
        """
        def bt_loss(r_c, r_r):
            return F.softplus(-(r_c - r_r))
        
        # Test 1: Correct ordering (r_c > r_r) should give low loss
        r_c = torch.tensor([5.0])
        r_r = torch.tensor([1.0])
        loss_correct = bt_loss(r_c, r_r)
        
        # Test 2: Wrong ordering (r_c < r_r) should give high loss
        r_c_wrong = torch.tensor([1.0])
        r_r_wrong = torch.tensor([5.0])
        loss_wrong = bt_loss(r_c_wrong, r_r_wrong)
        
        assert loss_correct < loss_wrong, \
            f"Correct preference should have lower loss: {loss_correct.item():.4f} vs {loss_wrong.item():.4f}"
        
        # Test 3: Equal rewards should give log(2) ≈ 0.693
        r_equal = torch.tensor([3.0])
        loss_equal = bt_loss(r_equal, r_equal)
        expected = np.log(2)  # softplus(0) = log(1 + exp(0)) = log(2)
        assert abs(loss_equal.item() - expected) < 1e-5, \
            f"Equal rewards should give log(2): {loss_equal.item():.4f} vs {expected:.4f}"
    
    def test_bt_loss_gradient_direction(self):
        """Verify gradient pushes r_c up and r_r down."""
        r_c = torch.tensor([2.0], requires_grad=True)
        r_r = torch.tensor([3.0], requires_grad=True)  # Wrong ordering
        
        loss = F.softplus(-(r_c - r_r))
        loss.backward()
        
        # Gradient should push r_c up (negative gradient)
        assert r_c.grad < 0, "Gradient should push r_c higher"
        # Gradient should push r_r down (positive gradient)
        assert r_r.grad > 0, "Gradient should push r_r lower"
    
    def test_bt_loss_numerical_stability(self):
        """Test numerical stability with extreme values."""
        def bt_loss(r_c, r_r):
            return F.softplus(-(r_c - r_r))
        
        # Very high difference (should not overflow)
        r_c = torch.tensor([100.0])
        r_r = torch.tensor([-100.0])
        loss = bt_loss(r_c, r_r)
        assert not torch.isnan(loss) and not torch.isinf(loss), "Should handle large differences"
        assert loss.item() < 1e-10, "Very confident correct should have ~0 loss"
        
        # Very wrong prediction (should not overflow)
        r_c = torch.tensor([-100.0])
        r_r = torch.tensor([100.0])
        loss = bt_loss(r_c, r_r)
        assert not torch.isnan(loss) and not torch.isinf(loss), "Should handle extreme wrong predictions"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: KL Divergence Mathematical Correctness
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestKLDivergence:
    """Verify KL divergence computation is correct."""
    
    def test_kl_formula(self):
        """
        KL(p || q) = Σ p(x) log(p(x)/q(x)) = Σ p(x) (log p(x) - log q(x))
        """
        # Simple 3-class distribution
        logp = torch.tensor([[np.log(0.7), np.log(0.2), np.log(0.1)]])  # [1, 3]
        logq = torch.tensor([[np.log(0.5), np.log(0.3), np.log(0.2)]])  # [1, 3]
        
        p = logp.exp()
        
        # Manual KL computation
        kl_manual = (p * (logp - logq)).sum()
        
        # Using torch.kl_div (note: expects log-probabilities for input, probabilities for target)
        # kl_div computes: target * (log(target) - input) when reduction='sum'
        # We want: p * (log(p) - log(q))
        kl_torch = F.kl_div(logq, p, reduction='sum')
        
        assert abs(kl_manual.item() - kl_torch.item()) < 1e-5, \
            f"KL computation mismatch: {kl_manual.item():.6f} vs {kl_torch.item():.6f}"
    
    def test_kl_properties(self):
        """Test KL divergence properties."""
        # Same distribution: KL should be 0
        logp = torch.tensor([[np.log(0.5), np.log(0.3), np.log(0.2)]])
        p = logp.exp()
        kl_same = (p * (logp - logp)).sum()
        assert abs(kl_same.item()) < 1e-10, "KL(p || p) should be 0"
        
        # KL is always >= 0 (Gibbs inequality)
        logp = torch.tensor([[np.log(0.8), np.log(0.15), np.log(0.05)]])
        logq = torch.tensor([[np.log(0.33), np.log(0.33), np.log(0.34)]])
        p = logp.exp()
        kl = (p * (logp - logq)).sum()
        assert kl.item() >= 0, f"KL should be non-negative: {kl.item()}"
    
    def test_kl_asymmetry(self):
        """KL is asymmetric: KL(p || q) ≠ KL(q || p) in general."""
        logp = torch.tensor([[np.log(0.9), np.log(0.1)]])
        logq = torch.tensor([[np.log(0.5), np.log(0.5)]])
        p = logp.exp()
        q = logq.exp()
        
        kl_pq = (p * (logp - logq)).sum()
        kl_qp = (q * (logq - logp)).sum()
        
        assert abs(kl_pq.item() - kl_qp.item()) > 0.1, \
            f"KL should be asymmetric: KL(p||q)={kl_pq.item():.4f}, KL(q||p)={kl_qp.item():.4f}"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Density Weighting Mathematical Correctness
# ═════════════════════════════════════════════════════════════════════════════

class TestDensityWeighting:
    """Verify density weighting math."""
    
    def test_inverse_density_formula(self):
        """
        density = mean distance to k-NN
        weight = 1 / density
        
        High density (small distance) → low weight
        Low density (large distance) → high weight
        """
        # Mock k-NN distances (excluding self at column 0)
        distances = np.array([
            [0.1, 0.1, 0.1],  # Dense cluster - mean=0.1
            [0.5, 0.6, 0.7],  # Medium - mean=0.6
            [2.0, 2.5, 3.0],  # Sparse - mean=2.5
        ])
        
        densities = distances.mean(axis=1)
        weights = 1.0 / densities
        weights = weights / weights.sum()  # Normalize
        
        # Sparse point should have highest weight
        assert weights[2] > weights[1] > weights[0], \
            f"Weights should increase with sparsity: {weights}"
        
        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0), f"Weights should sum to 1: {weights.sum()}"
    
    def test_weight_ratio(self):
        """Test that weight ratios match inverse density ratios."""
        densities = np.array([1.0, 2.0, 4.0])
        weights = 1.0 / densities
        
        # Weight ratio should be inverse of density ratio
        # w1/w2 = d2/d1
        assert np.isclose(weights[0] / weights[1], densities[1] / densities[0]), \
            "Weight ratio should be inverse of density ratio"
    
    def test_weight_capping(self):
        """Test that weight capping prevents extreme values."""
        densities = np.array([0.001, 1.0, 1000.0])  # Extreme range
        weights = 1.0 / densities
        
        cap = 10.0
        weights_capped = np.minimum(weights, cap)
        
        assert weights_capped.max() <= cap, "Capping should limit max weight"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Weight Normalization
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestWeightNormalization:
    """Verify weight normalization modes."""
    
    def test_mean_normalization(self):
        """
        Mode 'mean': wn = w / w.mean()
        After normalization, mean(wn) = 1
        """
        weights = torch.tensor([0.5, 1.0, 1.5, 2.0])
        wn = weights / (weights.mean() + 1e-12)
        
        assert abs(wn.mean().item() - 1.0) < 1e-5, \
            f"Normalized weights should have mean=1: {wn.mean().item()}"
    
    def test_sum_normalization(self):
        """
        Mode 'sum': wn = w * (n / w.sum())
        After normalization, sum(wn) = n
        """
        weights = torch.tensor([0.5, 1.0, 1.5, 2.0])
        n = weights.numel()
        wn = weights * (n / (weights.sum() + 1e-12))
        
        assert abs(wn.sum().item() - n) < 1e-5, \
            f"Normalized weights should have sum={n}: {wn.sum().item()}"
    
    def test_weighted_loss_effect(self):
        """Verify weighted loss changes gradient direction correctly."""
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # High weight on example 0
        weights_favor_0 = torch.tensor([10.0, 1.0, 1.0, 1.0])
        wn = weights_favor_0 / (weights_favor_0.mean() + 1e-12)
        weighted_loss = (wn * losses).mean()
        
        # Standard (uniform) loss
        uniform_loss = losses.mean()
        
        # Weighted loss should be lower (since we're emphasizing the low-loss example)
        assert weighted_loss < uniform_loss, \
            f"Weighting low-loss examples should reduce total loss: {weighted_loss.item()} vs {uniform_loss.item()}"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: GRPO Advantage Computation
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestGRPOAdvantage:
    """Verify GRPO advantage computation."""
    
    def test_advantage_formula(self):
        """
        A = (r_c - r_r) - β * (KL_c - KL_r)
        
        High reward difference → positive advantage
        High KL penalty → reduced advantage
        """
        dr = torch.tensor([2.0])  # Reward margin
        dkl = torch.tensor([0.5])  # KL margin
        beta = 0.1
        
        A = dr - beta * dkl
        expected = 2.0 - 0.1 * 0.5
        
        assert abs(A.item() - expected) < 1e-5, \
            f"Advantage computation: {A.item()} vs {expected}"
    
    def test_advantage_standardization(self):
        """Test advantage standardization."""
        A = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        A_std = (A - A.mean()) / (A.std(unbiased=False) + 1e-6)
        
        # Standardized should have mean≈0 and std≈1
        assert abs(A_std.mean().item()) < 1e-5, f"Standardized mean should be ~0: {A_std.mean().item()}"
        assert abs(A_std.std(unbiased=False).item() - 1.0) < 1e-5, f"Standardized std should be ~1: {A_std.std().item()}"
    
    def test_policy_gradient_direction(self):
        """
        loss = -A * (log π(c) - log π(r))
        
        If A > 0: should increase log π(c), decrease log π(r)
        If A < 0: should decrease log π(c), increase log π(r)
        """
        A_positive = torch.tensor([1.0])
        lp_c = torch.tensor([0.0], requires_grad=True)
        lp_r = torch.tensor([0.0], requires_grad=True)
        
        # Positive advantage
        loss = -(A_positive.detach() * (lp_c - lp_r)).mean()
        loss.backward()
        
        # d(loss)/d(lp_c) = -A < 0 when A > 0 → should increase lp_c
        assert lp_c.grad < 0, "Positive A should push to increase log π(chosen)"
        # d(loss)/d(lp_r) = +A > 0 when A > 0 → should decrease lp_r  
        assert lp_r.grad > 0, "Positive A should push to decrease log π(rejected)"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Log-Probability Computation
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not torch_available, reason="PyTorch not installed")
class TestLogProbability:
    """Verify log-probability computation over sequences."""
    
    def test_sum_logprobs_formula(self):
        """
        sum_logprobs = Σ_t log π(y_t | y_{<t}, x) for response tokens only
        """
        batch_size, seq_len, vocab_size = 2, 5, 10
        
        # Mock logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        logp = logits.log_softmax(dim=-1)
        
        # Labels: -100 for prompt, actual tokens for response
        labels = torch.tensor([
            [-100, -100, 3, 5, 7],  # Response starts at position 2
            [-100, -100, -100, 2, 4],  # Response starts at position 3
        ])
        
        # Manual computation
        def sum_logprobs_manual(logp, labels):
            tgt = labels[:, 1:].contiguous()
            logp_shifted = logp[:, :-1].contiguous()
            mask = (tgt != -100)
            gathered = logp_shifted.gather(2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            return (gathered * mask).sum(dim=1)
        
        result = sum_logprobs_manual(logp, labels)
        
        # Should be a tensor of shape [batch_size]
        assert result.shape == (batch_size,), f"Wrong shape: {result.shape}"
        
        # Should be negative (log-probs are <= 0)
        assert (result <= 0).all(), "Log-probs should be non-positive"
    
    def test_mask_correctness(self):
        """Verify that only response tokens contribute to log-prob."""
        batch_size, seq_len, vocab_size = 1, 5, 10
        
        # All zeros logits → uniform distribution → log(1/10) = -log(10) per token
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logp = logits.log_softmax(dim=-1)  # Each position: log(0.1) = -2.303
        
        # 2 response tokens
        labels = torch.tensor([[-100, -100, 3, 5, -100]])  # 2 response tokens at positions 2,3
        
        def sum_logprobs(logp, labels):
            tgt = labels[:, 1:].contiguous()
            logp_shifted = logp[:, :-1].contiguous()
            mask = (tgt != -100)
            gathered = logp_shifted.gather(2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            return (gathered * mask).sum(dim=1)
        
        result = sum_logprobs(logp, labels)
        
        # Expected: 2 tokens * log(0.1) = 2 * -2.303 ≈ -4.605
        expected = 2 * np.log(0.1)
        assert abs(result.item() - expected) < 1e-3, \
            f"Sum should be 2 * log(0.1): {result.item()} vs {expected}"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7: Hypothesis Validation - Density Sampling
# ═════════════════════════════════════════════════════════════════════════════

class TestHypothesisValidation:
    """Tests related to the density sampling hypothesis."""
    
    def test_density_sampling_covers_rare_prompts(self):
        """Verify that density sampling increases coverage of rare prompts."""
        np.random.seed(42)
        
        # 100 prompts: 90 common (clustered), 10 rare (sparse)
        n_common, n_rare = 90, 10
        n_total = n_common + n_rare
        
        # Simulate: common prompts have high density (low weight)
        # rare prompts have low density (high weight)
        common_weight = 0.1
        rare_weight = 1.0
        
        weights = np.array([common_weight] * n_common + [rare_weight] * n_rare)
        weights = weights / weights.sum()
        
        # Sample 100 examples
        n_samples = 100
        samples = np.random.choice(n_total, size=n_samples, p=weights, replace=True)
        
        # Count how many rare prompts (indices 90-99) were sampled
        rare_samples = np.sum(samples >= n_common)
        
        # With uniform sampling, we'd expect ~10 rare samples (10/100 * 100)
        # With density weighting, rare should be oversampled
        # Expected: 10 * 1.0 / (90 * 0.1 + 10 * 1.0) * 100 ≈ 52.6%
        
        assert rare_samples > 30, \
            f"Density sampling should oversample rare prompts: got {rare_samples}/100 rare"
        
        print(f"Rare prompts sampled: {rare_samples}/100 (expected ~52 with density weighting)")
    
    def test_mode_collapse_metric(self):
        """Test Gini coefficient as mode collapse metric."""
        def gini(values):
            """Gini coefficient: 0=uniform, 1=single mode."""
            sorted_vals = np.sort(values)
            n = len(sorted_vals)
            cumsum = np.cumsum(sorted_vals)
            return (2 * np.sum((np.arange(1, n+1) * sorted_vals)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
        
        # Uniform distribution (no collapse)
        uniform = np.ones(100) / 100
        gini_uniform = gini(uniform)
        
        # Peaked distribution (mode collapse)
        peaked = np.zeros(100)
        peaked[0] = 0.9
        peaked[1:11] = 0.01
        gini_peaked = gini(peaked)
        
        assert gini_uniform < gini_peaked, \
            f"Uniform should have lower Gini than peaked: {gini_uniform:.4f} vs {gini_peaked:.4f}"
        
        print(f"Gini uniform: {gini_uniform:.4f}, Gini peaked: {gini_peaked:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


