# Hypothesis Comparison: Where Should Density Sampling Go?

---

## Visual Pipeline Comparison

### Hypothesis A: Baseline (No Density Anywhere)

```
┌─────────────────────────────────────────────────────┐
│  DATASET: All 30k pairs                             │
│  - Rare prompts: ~1% of pairs                       │
│  - Common prompts: ~99% of pairs                    │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────▼─────────────┐
        │  RM TRAINING           │
        │  (Uniform Sampling)    │
        │  sees each pair ~1x    │
        │  rare prompts ~30x     │
        └──────────────┬─────────┘
                       │
            ┌──────────▼──────────────┐
            │  RM LEARNS              │
            │  ✗ Noisy on rare        │
            │  ✓ Good on common       │
            └──────────┬───────────────┘
                       │
                       │ (rm_baseline checkpoint)
                       │
        ┌──────────────▼─────────────┐
        │  GRPO TRAINING            │
        │  (Uniform Sampling)       │
        │  sees each pair ~1x       │
        └──────────────┬─────────────┘
                       │
            ┌──────────▼──────────────┐
            │  POLICY LEARNS          │
            │  ✗ From noisy rewards   │
            │  ✗ Limited diversity    │
            └──────────────────────────┘

TRAINING DATA SEEN:
Rare prompts:  [====] 1x
Common prompts:[================================================] ~99x
```

**Result:** Baseline performance. Rare prompts underrepresented in both RM and GRPO.

---

### Hypothesis B: Density in RM (NEW - Your Proposal)

```
┌─────────────────────────────────────────────────────┐
│  DATASET: All 30k pairs                             │
│  - Rare prompts: ~1% of pairs                       │
│  - Common prompts: ~99% of pairs                    │
└──────────────────┬──────────────────────────────────┘
                   │
                   │ ┌─ Density weights
                   │ │ (1/local_density)
                   │ │ Rare: weight ~0.01 (sample 1%)
                   │ └─ Common: weight ~0.001 (sample 0.1%)
                   │
        ┌──────────▼──────────────────┐
        │  RM TRAINING               │
        │  (WEIGHTED Sampling)       │
        │  by inverse density        │
        │  rare prompts ~100x        │
        │  common prompts ~1x        │
        └──────────────┬──────────────┘
                       │
            ┌──────────▼──────────────┐
            │  RM LEARNS              │
            │  ✓ Robust on rare!      │
            │  ✓ Good on common       │
            └──────────┬───────────────┘
                       │
                       │ (rm_density checkpoint)
                       │
        ┌──────────────▼─────────────┐
        │  GRPO TRAINING            │
        │  (Uniform Sampling)       │
        │  sees each pair ~1x       │
        └──────────────┬─────────────┘
                       │
            ┌──────────▼──────────────┐
            │  POLICY LEARNS          │
            │  ✓ From clean rewards!  │
            │  ✓ Natural diversity    │
            └──────────────────────────┘

TRAINING DATA SEEN BY RM:
Rare prompts:  [================================================] ~100x
Common prompts:[====] ~1x

TRAINING DATA SEEN BY GRPO:
Rare prompts:  [====] 1x
Common prompts:[================================================] ~99x
```

**Result:** RM becomes expert on rare prompts. GRPO inherits this expertise and learns better.

---

### Hypothesis C: Density in GRPO (Original Approach)

```
┌─────────────────────────────────────────────────────┐
│  DATASET: All 30k pairs                             │
│  - Rare prompts: ~1% of pairs                       │
│  - Common prompts: ~99% of pairs                    │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────▼─────────────┐
        │  RM TRAINING           │
        │  (Uniform Sampling)    │
        │  sees each pair ~1x    │
        │  rare prompts ~30x     │
        └──────────────┬─────────┘
                       │
            ┌──────────▼──────────────┐
            │  RM LEARNS              │
            │  ✗ Noisy on rare        │
            │  ✓ Good on common       │
            └──────────┬───────────────┘
                       │
                       │ (rm_baseline checkpoint)
                       │
                       │ ┌─ Density weights
                       │ │ (1/local_density)
                       │ │ Rare: weight ~0.01 (sample 1%)
                       │ └─ Common: weight ~0.001 (sample 0.1%)
                       │
        ┌──────────────▼──────────────┐
        │  GRPO TRAINING             │
        │  (WEIGHTED Sampling)       │
        │  by inverse density        │
        │  rare prompts ~100x        │
        │  common prompts ~1x        │
        └──────────────┬──────────────┘
                       │
            ┌──────────▼──────────────┐
            │  POLICY LEARNS          │
            │  ✗ From noisy rewards!  │
            │  ✓ Forced diversity     │
            └──────────────────────────┘

TRAINING DATA SEEN BY RM:
Rare prompts:  [====] 1x
Common prompts:[================================================] ~99x

TRAINING DATA SEEN BY GRPO:
Rare prompts:  [================================================] ~100x
Common prompts:[====] ~1x
```

**Problem:** RM doesn't learn well on rare prompts. Even though GRPO sees them more, the reward signal is noisy.

---

## Data Flow Diagram

### Hypothesis A & B: Where They Differ

```
                    Hypothesis A              Hypothesis B
                    (Baseline)               (Density in RM)
                    ──────────────           ───────────────

RM Training Data:   [====][====]...          [=======][==]...
                    uniform                  weighted by 1/density
                    rare=1%                  rare=100%

          │                                  │
          ▼                                  ▼

RM Quality:         RM(rare) = noisy         RM(rare) = robust
                    RM(common) = good        RM(common) = good

          │                                  │
          ▼                                  ▼

GRPO Training Data: [====][====]...          [====][====]...
                    uniform                  uniform
                    rare=1%                  rare=1%

          │                                  │
          ▼                                  ▼

Policy Quality:     Noisy training          Clean training
                    Limited diversity        Natural diversity
```

---

## The Key Insight: Why B > C

**Hypothesis C (GRPO density) assumes:**
- We can fix poor RM via forced exposure at GRPO level
- Sampling more rare prompts compensates for noisy RM rewards

**Reality:**
- If RM gives score=0.3 for rare prompt, seeing it 100 times in GRPO trains it on that noisy reward
- The GRPO loss will be unstable (reward signal is jittery)
- Policy learns from bad signal, doesn't help diversity

**Hypothesis B (RM density) assumes:**
- If RM sees rare prompts 100x during training, it learns their true reward function
- Then GRPO gets clean, stable reward signals for all prompts
- Policy trains on clean signal → learns natural diversity without forced sampling

**Analogy:**
- Hypothesis C: Student memorizes poorly-explained material more times = doesn't help
- Hypothesis B: Teacher learns material better first → student gets clear explanation

---

## Detailed Comparison Table

| Aspect | Hypothesis A | Hypothesis B | Hypothesis C |
|--------|------------|------------|------------|
| **RM sees** | Rare: 1% | Rare: 100% | Rare: 1% |
| **RM learns** | Noisy rare | Robust rare | Noisy rare |
| **GRPO sees** | Rare: 1% | Rare: 1% | Rare: 100% |
| **GRPO trains on** | Noisy rewards | Clean rewards | Noisy rewards |
| **Policy learns** | Baseline | Natural diversity | Forced diversity |
| **Root problem** | No focus on rare | ✅ Solved! | Still has noisy RM |
| **Extra sampling needed** | No | No | Yes (wastes GRPO iters) |
| **Expected performance** | Baseline | ⬆️ Better | Marginal improvement |

---

## Why Hypothesis B Should Win

**The Core Argument:**

Quality > Quantity

- **Hypothesis C:** Sees rare prompts more times (↑ quantity) BUT gets bad signal (↓ quality)
- **Hypothesis B:** RM learns better first (↑ quality) so GRPO gets clean signal

**Mathematical intuition:**

```
GRPO Loss = reward_loss + kl_penalty

If reward_signal_is_noisy:
    GRPO Loss might actually increase with more rare samples
    (because it's learning from worse data)

If reward_signal_is_clean:
    GRPO Loss improves steadily
    (quality signal leads to quality learning)
```

---

## Predictions for Each Hypothesis

### If Hypothesis B Works

```
Metric                    A (Baseline)    B (Density RM)    Improvement
─────────────────────────────────────────────────────────────────────
RM Loss (rare prompts)        0.45            0.35              22%
RM Loss (common prompts)      0.42            0.41              2%
GRPO Loss convergence        5000            4000              20%
Policy reward (rare)           0.5            0.65              30%
Policy reward (common)         0.7            0.72              3%
Policy diversity score        0.6            0.72              20%
```

✅ Indicates B is better because:
- RM learns rare prompts better
- GRPO converges faster (cleaner signal)
- Policy handles rare prompts better

### If Hypothesis B Fails

```
Metric                    A (Baseline)    B (Density RM)    Verdict
─────────────────────────────────────────────────────────────────
RM Loss (rare)                0.45            0.44              Similar
GRPO Convergence             5000            4900              Same
Final Policy Quality          0.60            0.61              No improvement
```

Possible reasons:
1. RM architecture already handles rare prompts fine
2. Density weights are too extreme (sampling bias)
3. Rare prompts are actually harder, not just undersampled
4. The training regime length matters more than sampling

---

## The Actual Data

**What you have:**

```
pairs_all.jsonl: 30,000 pairs

Prompt frequency distribution (estimated):
- Top 10% most common prompts: ~50% of pairs
- Middle 40%: ~40% of pairs  
- Bottom 50% (rare): ~10% of pairs

Density weights (1/local_density):
- Rare prompts: weight ≈ 0.01-0.05 (sample ~1-5%)
- Common prompts: weight ≈ 0.001-0.002 (sample ~0.1%)

Effect of density sampling:
- Rare prompt appears ~10x more often per epoch
- Common prompts appear less frequently
- Better coverage of edge cases
```

**Why Hypothesis B helps:**
- RM might overfit to common prompt patterns with uniform sampling
- By seeing rare prompts 10x more, RM learns their features
- Then GRPO can leverage those learned features

---

## How to Validate During Training

### Look for These Signs

**Hypothesis B is working if:**

✅ RM loss on rare prompts drops faster in density-aware version
✅ RM loss on common prompts stays similar (not degraded)
✅ GRPO loss with density-aware RM is cleaner (less variance)
✅ Policy learns to handle more diverse prompt types

**Hypothesis B is NOT working if:**

❌ RM overfits to rare prompts (loss drops but not generalizing)
❌ GRPO loss becomes higher/noisier with density-aware RM
❌ No difference in final policy performance
❌ Policy becomes worse on common prompts

---

## The Simplest Explanation

**Current state:**
- RM trained uniformly: "I see 30k pairs, 99% are about common topics, I'm pretty good at those"
- GRPO uses this RM: "I get good rewards for common patterns, noisy for rare ones"
- Result: Policy learns common patterns well, struggles with rare

**With Hypothesis B:**
- RM trained with density: "I see rare prompts 100 times each, common ones fewer times, all balanced"
- RM learns: "Rare topics have distinct patterns I should memorize"
- GRPO uses this RM: "I get good rewards for rare patterns too!"
- Result: Policy learns balanced across all prompt types

---

## Next Steps

1. **Train baseline** (Hypothesis A) - get reference numbers
2. **Train density-RM** (Hypothesis B) - compare RM quality metrics
3. **Train GRPO from each** - see which gives better policy
4. **Evaluate:**
   - Loss curves during training
   - RM calibration (reward scores distribution)
   - Policy diversity metrics
   - Benchmark performance

**Expected outcome:** Hypothesis B > Hypothesis A by 10-30% on diversity metrics
