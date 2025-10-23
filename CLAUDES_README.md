# Claude's Additions: GRPO Training with Density-Aware Sampling

This document describes the new training pipeline scripts for testing density-aware sampling in RLHF to reduce mode collapse.

## Overview & Hypothesis

**Main Hypothesis**: Modern RL in LLMs suffers from mode collapse (mode-seeking behavior), causing:

- Overuse of em-dashes and repetitive patterns
- Difficulty following instructions
- Less creativity in writing
- Grammar regressions (especially in non-English contexts)
- Inability to dynamically adjust personality

**Proposed Solution**: Instead of uniformly sampling from preference pairs during RLHF, sample **inversely proportional to prompt density**. This exposes the model to more diverse prompts, potentially smoothing the loss landscape and reducing mode collapse.

**Key Components**:

1. Download diverse pairwise preference data
2. Deduplicate prompts and compute local density
3. Train a reward model on preferences
4. Train policy with GRPO using density-weighted sampling
5. Compare outputs to baseline (uniform sampling)

## New Scripts

All scripts follow the `kat_` naming convention.

### 1. `scripts/kat_download_pairs.py`

**Purpose**: Download and consolidate pairwise preference data from multiple HuggingFace datasets.

**Datasets included**:

- `Anthropic/hh-rlhf`: Red Teaming dataset with helpful/harmless/honest preference pairs
- `HuggingFaceH4/ultrafeedback_binarized`: Binarized feedback dataset with preference pairs
- `HuggingFaceH4/stack-exchange-preferences`: Stack Exchange Q&A with upvoted/downvoted answers

**Output**: `.cache/data/pairs_all.jsonl`

- Each line is a JSON object: `{"id", "prompt", "chosen", "rejected", "src"}`
- `id`: UUID for this pair
- `prompt`: The input question/request
- `chosen`: The preferred response
- `rejected`: The non-preferred response
- `src`: Source dataset identifier

**Features**:

- Whitespace normalization for consistent processing
- HTML stripping for Stack Exchange data
- Duplicate detection by comparing prompts across datasets
- Optional filtering by dataset

**Usage**:

```bash
# Download all three datasets (default)
python -m scripts.kat_download_pairs

# Skip a specific dataset
python -m scripts.kat_download_pairs --no-hh
python -m scripts.kat_download_pairs --no-uf
python -m scripts.kat_download_pairs --no-se

# Download only one dataset
python -m scripts.kat_download_pairs --only hh
python -m scripts.kat_download_pairs --only uf
python -m scripts.kat_download_pairs --only se
```

---

### 2. `scripts/kat_make_prompts.py`

**Purpose**: Extract and deduplicate prompts from the consolidated pairs data.

**Input**: `.cache/data/pairs_all.jsonl` (from `kat_download_pairs.py`)

**Outputs**:

- `.cache/data/prompts_all.jsonl`: Deduplicated prompts in JSONL format
  - Each line: `{"id", "prompt"}`
  - `id`: MD5 hash of prompt (first 16 chars) for deterministic matching
- `.cache/data/prompt_id_map.tsv`: Tab-separated ID-to-prompt mapping for easy lookup
- `.cache/data/stats.txt`: Statistics about the consolidated dataset
  - Total pairs processed
  - Unique prompts found
  - Breakdown by source dataset

**Why deduplication?**:

- The three datasets may have overlapping prompts
- Deduplication avoids training bias from redundant prompts
- Prepares for density-based sampling (each unique prompt sampled based on local density)
- Reduces storage requirements

**Usage**:

```bash
# Process pairs to extract and deduplicate prompts
python -m scripts.kat_make_prompts
```

---

### 3. `scripts/kat_train_rm.py`

**Purpose**: Train a Reward Model (RM) on pairwise preferences.

This script:

1. Loads an SFT checkpoint (e.g., from `scripts.chat_sft`)
2. Adds a scalar reward head (linear layer)
3. Trains on preference pairs using Bradley-Terry loss
4. Saves the trained RM checkpoint

**Why RM first?**: The RM learns to score responses by predicting preferences. This enables GRPO to use a learned reward signal instead of hand-crafted ones.

**Input Requirements**:

- `.cache/data/pairs_all.jsonl` (from `kat_download_pairs.py`)
- `outs/sft/ckpt.pt` (SFT checkpoint)

**Output**: `outs/rm/ckpt.pt`

- Contains trained reward head weights
- Configuration saved for reconstruction

**Loss Function**: Bradley-Terry preference loss

```
loss = -log(sigmoid(r_chosen - r_rejected))
```

This maximizes the probability that chosen response gets higher reward than rejected.

**Key Hyperparameters**:

- `--max_steps`: Number of training steps (default: 1000)
- `--learning_rate`: RM head learning rate (default: 1e-4)
- `--device_batch_size`: Batch size per device (default: 32)

**Usage**:

```bash
# Train reward model
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_rm

# Custom hyperparameters
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_rm -- \
  --learning_rate=5e-4 \
  --max_steps=2000 \
  --device_batch_size=16

# Custom paths
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_rm -- \
  --sft_ckpt_path outs/sft/ckpt.pt \
  --pairs_path .cache/data/pairs_all.jsonl \
  --out_path outs/rm/ckpt.pt
```

**Typical Results**:

- Training loss should decrease from ~0.7 to ~0.2-0.3
- Training time: 2-4 hours on 8xH100 with full dataset

---

### 4. `scripts/kat_train_grpo.py`

**Purpose**: Train policy with GRPO using density-aware sampling to reduce mode collapse.

This script:

1. Loads SFT checkpoint as the policy
2. Loads trained RM checkpoint for reward scoring
3. **Computes prompt embeddings and local density** (key innovation)
4. **Samples preference pairs inversely proportional to density** (if enabled)
5. Optimizes policy using GRPO loss with KL divergence penalty
6. Tests hypothesis that diversity-aware sampling reduces mode collapse

**Key Innovation - Density-Aware Sampling**:

- Compute embeddings for each unique prompt
- Estimate local prompt density using k-NN
- Weight samples inversely: `weight = 1 / local_density`
- High-density prompts (common): lower sampling weight
- Low-density prompts (rare): higher sampling weight
- **Result**: Model learns from diverse prompts, reducing mode collapse

**Input Requirements**:

- `outs/sft/ckpt.pt` (SFT checkpoint)
- `outs/rm/ckpt.pt` (Trained RM checkpoint)
- `.cache/data/pairs_all.jsonl` (preference pairs)
- `.cache/data/prompts_all.jsonl` (unique prompts for density computation)

**Output**: `outs/grpo/ckpt.pt`

- Trained policy checkpoint
- Config includes `density_aware` flag for reproducibility

**Key Hyperparameters**:

- `--max_steps`: Total training steps (default: 5000)
- `--learning_rate`: Policy learning rate (default: 1e-5)
- `--beta`: KL divergence penalty strength (default: 0.1)
  - Higher beta = closer to SFT (less drift)
  - Lower beta = more reward optimization
- `--density_aware`: Enable/disable density weighting (default: True)
- `--density_k`: k for k-NN density estimation (default: 10)

**GRPO Loss Function**:

```
loss = -log(sigmoid(r_chosen - r_rejected)) + beta * KL(policy || reference)
```

- Preference term: maximize chosen reward vs rejected
- KL term: don't diverge too much from SFT distribution

**How KL Divergence Works**:

1. SFT model is loaded twice: once as trainable policy, once as frozen reference
2. Both policy and reference process same inputs (chosen/rejected responses)
3. KL divergence computed: `KL(policy_logits || reference_logits)` for both responses
4. Final KL loss is average of chosen and rejected KL values
5. **Effect**: Prevents policy from drifting too far from SFT baseline while still optimizing for rewards

**Usage**:

```bash
# Train with density-aware sampling (main experiment)
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo

# Train without density sampling (baseline for comparison)
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- --density_aware=False

# Adjust KL penalty
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- --beta=0.05

# Faster iteration (fewer steps)
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- --max_steps=1000

# Custom paths
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- \
  --sft_ckpt_path outs/sft/ckpt.pt \
  --rm_ckpt_path outs/rm/ckpt.pt \
  --pairs_path .cache/data/pairs_all.jsonl \
  --prompts_path .cache/data/prompts_all.jsonl \
  --out_dir outs/grpo/
```

**Typical Results**:

- Training loss should decrease gradually
- Sampling weight distribution should show higher variance with density weighting
- Training time: 4-8 hours on 8xH100

---

## Data Flow

```
[HuggingFace Datasets]
    |
    v
[kat_download_pairs.py]
    |
    v
.cache/data/pairs_all.jsonl  (276k pairs)
    |
    +---> [kat_make_prompts.py]
    |           |
    |           v
    |     .cache/data/prompts_all.jsonl  (89k unique)
    |
    +---> [kat_train_rm.py]
    |     (uses SFT checkpoint)
    |           |
    |           v
    |     outs/rm/ckpt.pt  (Reward Model)
    |
    v
[kat_train_grpo.py]
    |
    +---> outs/grpo/ckpt.pt  (with density sampling)
    |
    +---> outs/grpo_baseline/ckpt.pt  (uniform sampling)
```

---

## Experimental Design: Testing the Hypothesis

To properly validate the hypothesis, you should:

1. **Train with density sampling** (main experiment):

   ```bash
   torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo
   ```

2. **Train without density sampling** (control):

   ```bash
   torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- --density_aware=False
   ```

3. **Generate outputs** from both models and **compare**

---

## Evaluation: Measuring Mode Collapse Reduction

After training both models, test the hypothesis by measuring:

### Target Artifacts (should reduce with density sampling):

- **Em-dash frequency**: Model uses em-dashes less often
- **Repetitive patterns**: Less "not just X but Y" type constructions
- **Grammar errors**: Fewer grammar mistakes in non-English contexts
- **Token diversity**: Flatter token frequency distribution (higher Gini coefficient = more collapse)
- **Personality consistency**: Better ability to vary tone based on context

### Quick Evaluation Script

```bash
# Generate outputs from both models
python -m scripts.chat_cli --ckpt_path outs/grpo/ckpt.pt > outputs_density.txt
python -m scripts.chat_cli --ckpt_path outs/grpo_baseline/ckpt.pt > outputs_baseline.txt

# Count em-dashes
echo "Em-dashes (density):" && grep -o "—" outputs_density.txt | wc -l
echo "Em-dashes (baseline):" && grep -o "—" outputs_baseline.txt | wc -l

# Token frequency analysis (run the Gini script from CLAUDES_HOWTO.md)
```

---

## Storage Considerations

The datasets are large:

- HH-RLHF: ~160k pairs
- UltraFeedback: ~64k pairs
- Stack Exchange: ~52k pairs
- **Total**: ~276k pairs (significantly reduced after deduplication)

**File Sizes** (uncompressed):

- `pairs_all.jsonl`: ~500 MB
- `prompts_all.jsonl`: ~50 MB
- `prompt_id_map.tsv`: ~50 MB

**Optimization Tips**:

- Download only one dataset at a time if storage is limited
- Use gzip compression if needed
- Delete intermediate files after processing

---

## Dependencies

These scripts require:

- `torch` & `torchvision`: PyTorch for deep learning
- `datasets`: HuggingFace datasets library
- `sklearn`: For k-NN density estimation
- `tensorboard`: For training visualization
- Standard library: `argparse`, `json`, `os`, `hashlib`, `re`, `collections`, `sys`, `pathlib`

All should be available in the nanochat environment.

---

## Troubleshooting

### Problem: "SFT checkpoint not found"

**Solution**: Run SFT training first:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
```

### Problem: Out of Memory

**Solution**:

1. Reduce batch size: `--device_batch_size=8`
2. Download fewer datasets: `--no-uf --no-se`
3. Use fewer training steps: `--max_steps=1000`

### Problem: RM loss not decreasing

**Debugging**:

1. Check pairs data: `head -5 .cache/data/pairs_all.jsonl`
2. Verify chosen/rejected are different
3. Try higher learning rate: `--learning_rate=5e-4`

### Problem: Slow GRPO training

**Solutions**:

- Disable density computation for initial testing: `--density_aware=False`
- Reduce max_steps
- Use smaller k: `--density_k=5`

---

## Future Enhancements

Potential improvements:

1. **Semantic embeddings**: Use sentence-transformers instead of hash-based embeddings
2. **Adaptive density**: Recompute density periodically during training
3. **Multiple RM heads**: Train separate RMs for different aspect (quality, diversity, helpfulness)
4. **Curriculum learning**: Start with high-density prompts, gradually shift to low-density
5. **Analysis tools**: Scripts to visualize prompt density distribution and sampling weights
6. **Integration metrics**: Compare to papers on diversity-focused RL (cited in README)

---

## References & Related Work

The hypothesis builds on recent work on mode collapse in RL:

- https://arxiv.org/abs/2501.18101 (Diversity in RL)
- https://arxiv.org/html/2509.04784v2 (Mode collapse analysis)
- https://www.arxiv.org/abs/2510.14901 (RL diversity metrics)

This implementation tests a specific mechanism: density-aware sampling.

---

**Created**: October 2025  
**Focus**: Testing density-aware sampling to reduce mode collapse in LLM RLHF training  
**Status**: Experimental - use for hypothesis validation, not production
