# Data Flow Reference: Prefix-Based Density-Aware Pipeline

**Base directory:** `$HOME/.cache/nanochat/data/` (set by `get_base_dir()`)

---

## Overview

Full pipeline for downloading preference pairs, extracting unique prefixes, computing embeddings with density weights, and training reward models with density-aware sampling.

---

## Quick Start

```bash
# Setup
python -m scripts.kat_download_pairs           # Downloads pairs from 3 sources
python -m scripts.kat_make_prefixes            # Deduplicates by first user msg
python -m scripts.kat_compute_embeddings_offline  # Computes embeddings + density

# Train RM baseline (uniform sampling)
python -m scripts.kat_train_rm --max_steps=1000

# Train RM with density awareness
python -m scripts.kat_train_rm --density_aware --max_steps=1000

# Train GRPO policy with RM
python -m scripts.kat_train_grpo --rm_source rm --max_steps=5000
```

---

## Script Reference

### 1. `scripts/kat_download_pairs.py`

**Purpose:** Download preference pairs from HuggingFace datasets

**Inputs:** None (downloads from 3 remote sources)

**Outputs:**

- `~/.cache/nanochat/data/pairs_all.jsonl` (~170k lines)

**Format (each line):**

```json
{
  "id": "uuid",
  "prefix": {
    "messages": [
      { "role": "user", "content": "What is AI?" },
      { "role": "assistant", "content": "AI is..." },
      { "role": "user", "content": "More details?" }
    ]
  },
  "chosen": "Machine learning is a subset of AI...",
  "rejected": "I don't know.",
  "src": "hh-rlhf"
}
```

**Data shape:**

- ~169k pairs from HH-RLHF
- ~64k pairs from UltraFeedback-binarized
- ~31.3M pairs from Stack-Exchange-Preferences (but usually truncated)
- **Total: ~170-200k pairs** (budget-dependent)

**Key fields:**

- `prefix`: Full conversation object (dict with "messages" list)
  - Each message: `{"role": "user"|"assistant", "content": str}`
  - Last message is ALWAYS from user (ready for assistant response)
- `chosen`: String response from assistant (positive example)
- `rejected`: String response from assistant (negative example)
- `src`: Source dataset identifier

**Sources:**

- `hh-rlhf`: Anthropic HH-RLHF dataset
- `ultrafeedback-binarized`: UltraFeedback-binarized
- `stack-exchange-preferences`: StackExchange preferences

---

### 2. `scripts/kat_make_prefixes.py`

**Purpose:** Deduplicate pairs by first user message, store full prefix objects

**Inputs:**

- `pairs_all.jsonl` (~170k lines)

**Outputs:**

- `~/.cache/nanochat/data/prefixes_all.jsonl` (~28k lines, deduplicated)
- `~/.cache/nanochat/data/prefix_id_map.tsv` (for reference)
- `~/.cache/nanochat/data/stats.txt` (statistics)

**Format (prefixes_all.jsonl - each line):**

```json
{
  "id": "md5_hash_16chars",
  "prefix": {
    "messages": [
      { "role": "user", "content": "..." },
      { "role": "assistant", "content": "..." }
    ]
  }
}
```

**Data shape:**

- Deduplicates by: `md5(first_user_message)[:16]`
- Stores: **FULL prefix object** (entire conversation)
- ~28,000 unique first-user-messages from 170,000 pairs
- Dedup ratio: ~16% (84% duplicate prompts)

**Key points:**

- ID is deterministic: same prompt always gets same hash
- Stores complete conversation context (not just first prompt)
- Later used for embedding computation

---

### 4. `scripts/kat_compute_embeddings_offline.py`

**Purpose:** Precompute embeddings and density weights using sklearn

**Inputs:**

- `--prompts_path` JSONL with either prefix or prompt format
- `--model_name` default "base", mean pools hidden states to get embedding
- `--k` (default: 20, neighbors for density)

**Outputs (to `--output_dir`):**

```
embeddings_offline/
├── embeddings.npy           [N, 384]  (float32)
├── density_weights.npy      [N,]      (float32, normalized to sum=1)
├── ids.json                 (ordered list of item IDs, maps embeddings to prefixes_all.jsonl)
└── embeddings_metadata.json (config + stats)
```

**Usage:**

```bash
python -m scripts.kat_compute_embeddings_offline \
  --prompts_path data/prefixes_all.jsonl \
  --output_dir data/embeddings_offline \
  --k 20 \
  --batch_size 32
```

**Data shape:**

- Embeddings: `[28000, 384]` (sentence-transformers output)
- Weights: `[28000,]`, normalized to sum=1.0
- Higher weight = rarer prompt
- Formula: `weight = 1 / (density + eps)` where density = k-NN similarity

---

### 5. `scripts/kat_train_rm.py`

**Purpose:** Train reward model on preference pairs (optionally with density weighting)

**Inputs:**

- `pairs_all.jsonl` (170k pairs with full prefix objects)
- `prefixes_all.jsonl` (28k unique prefixes, for ID mapping)
- `density_weights.npy` (if `--density_aware` flag)

**Outputs:**

- Checkpoint: `~/.cache/nanochat/rm_checkpoints/d20/model_*.pt`

**Data flow:**

1. Load pairs (each with full `prefix` object)
2. For each pair: tokenize with `tokenizer.render_for_completion(prefix)`
3. Append chosen/rejected response tokens
4. If density-aware: assign weight per pair based on first-user-message hash
5. Use WeightedRandomSampler for training

**Command examples:**

```bash
# Baseline: uniform sampling
python -m scripts.kat_train_rm --max_steps=1000

# Density-aware: rare prompts ~100x more often
python -m scripts.kat_train_rm --density_aware --max_steps=1000 \
  --out_dir ~/.cache/nanochat/rm_checkpoints/d20_density
```

**Key processing:**

- Input prefix: `{"messages": [...]}` (full conversation)
- `tokenizer.render_for_completion(prefix)` → token sequence ending with `<|assistant_start|>`
- Append `chosen_tokens` or `rejected_tokens` to prefix
- Creates paired examples: (prefix+chosen, prefix+rejected)

---

### 6. `scripts/kat_train_dpo.py`

**Purpose:** Train policy with DPO (Direct Preference Optimization)

**Inputs:**

- `pairs_all.jsonl` (170k pairs with prefix objects)

**Outputs:**

- Checkpoint: `~/.cache/nanochat/dpo_checkpoints/model_*.pt`

**Data processing:**

- Each pair (prefix, chosen, rejected) becomes two training examples
- Uses `tokenizer.render_for_completion(prefix)` to get prefix tokens
- Appends chosen/rejected response tokens
- Computes DPO loss

**Command:**

```bash
python -m scripts.kat_train_dpo --pairs data/pairs_all.jsonl
```

---

### 7. `scripts/kat_train_grpo.py`

**Purpose:** Train policy with GRPO (Generative Reward Policy Optimization)

**Inputs:**

- `pairs_all.jsonl` (training pairs)
- RM checkpoint (from `kat_train_rm.py`)

**Outputs:**

- Checkpoint: `~/.cache/nanochat/grpo_checkpoints/model_*.pt`

**Command:**

```bash
python -m scripts.kat_train_grpo \
  --rm_source rm_density \
  --max_steps=5000
```

---

### 8. `scripts/kat_viz_embeddings.py`

**Purpose:** 3D visualization of prefix embeddings and density

**Inputs:**

- `pairs_all.jsonl` (with prefix objects)
- Embedding model

**Outputs:**

- `.cache/embeddings_3d.json` (JSON for web visualization)

**Extracts text from:**

- New format: concatenates all messages from prefix
- Legacy format: uses direct prompt field

---

## File Dependency Graph

```
┌─ HH-RLHF (HuggingFace)
├─ UltraFeedback (HuggingFace)
└─ Stack-Exchange (HuggingFace)
         ↓
    kat_download_pairs.py
         ↓
    pairs_all.jsonl (~170k)
    ├─ prefix: full conversation objects
    ├─ chosen: response string
    ├─ rejected: response string
    └─ src: dataset source
         ↓
    kat_make_prefixes.py
         ↓
    prefixes_all.jsonl (~28k, deduplicated)
    └─ prefix: full conversation objects
         ↓
    kat_compute_embeddings_offline.py
         ↓
    embeddings_offline/
    ├── embeddings.npy [28k, 384]
    ├── density_weights.npy [28k,]
    └── ids.json
         ↓
    ┌─────────────────────────────┐
    │ kat_train_rm.py             │
    │ (reads pairs_all.jsonl +    │
    │  density_weights.npy)       │
    └──────────┬──────────────────┘
               ↓
    rm_checkpoints/d20*/model_*.pt
               ↓
    ┌─────────────────────────────┐
    │ kat_train_grpo.py           │
    │ kat_train_dpo.py            │
    └─────────────────────────────┘
               ↓
    policy_checkpoints/model_*.pt
```

---

## Data Format Summary

| File                  | Format | Lines | Shape       | Key Fields                                    |
| --------------------- | ------ | ----- | ----------- | --------------------------------------------- |
| `pairs_all.jsonl`     | JSONL  | ~170k | —           | `prefix` (conversation), `chosen`, `rejected` |
| `prefixes_all.jsonl`  | JSONL  | ~28k  | —           | `id` (md5 hash), `prefix` (conversation)      |
| `embeddings.npy`      | NumPy  | —     | [28k, 384]  | float32 embeddings                            |
| `density_weights.npy` | NumPy  | —     | [28k,]      | float32, normalized to sum=1                  |
| `ids.json`            | JSON   | —     | [28k items] | Ordered list of item IDs                      |

---

## Validation Checklist

When running the full pipeline:

```
✓ kat_download_pairs.py → 170k+ pairs downloaded
✓ kat_make_prefixes.py → 28k unique prefixes extracted
✓ kat_compute_embeddings_offline.py → embeddings.npy [28k,384], weights [28k,]
✓ kat_train_rm.py --density_aware → All 170k pairs assigned weights
  - Weights span large range (1e-6 to 0.1+)
  - Rare prompts get 50-100x higher weight
✓ kat_train_grpo.py → Policy trained successfully
```

---

## Key Implementation Details

### Prefix Format

- **Input to training:** Full conversation object `{"messages": [...]}`
- **Last message:** ALWAYS from user (enables assistant response)
- **Multi-turn support:** HH-RLHF preserves full conversation history
- **Processing:** `tokenizer.render_for_completion(prefix)` adds `<|assistant_start|>` token

### Density Weighting

- **Computed on:** First user message (deterministic hashing)
- **Method:** k-NN density with cosine similarity
- **Formula:** `weight = 1 / (density + eps)`
- **Applied at:** RM training (weighted sampler)
- **Effect:** Rare prompts ~100x more training samples

### Backward Compatibility

All scripts handle both:

- **New format:** `prefix` dict with full conversation
- **Legacy format:** `prompt` string field

This enables graceful migration and mixed data pipelines.
