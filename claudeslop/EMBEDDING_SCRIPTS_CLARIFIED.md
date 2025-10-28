# Embedding Scripts: kat_embed.py vs kat_compute_embeddings_offline.py

## Quick Comparison

| Feature                 | `kat_embed.py`               | `kat_compute_embeddings_offline.py` |
| ----------------------- | ---------------------------- | ----------------------------------- |
| **Purpose**             | Quick embedding on-demand    | Offline precomputation for training |
| **Model Used**          | Trained (SFT/RM/GRPO/etc)    | Configurable (default: base)        |
| **Input**               | Any JSONL with prefix/prompt | `prefixes_all.jsonl`                |
| **Density Computation** | ❌ No                        | ✅ Yes (k-NN based)                 |
| **Output Files**        | Single `.pt` file            | Multiple `.npy` + `.json` files     |
| **Use Case**            | Development/exploration      | GRPO/RM training                    |

---

## Detailed Explanation

### `kat_embed.py` - Flexible Embeddings

**What it does:**

- Takes ANY JSONL file with `id` and either `prefix` (conversation) or `prompt` (string) fields
- Extracts text intelligently from prefix objects (concatenates all messages)
- Runs forward pass through a **trained model** (default: SFT checkpoint)
- Averages hidden states across all tokens
- L2-normalizes embeddings
- Outputs a single `.pt` file

**Output format:**

```python
torch.save({
    "ids": ["id1", "id2", ...],          # Original IDs from input JSONL
    "emb": tensor([N, 768])              # Embeddings (L2-normalized, float32)
}, "output.pt")
```

**When to use:**

- Quick exploration or development
- Computing embeddings for custom data
- Using a specific trained model checkpoint
- One-off embedding tasks

**Example:**

```bash
python -m scripts.kat_embed \
  --ckpt_source sft \
  --data data/my_data.jsonl \
  --out data/my_embeddings.pt \
  --batch_size 32
```

---

### `kat_compute_embeddings_offline.py` - Density-Aware Precomputation

**What it does:**

- Takes `prefixes_all.jsonl` (deduplicated prefixes)
- Extracts text from prefix objects
- Runs forward pass through **configurable model** (default: base, but can use sft/mid/grpo/rm)
- Averages hidden states, L2-normalizes
- **Computes k-NN density weights** (local density estimation)
- Inverts weights so rare prompts get higher sampling weight
- Outputs multiple files for efficient loading during training

**Output files (in `--output_dir`):**

```
embeddings_offline/
├── embeddings.npy           # [28k, 384] float32 - text embeddings
├── density_weights.npy      # [28k,] float32 - inverse density weights
├── ids.json                 # Ordered list of item IDs (maps to prefixes_all.jsonl)
└── embeddings_metadata.json # Stats: n_items, embedding_dim, weight statistics
```

**When to use:**

- Training reward models with density awareness
- Training GRPO policies with density-aware sampling
- One-time preprocessing before large-scale training
- Need to preserve index ordering for efficient lookup

**Example:**

```bash
# With default base model
python -m scripts.kat_compute_embeddings_offline \
  --prompts_path data/prefixes_all.jsonl \
  --output_dir data/embeddings_offline \
  --k 20 \
  --batch_size 16

# With trained SFT model
python -m scripts.kat_compute_embeddings_offline \
  --base_model_source sft \
  --prompts_path data/prefixes_all.jsonl \
  --output_dir data/embeddings_offline_sft \
  --k 20
```

---

## Key Differences

### 1. **Model Used**

- `kat_embed.py`: Fixed model source (defaults to SFT, configurable via `--ckpt_source`)
- `kat_compute_embeddings_offline.py`: Configurable model source (defaults to base)

### 2. **Density Computation**

- `kat_embed.py`: ❌ Just embeddings, no density weights
- `kat_compute_embeddings_offline.py`: ✅ **KEY DIFFERENCE** - Computes density + inverse weights

### 3. **Output Format**

- `kat_embed.py`: Single `.pt` file (PyTorch tensors) - fast to load in memory
- `kat_compute_embeddings_offline.py`: NumPy arrays + metadata JSON - efficient for large-scale training

### 4. **Data Preservation**

- `kat_embed.py`: Only saves `ids` and embeddings
- `kat_compute_embeddings_offline.py`: Saves `ids` (not full items to avoid duplication)

### 5. **Embedding Dimension**

- `kat_embed.py`: Model-specific (usually 768 dims)
- `kat_compute_embeddings_offline.py`: Model-specific (768 for trained models, 384 for sentence-transformers)

### 6. **Primary Use Cases**

- `kat_embed.py`: Flexible, on-demand embeddings for exploration
- `kat_compute_embeddings_offline.py`: Precomputed with density weights for scalable training

---

## Why `ids.json` Instead of Full Items?

Previously, `kat_compute_embeddings_offline.py` saved `items_list.json` containing full prefix objects. This was:

❌ **Redundant** - Full items already in `prefixes_all.jsonl`  
❌ **Wasteful** - Stores ~28k full conversation objects (MB of data)  
❌ **Unnecessary** - Training code only needs to map embeddings → prefix IDs

**Fix:** Now saves only `ids.json`:
✅ Tiny file (~few KB)  
✅ Maps embedding index → prefix ID  
✅ Training code can load prefixes on-demand from `prefixes_all.jsonl`

---

## Workflow Integration

```
pairs_all.jsonl (170k pairs)
         ↓
   kat_make_prefixes.py
         ↓
prefixes_all.jsonl (28k unique)
         ↓
   ┌─────┴─────┐
   ↓           ↓
kat_embed.py  kat_compute_embeddings_offline.py
   ↓           ↓
prefixes_emb.pt  embeddings.npy + density_weights.npy + ids.json
   ↓           ↓
[Development]  [Training with GRPO/RM]
```

---

## Loading Precomputed Embeddings in Training

```python
# In kat_train_grpo.py or kat_train_rm.py
import numpy as np
import json

# Load embeddings
embeddings = np.load("data/embeddings_offline/embeddings.npy")  # [28k, 384]
weights = np.load("data/embeddings_offline/density_weights.npy")  # [28k,]
ids = json.load(open("data/embeddings_offline/ids.json"))  # List of IDs

# ids[i] tells you which prefix_id corresponds to embeddings[i]
# Use this to map density weights back to pairs during training
```

---

## How `kat_train_rm.py` Uses Precomputed Files

Here's the concrete workflow:

```
1. Load density_weights.npy + prefixes_all.jsonl
   ↓
2. Build density mapping: {prefix_id → weight}
   - Read IDs from prefixes_all.jsonl
   - Load weights from density_weights.npy
   - Create dict: id[i] → weights[i]
   ↓
3. Load pairs from pairs_all.jsonl
   ↓
4. For each pair:
   a. Extract prefix → extract first user message → hash to ID
   b. Look up ID in density mapping → get weight
   c. Assign weight to pair (used in loss scaling)
   ↓
5. Train with weighted loss:
   loss = sum(weight[i] * bradley_terry_loss[i]) / batch_size
```

**Key code sections from `kat_train_rm.py`:**

**Loading the density mapping** (lines 147-155):

```python
def load_density_mapping(prompts_path: Path, weights_path: Path) -> Dict[str, float]:
    """Build prompt_id -> inverse-density weight mapping."""
    ids: List[str] = []
    with prompts_path.open("r", encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)["id"])
    weights = np.load(weights_path)
    assert len(ids) == len(weights), "prefixes_all.jsonl and density_weights.npy misaligned"
    return {pid: float(w) for pid, w in zip(ids, weights.tolist())}
```

**Using weights in training** (lines 269-273):

```python
density = None
if density_aware:  # rm_source == "rm_density"
    print0(f"Loading density weights from {density_weights_path}...")
    prefixes_path = os.path.join(base, "data", "prefixes_all.jsonl")
    density = load_density_mapping(Path(prefixes_path), Path(density_weights_path))
```

**Applying weights per batch** (lines 315-316):

```python
loss_vec = bt_loss(rc, rr)  # [batch_size]
loss = apply_weights(loss_vec, w, weight_mode, weight_cap)  # weighted mean
```

---

## File Usage Summary

| File                       | How It's Used                              | Loaded By                       |
| -------------------------- | ------------------------------------------ | ------------------------------- |
| `embeddings.npy`           | ❌ Not used by RM training                 | Only for visualization/analysis |
| `density_weights.npy`      | ✅ Loaded to build `{id → weight}` mapping | `load_density_mapping()`        |
| `ids.json`                 | ✅ Maps density weights back to prefix IDs | Used when building mapping      |
| `embeddings_metadata.json` | ❌ Not used by training                    | Only for reference/debugging    |

**Bottom line:** RM training only needs `density_weights.npy` + `ids.json` (not the actual embeddings!)

- The embeddings are used for **computing** the density weights (k-NN)
- But the actual embeddings aren't needed during training
- Only the final weights matter
