# Offline Embeddings Guide: Precompute Density Weights

## Overview

Instead of computing embeddings on-the-fly during training (which takes ~3 minutes), you can **precompute them once** as a dataset preparation step. This is especially useful when:

- You want training to start immediately (no 3-min delay)
- You're running multiple training experiments on the same dataset
- You want to scale to larger datasets (100k+ prompts)
- You want reproducible/frozen embeddings

---

## Pipeline Overview

```
Old Pipeline (Online):
  kat_download_pairs.py
    ↓
  kat_make_prompts.py
    ↓
  kat_train_grpo.py  ← Spends 3 min computing embeddings here
    ↓
  Start training (finally!)

New Pipeline (Offline):
  kat_download_pairs.py
    ↓
  kat_make_prompts.py
    ↓
  kat_compute_embeddings_offline.py  ← Precompute once
    ↓ (outputs: embeddings.npy, density_weights.npy)
    ↓
  kat_train_grpo.py  ← Training starts immediately!
    ↓
  Start training (no delay!)
```

---

## Step 1: Download Preference Pairs

Same as before:

```bash
python -m scripts.kat_download_pairs --only hh
```

Output: `data/pairs_all.jsonl` (~28k preference pairs)

---

## Step 2: Extract Unique Prompts

Same as before:

```bash
python -m scripts.kat_make_prompts
```

Output: `data/prompts_all.jsonl` (~28k unique prompts)

---

## Step 3: Precompute Embeddings & Density Weights (NEW!)

```bash
python -m scripts.kat_compute_embeddings_offline \
    --base_model_source base \
    --output_dir $HOME/.cache/nanochat/data/embeddings_offline
```

**What this does:**

1. Loads the base model
2. Encodes all 28k prompts through the model in batches
3. Computes local density for each prompt using k-NN
4. Saves embeddings and weights to disk

**Time:** ~5-10 minutes (one-time, CPU-friendly)

**Output files:**

```
~/.cache/nanochat/data/embeddings_offline/
├── embeddings.npy              (5.25 GB) - 28k x 50304 float32 array
├── density_weights.npy         (112 KB)  - 28k weights summing to 1.0
├── prompts_list.json           (JSON)    - ordered list of prompts
└── embeddings_metadata.json    (JSON)    - statistics and parameters
```

---

## Step 4: Train GRPO with Precomputed Embeddings

Instead of the old way, use the new flag:

```bash
torchrun --nproc_per_node=8 -m scripts.kat_train_grpo \
    --use_precomputed_embeddings \
    --embeddings_dir $HOME/.cache/nanochat/data/embeddings_offline \
    --max_steps 5000 \
    --learning_rate 1e-5 \
    --beta 0.1 \
    --out_dir outs/grpo_density
```

**What happens:**

1. Training loads pre-computed embeddings (instant!)
2. Density weights already available (no k-NN computation)
3. WeightedRandomSampler uses pre-computed weights
4. Training starts immediately

**Time to start:** <1 second (vs 3 minutes before!)

---

## Command Reference

### Basic Usage

```bash
# Simplest (uses defaults)
python -m scripts.kat_compute_embeddings_offline

# With custom k value
python -m scripts.kat_compute_embeddings_offline --k 5

# Larger batches (faster if you have GPU memory)
python -m scripts.kat_compute_embeddings_offline --batch_size 32

# Use mid-trained model instead of base
python -m scripts.kat_compute_embeddings_offline --base_model_source mid

# Custom output directory
python -m scripts.kat_compute_embeddings_offline \
    --output_dir /my/custom/path/embeddings
```

### Parameters

| Parameter             | Default             | Description                                             |
| --------------------- | ------------------- | ------------------------------------------------------- |
| `--base_model_source` | `base`              | Which model to extract embeddings from (base\|mid\|sft) |
| `--prompts_path`      | Auto-detect         | Path to prompts_all.jsonl                               |
| `--output_dir`        | Auto-detect         | Where to save embeddings                                |
| `--k`                 | `10`                | Number of neighbors for density computation             |
| `--batch_size`        | `8`                 | Prompts per batch (increase for faster processing)      |
| `--device`            | `cuda` if available | Device to use (cuda\|cpu)                               |

---

## Integration with Updated kat_train_grpo.py

The training script needs to support loading precomputed embeddings. Add this to `kat_train_grpo.py`:

```python
parser.add_argument(
    "--use_precomputed_embeddings",
    action="store_true",
    help="Load pre-computed embeddings instead of computing online"
)
parser.add_argument(
    "--embeddings_dir",
    default=None,
    help="Path to precomputed embeddings directory"
)

# ... later in main() ...

if args.use_precomputed_embeddings:
    print(f"Loading pre-computed embeddings from {args.embeddings_dir}...")
    embeddings = np.load(
        os.path.join(args.embeddings_dir, "embeddings.npy")
    )
    weights = np.load(
        os.path.join(args.embeddings_dir, "density_weights.npy")
    )
    weights_tensor = torch.from_numpy(weights).float()

    sampler = WeightedRandomSampler(
        weights_tensor,
        len(dataset),
        replacement=True
    )
    print(f"✓ Loaded {embeddings.shape[0]} pre-computed embeddings")
else:
    # Original online computation
    sampler = DensityAwareSampler(...)
```

---

## Metadata Output

The script saves `embeddings_metadata.json` with useful information:

```json
{
  "n_prompts": 28000,
  "embedding_dim": 50304,
  "embedding_dtype": "float32",
  "weights_dtype": "float32",
  "model_vocab_size": 50304,
  "model_n_embd": 768,
  "normalization": "L2 unit vectors",
  "weights_min": 0.000045,
  "weights_max": 0.000892,
  "weights_mean": 0.000036,
  "weights_std": 0.000067
}
```

---

## Comparison: Online vs Offline

| Aspect                    | Online                  | Offline                         |
| ------------------------- | ----------------------- | ------------------------------- |
| **Training startup time** | 3 minutes               | <1 second                       |
| **RAM during training**   | 7-10 GB                 | 5-6 GB                          |
| **Embeddings freshness**  | Always fresh from model | Fixed at compute time           |
| **Reusability**           | N/A                     | High (one compute, many trains) |
| **Disk space**            | N/A                     | 2.5 GB (compressed: ~1 GB)      |
| **Complexity**            | Simpler (1 script)      | Slightly more (2 scripts)       |
| **Scaling**               | Works for ~300k prompts | Works for millions              |

---

## When to Use Each

### Use **Online** If:

- ✅ You're doing a one-off experiment
- ✅ You have <100k prompts
- ✅ You want simplicity
- ✅ You're running small experiments quickly

### Use **Offline** If:

- ✅ You're running multiple experiments on same data
- ✅ You have >100k prompts
- ✅ You want instant training startup
- ✅ You want reproducible/frozen embeddings
- ✅ You're distributing data across team

---

## Advanced: Batch Processing for Large Datasets

For 1M+ prompts, process in chunks:

```bash
# Process first 100k prompts
python -m scripts.kat_compute_embeddings_offline \
    --prompts_path data/prompts_part1.jsonl \
    --output_dir data/embeddings_part1

# Process second 100k prompts
python -m scripts.kat_compute_embeddings_offline \
    --prompts_path data/prompts_part2.jsonl \
    --output_dir data/embeddings_part2

# Merge k-NN indices (use FAISS for this)
# python -m scripts.kat_merge_embeddings \
#     --parts data/embeddings_part1 data/embeddings_part2 \
#     --output data/embeddings_merged
```

---

## Troubleshooting

### "No valid prompts found"

Run `kat_make_prompts.py` first to generate `prompts_all.jsonl`

### "CUDA out of memory"

- Lower `--batch_size` (default 8, try 4 or 2)
- Use `--device cpu` (slower but always works)

### "Embeddings file is huge"

- Normal: 28k × 50k float32 = 5.25 GB
- Can compress with numpy: `np.savez_compressed()` (saves ~1 GB)

### Embeddings differ between runs

- Expected: floating point operations have small numerical variations
- If you need exact reproducibility: set `PYTHONHASHSEED=0 torch.manual_seed(42)`

---

## Next Steps

1. Run the offline embedding script
2. Update `kat_train_grpo.py` to support `--use_precomputed_embeddings` flag
3. Update `kat_speedrun.sh` to call offline embedding script
4. Update `kat_speedrun.sh` to pass embeddings dir to training

---

## Files Reference

- **Creator:** `scripts/kat_compute_embeddings_offline.py`
- **Consumer:** `scripts/kat_train_grpo.py` (needs updates)
- **Configuration:** Updated shell script `kat_speedrun.sh`
