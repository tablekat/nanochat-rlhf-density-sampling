# 3D Embedding Space Visualization

Visualize conversation/prompt embedding space in 3D with cluster-preserving projection (UMAP). See how density-aware sampling affects prompt distribution.

## What It Does

The visualization shows:

- **Each point** = One prompt/conversation from the dataset
- **Point color** = Source dataset (HH-RLHF=green, UltraFeedback=blue, Stack Exchange=red)
- **Point size** = Inverse of local density (rare prompts = bigger, common prompts = smaller)
- **3D space** = Cluster-preserving projection from embedding dimension
- **Interactions** = Hover for preview, click to view in chat

## Quick Start

### Step 1: Compute Embeddings & Project to 3D

```bash
# Uses sentence-transformers for embeddings + UMAP for 3D projection
# Computes local density to show clustering
python -m scripts.kat_viz_embeddings \
    --pairs_path .cache/data/pairs_all.jsonl \
    --output_path .cache/embeddings_3d.json
```

**Time**: ~30-60 minutes (depends on number of pairs, ~276k pairs)

**Output**: `.cache/embeddings_3d.json` (~50-100 MB)

### Step 2: Launch Web Server

```bash
# Start chat web server
python -m scripts.chat_web --num-gpus 1
```

Server runs at: `http://localhost:8000`

### Step 3: View Visualization

Navigate to: **`http://localhost:8000/viz`**

## Features

### Visualization Controls

**Left sidebar** has:

- **Point Size** slider - Adjust size of all points (0.1x to 5x)
- **Opacity** slider - Transparency of point cloud
- **Color legend** - Dataset source mapping
- **Statistics** - Total points, datasets, average density, selected count

### Interaction

| Control        | Action                            |
| -------------- | --------------------------------- |
| **Left drag**  | Rotate view                       |
| **Right drag** | Pan view                          |
| **Scroll**     | Zoom in/out                       |
| **Hover**      | Preview prompt (density %)        |
| **Click**      | Select and view full conversation |

### Visual Features

- **Auto-rotation** when idle (stops on hover)
- **Reference grid** and axes for orientation
- **Real-time statistics** update as you explore
- **Preview pane** shows:
  - Full prompt text
  - Source dataset
  - Local density percentage
  - Selected conversation details

## Understanding Density

**Density percentage** shown for each point represents:

- **0-25%** = Very rare prompt (unique topic, unusual phrasing)
- **25-50%** = Uncommon prompt
- **50-75%** = Common prompt
- **75-100%** = Very common prompt (clustered with many others)

**Why density matters for GRPO:**

- High-density prompts (clusters): Uniform sampling underrepresents
- Low-density prompts (rare): Model sees fewer unique variations
- Density-aware sampling: Weights inversely to balance exposure

## Customization

### Adjust Embedding Model

```bash
python -m scripts.kat_viz_embeddings \
    --pairs_path .cache/data/pairs_all.jsonl \
    --model_name sentence-transformers/all-mpnet-base-v2  # Larger model
```

Available models:

- `sentence-transformers/all-MiniLM-L6-v2` (fast, 384-dim) ← default
- `sentence-transformers/all-mpnet-base-v2` (slower, 768-dim)
- `sentence-transformers/all-roberta-large-v1` (slow, 1024-dim)

### Adjust UMAP Parameters

```bash
python -m scripts.kat_viz_embeddings \
    --pairs_path .cache/data/pairs_all.jsonl \
    --n_neighbors 30      # Higher = more global structure (default: 15)
    --min_dist 0.01       # Lower = tighter clusters (default: 0.1)
    --density_k 20        # k for k-NN density (default: 10)
```

- **n_neighbors**: Controls balance between local clusters vs global structure
  - Higher (30-50): See broad organization
  - Lower (5-10): See tight local clusters
- **min_dist**: Minimum distance between embedded points

  - Higher (0.5): Points spread out
  - Lower (0.01): Points cluster tightly

- **density_k**: Neighbors for local density estimation
  - Higher: Smoother density field
  - Lower: More granular

### Limit for Quick Testing

```bash
python -m scripts.kat_viz_embeddings \
    --pairs_path .cache/data/pairs_all.jsonl \
    --max_samples 5000  # Only use first 5000 pairs (runs in ~5 min)
```

## Analyzing Results

### Good Density Sampling Outcome

You'd see:

- **Distributed clusters** - Points spread evenly across space
- **Similar-sized points** - Fewer tiny points (means density weighting worked)
- **Varied densities** - Mix of 20%, 60%, 90% density points across space
- **Color mixing** - Datasets well-distributed, not segregated

### Poor Density Sampling Outcome

You'd see:

- **One huge cluster** - All points in small region (mode collapse)
- **Tiny points** - Many low-density outliers (unbalanced)
- **Color segregation** - Datasets form separate clusters (imbalanced sampling)
- **Sparse regions** - Large empty areas in space

## Integration with GRPO

The visualization helps validate the hypothesis:

1. **Before GRPO**: Run embedding projection on preference pairs

   - See baseline prompt distribution
   - Identify dense clusters vs rare prompts

2. **After GRPO Training**: Run embedding projection on generated outputs

   - Compare distribution to input pairs
   - Check if density-aware training improved coverage

3. **Compare Models**:
   - Generate outputs from both density-aware and baseline models
   - Project new outputs to see difference

## Technical Details

### How It Works

1. **Embedding**: Sentence-transformers encodes each prompt to 384/768/1024-dim vector
2. **Projection**: UMAP reduces to 3D preserving cluster structure
3. **Density**: k-NN computes local density (inverse of avg distance to k neighbors)
4. **Visualization**: Three.js renders as interactive point cloud with hover/click

### Performance Notes

| Metric              | Time        | Memory    |
| ------------------- | ----------- | --------- |
| Embed 89k prompts   | ~45 min     | ~8 GB     |
| UMAP project to 3D  | ~15 min     | ~4 GB     |
| Density computation | ~5 min      | ~2 GB     |
| **Total**           | **~65 min** | **~8 GB** |

With `--max_samples 5000`: ~5 minutes total

### File Sizes

- `embeddings_3d.json` (276k points): ~100-150 MB
- Compressed (gzip): ~20-30 MB

## Troubleshooting

### "3D visualization not found"

```bash
# Run the embedding script first
python -m scripts.kat_viz_embeddings
```

### "Embedding data not found"

```bash
# Same fix
python -m scripts.kat_viz_embeddings
```

### Slow performance in browser

Try these:

1. Reduce max samples: `--max_samples 10000`
2. Lower browser zoom level
3. Close other tabs
4. Use Chrome/Brave instead of Firefox (better WebGL)

### Crash or freezes

Likely out of memory. Try:

```bash
# Use smaller model
python -m scripts.kat_viz_embeddings \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --max_samples 5000
```

## Example Workflow

```bash
# 1. Download pairs
python -m scripts.kat_download_pairs --only hh

# 2. Deduplicate
python -m scripts.kat_make_prompts

# 3. Compute embeddings & project (40-60 min)
python -m scripts.kat_viz_embeddings

# 4. Start web server
python -m scripts.chat_web

# 5. Open browser to http://localhost:8000/viz
# 6. Explore! Rotate, zoom, hover, click

# 7. Later: Generate outputs from trained models
python -m scripts.chat_cli --ckpt_path outs/grpo_density/ckpt.pt > outputs.txt

# 8. Compare by creating new embeddings from outputs
# and projecting again
```

## Files Created

```
.cache/embeddings_3d.json  # 3D projection data
nanochat/ui_3d.html        # 3D visualization UI  (added to web server)
scripts/kat_viz_embeddings.py  # Projection script
```

---

**Next**: Navigate to `/viz` in your running chat web server to explore!
