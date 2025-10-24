# Embeddings Guide: Complete Workflow

This document explains the three separate embedding systems in the KAT pipeline and how they work together.

---

## Quick Answer

**Yes, kat_embed.py pairs datasets with embeddings** - but embeddings are used for **multiple purposes**:

1. **kat_embed.py** → Offline embedding extraction (for analysis/retrieval)
2. **kat_train_grpo.py's DensityAwareSampler** → Real-time embeddings for density-aware sampling
3. **kat_viz_embeddings.py** → 3D visualization of prompt space

---

## System 1: kat_embed.py (Offline Embedding Extraction)

### Purpose

Extract semantic embeddings from conversations/prompts and save them for later use (retrieval, analysis, visualization).

### How It Works

```
Input Dataset (JSONL)
    ↓
[{"id": "...", "prompt": "...", "assistant": "..."}, ...]
    ↓
Model Forward Pass (with return_hidden_states=True)
    ↓
Hidden States Extraction
    ↓
Average Pooling + L2 Normalization
    ↓
Output Tensor {"ids": [...], "emb": [N, D]}
```

### Step-by-Step Process

```python
# 1. Load model and data
model, tokenizer, _ = load_model("sft", device, phase="eval")
rows = json.loads(...)  # Load conversations

# 2. For each batch:
for batch in rows:
    # Convert to chat format
    texts = [to_chat_text(r["prompt"], r.get("assistant")) for r in batch]

    # 3. Tokenize
    tokens = [tokenizer.encode(t) for t in texts]
    x = torch.tensor(tokens_padded)  # [B, T]

    # 4. Forward pass with new parameter
    output = model(x, return_hidden_states=True)
    logits = output['logits']          # [B, T, vocab_size]
    hidden_states = output['hidden_states']  # [B, T, n_embd]

    # 5. Average pooling (mean over sequence, masking padding)
    pad_mask = (x != pad_token_id).unsqueeze(-1)  # [B, T, 1]
    h = hidden_states * pad_mask       # Zero out padding
    h_pooled = h.sum(dim=1) / pad_mask.sum(dim=1)  # [B, n_embd]

    # 6. L2 normalize
    h_normalized = h_pooled / (||h_pooled|| + 1e-8)  # [B, n_embd]

    embeddings.append(h_normalized)

# 7. Save
torch.save({"ids": ids, "emb": embeddings}, "embeddings.pt")
```

### Output Format

```python
{
    "ids": ["id1", "id2", ...],          # Original row IDs
    "emb": torch.tensor([N, 768])        # N embeddings of dimension 768 (or n_embd)
}
```

### Usage After Extraction

```bash
# Offline use:
python scripts/kat_embed.py \
  --ckpt_source sft \
  --data data/conversations.jsonl \
  --out embeddings.pt \
  --batch_size 32
```

Then the `embeddings.pt` can be used for:

- Retrieval augmented generation (RAG)
- Clustering analysis
- Nearest neighbor search
- 3D visualization (with kat_viz_embeddings.py)
- Inverse-density sampling (with kat_inv_density_sample.py)

---

## System 2: DensityAwareSampler in kat_train_grpo.py (Runtime Sampling)

### Purpose

During GRPO training, sample preference pairs **inversely proportional to their density** in prompt space. Rare prompts get higher sampling weight.

### How It Works

```
Prompts List
    ↓
Compute Real-Time Embeddings
(using base model's logits)
    ↓
k-NN Density Computation
(mean distance to k neighbors)
    ↓
Inverse Density Weighting
(weight = 1 / density)
    ↓
WeightedRandomSampler
    ↓
Biased Training Data
(rare prompts sampled more)
```

### Step-by-Step Process

```python
# 1. Initialize sampler
sampler = DensityAwareSampler(
    prompts=["Explain X", "What is Y", ...],
    base_model=base_model,
    tokenizer=tokenizer,
    device=device,
    k=10  # Use 10 nearest neighbors
)

# 2. Compute embeddings from base model
def _compute_embeddings_from_model(prompts):
    embeddings = []
    for batch in prompts:
        # Encode
        tokens = [tokenizer.encode(p) for p in batch]
        x = torch.tensor(tokens_padded)

        # Forward pass (standard - just logits)
        logits = base_model(x)  # [B, T, vocab_size]

        # Average pooling over sequence
        embeddings_batch = logits.mean(dim=1)  # [B, vocab_size]
        embeddings.append(embeddings_batch.cpu().numpy())

    # Normalize
    embeddings = np.concatenate(embeddings)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return embeddings

# 3. Compute local density using k-NN
def _compute_inverse_density_weights():
    # Build k-NN index
    nbrs = NearestNeighbors(n_neighbors=k+1)
    nbrs.fit(embeddings)

    # Find k+1 nearest neighbors (including self)
    distances, indices = nbrs.kneighbors(embeddings)

    # Compute density = mean distance to k nearest (excluding self)
    local_densities = np.mean(distances[:, 1:], axis=1)

    # Inverse weighting: rare (large distance) → high weight
    weights = 1.0 / local_densities
    weights = weights / weights.sum()  # Normalize to probabilities

    return weights

# 4. Use in DataLoader
weights = torch.from_numpy(sampler.density_weights).float()
sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

# 5. Training loop
for batch in dataloader:
    # Batch has more rare prompts, fewer common ones
    loss = compute_grpo_loss(batch)
    loss.backward()
```

### Key Difference from kat_embed.py

| Aspect            | kat_embed.py                                            | DensityAwareSampler               |
| ----------------- | ------------------------------------------------------- | --------------------------------- |
| **When**          | Before training (offline)                               | During training (online)          |
| **Hidden States** | Uses `return_hidden_states=True` (actual hidden states) | Uses logits then average pools    |
| **Purpose**       | Save embeddings for later use                           | Compute sampling weights          |
| **Model**         | Any checkpoint (sft, base, grpo)                        | Base model (requires pre-trained) |
| **Saved?**        | Yes, to disk                                            | No, computed on-the-fly           |
| **Dimension**     | n_embd (768)                                            | vocab_size (50304)                |

---

## System 3: kat_viz_embeddings.py (3D Visualization)

### Purpose

Create a 3D visualization of the prompt space showing which prompts are rare vs common.

### How It Works

```
Pairs JSONL
    ↓
Extract Prompts
    ↓
Compute Embeddings (sentence-transformers)
    ↓
Project to 3D (UMAP)
    ↓
Compute Local Density
    ↓
Create JSON Visualization
```

### Process

```python
# 1. Load pairs and extract prompts
pairs = load_pairs("pairs_all.jsonl")
prompts = [p.get('prompt', '') for p in pairs]

# 2. Compute embeddings using sentence-transformers (384-dim)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(prompts, batch_size=32)  # [N, 384]

# 3. Project to 3D using UMAP (cluster-preserving)
import umap
reducer = umap.UMAP(n_components=3, metric='cosine')
coords_3d = reducer.fit_transform(embeddings)  # [N, 3]
coords_normalized = 2 * (coords_3d - min) / (max - min) - 1  # [-1, 1]

# 4. Compute local density for point sizing
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=10).fit(embeddings)
distances, indices = nbrs.kneighbors(embeddings)
local_density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-6)
local_density = (local_density - min) / (max - min)  # [0, 1]

# 5. Create visualization
viz_data = {
    'points': [
        {
            'x': coords[i, 0],
            'y': coords[i, 1],
            'z': coords[i, 2],
            'size': 0.5 + (1.0 - density[i]) * 2.0,  # Rare = bigger
            'color': get_color_for_source(pairs[i]['src']),
            'prompt': pairs[i]['prompt'][:200],
        }
        for i in range(N)
    ]
}

# 6. Save JSON
json.dump(viz_data, "embeddings_3d.json")
```

### Output

Interactive 3D visualization where:

- **Each point** = one prompt
- **Color** = source dataset
- **Size** = inverse of density (rare prompts are bigger)
- **Position** = semantic similarity (nearby prompts are similar)

---

## System 4: kat_inv_density_sample.py (Offline Inverse Density Sampling)

### Purpose

Given pre-computed embeddings, select a subset of data inversely proportional to density (rare samples get higher priority).

### How It Works

```
Embeddings File (from kat_embed.py)
    ↓
k-NN Density Search (FAISS)
    ↓
Compute Weights
    ↓
Probabilistic Sampling
    ↓
Output Selected IDs
```

### Process

```python
# 1. Load embeddings
blob = torch.load("embeddings.pt", map_location="cpu")
ids = blob["ids"]          # Original IDs
X = blob["emb"].numpy()    # [N, 768], L2-normalized

# 2. Build FAISS index (fast k-NN)
index = faiss.IndexFlatIP(X.shape[1])  # Inner product for cosine
index.add(X)

# 3. Find k+1 nearest neighbors (self + k others)
D, I = index.search(X, k=20+1)  # Distance/indices

# 4. Compute density from k-NN distances
similarity = D[:, 1:].mean(axis=1)  # Mean similarity to k neighbors (excluding self)
density = np.maximum(0.0, similarity)  # Higher sim = higher density

# 5. Inverse weighting
eps = 1e-3
weights = 1.0 / (eps + density)
weights = weights / weights.sum()  # Normalize to probabilities

# 6. Sample without replacement
chosen = np.random.choice(len(ids), size=budget, replace=False, p=weights)

# 7. Output selected IDs
with open("sampled_ids.txt", "w") as f:
    for i in chosen:
        f.write(str(ids[i]) + "\n")
```

---

## Complete Workflow Example

### Scenario: Train GRPO with Density-Aware Sampling

```bash
# Step 1: Download preference pairs
python -m scripts.kat_download_pairs
# Output: ~/.cache/nanochat/data/pairs_all.jsonl (100k+ pairs)

# Step 2: Extract unique prompts
python -m scripts.kat_make_prompts
# Output: ~/.cache/nanochat/data/prompts_all.jsonl

# Step 3: Train Reward Model
python -m scripts.kat_train_rm --max_steps=5000
# Output: ~/.cache/nanochat/rm_checkpoints/d20/model_000000.pt

# Step 4: Train GRPO with density-aware sampling
# (DensityAwareSampler runs automatically)
python -m scripts.kat_train_grpo --density_aware=True --max_steps=5000
# Inside: Loads prompts → Computes embeddings → Samples inversely proportional to density

# Step 5: Visualize prompt space
python -m scripts.kat_viz_embeddings \
  --pairs_path ~/.cache/nanochat/data/pairs_all.jsonl \
  --output_path embeddings_3d.json
# Output: Interactive 3D visualization

# Optional: Extract embeddings for retrieval
python -m scripts.kat_embed \
  --ckpt_source sft \
  --data prompts_all.jsonl \
  --out embeddings.pt

# Optional: Sample using pre-computed embeddings
python scripts/kat_inv_density_sample.py \
  --emb embeddings.pt \
  --budget 10000 \
  --out_indices sampled_ids.txt
```

---

## Dimension Comparison

| System                        | Embedding Dim | Source                | Method                          |
| ----------------------------- | ------------- | --------------------- | ------------------------------- |
| **kat_embed.py**              | 768           | Model hidden states   | Pool + normalize                |
| **DensityAwareSampler**       | 50,304        | Model logits          | Average pool logits + normalize |
| **kat_viz_embeddings.py**     | 384           | sentence-transformers | Pre-trained SBERT               |
| **kat_inv_density_sample.py** | 768+          | Stored .pt file       | Pre-computed (flexible)         |

---

## Why Multiple Systems?

1. **kat_embed.py** = General-purpose offline embedding extraction

   - Can be reused for retrieval, analysis, clustering
   - Saves to disk for repeated use
   - Uses actual hidden states (better quality)

2. **DensityAwareSampler** = Online runtime sampling

   - Computes density on-the-fly during training
   - Doesn't require pre-computation or storage
   - Uses logits (simpler, lighter weight)
   - Core mechanism for testing hypothesis

3. **kat_viz_embeddings.py** = Visualization & analysis

   - Uses sentence-transformers (specialized for semantic similarity)
   - UMAP preserves cluster structure
   - Web-friendly 3D visualization

4. **kat_inv_density_sample.py** = Offline sampling pipeline
   - For pre-filtering/pre-processing data
   - Uses pre-computed embeddings
   - FAISS for fast k-NN at scale

---

## Key Insight: Density-Aware Sampling Hypothesis

The core experimental hypothesis:

> **Sampling preference pairs inversely proportional to prompt density increases model diversity and reduces mode collapse.**

This is tested in GRPO training:

```
Control:    Uniform sampling from all pairs
            → All prompts equally likely
            → Model might collapse to common prompt distributions

Experiment: Density-aware sampling
            → Rare prompts sampled more frequently
            → Model trained on more diverse prompt distributions
            → Expected: Higher diversity in outputs
            → Measured: Lower em-dash frequency, higher vocabulary ratio
```

---

## Summary Table

| Script                        | Input                      | Output             | Usage                         |
| ----------------------------- | -------------------------- | ------------------ | ----------------------------- |
| **kat_embed.py**              | JSONL conversations        | embeddings.pt      | Analysis, retrieval, viz      |
| **kat_train_grpo.py**         | (uses DensityAwareSampler) | GRPO checkpoint    | Core RLHF training            |
| **DensityAwareSampler**       | Prompts + base model       | Sampling weights   | Used inside kat_train_grpo.py |
| **kat_viz_embeddings.py**     | pairs_all.jsonl            | embeddings_3d.json | 3D web visualization          |
| **kat_inv_density_sample.py** | embeddings.pt              | sampled_ids.txt    | Data pre-filtering            |

---

## Does Embedding from kat_embed.py Get Used for Sampling?

**Short answer: Not directly.**

- **kat_embed.py** creates offline embeddings for general use
- **DensityAwareSampler** computes its own embeddings on-the-fly during training
- They use different dimensions (768 vs 50,304) and extraction methods

However, **kat_embed.py outputs could be used**:

1. To pre-compute density weights offline
2. For retrieval-augmented training
3. For analysis/visualization

But the actual GRPO training uses **DensityAwareSampler's own embeddings** for sampling weights.
