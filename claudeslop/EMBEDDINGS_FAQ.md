# Embeddings FAQ: Answers to Your Questions

---

## Q1: How does embedding work?

### General Concept

Embedding = converting text/data into a numerical vector that captures semantic meaning.

**In this pipeline**:

```
Text Input
    ↓
Tokenization (convert words to token IDs)
    ↓
Model Forward Pass (compute hidden states or logits)
    ↓
Pooling (reduce sequence [T, D] → [D])
    ↓
Normalization (L2 norm → unit vectors)
    ↓
Embedding Vector
```

### Example

```python
# Input
prompt = "What is machine learning?"

# After tokenization
tokens = [123, 456, 789, ...]  # Token IDs

# After forward pass
hidden_states = shape [seq_len=5, hidden_dim=768]

# After average pooling
pooled = shape [768]

# After L2 normalization
embedding = normalized vector of shape [768]

# Now you can:
# 1. Compare with other embeddings (cosine similarity)
# 2. Find similar prompts (nearest neighbors)
# 3. Compute local density
# 4. Use for clustering, retrieval, etc.
```

---

## Q2: Does kat_embed.py pair the dataset with embeddings?

### Short Answer

**Yes, it pairs each row ID with its embedding.**

### What It Does

```python
# Input: JSONL file with conversations
{
    "id": "abc123",
    "prompt": "What is ML?",
    "assistant": "Machine learning is..."
}

# Output: .pt file with paired embeddings
{
    "ids": ["abc123", "def456", ...],
    "emb": torch.tensor([
        [0.1, 0.2, 0.3, ...],  # Embedding for abc123
        [0.4, 0.5, 0.6, ...],  # Embedding for def456
        ...
    ])  # Shape: [N, 768]
}
```

### Why This Pairing?

So you can later:

```python
# Load embeddings
blob = torch.load("embeddings.pt")
ids = blob["ids"]
embeddings = blob["emb"]

# Find most similar prompts to a query
query_embedding = compute_embedding("Give me advice")
similarities = cosine_similarity(query_embedding, embeddings)
top_k_indices = similarities.argsort()[-5:]
related_ids = [ids[i] for i in top_k_indices]
```

---

## Q3: Does that embedding get used for sampling?

### Short Answer

**No - not directly.** But the **concept** does.

### Why Not Directly?

**kat_embed.py embeddings:**

- Extracted **offline** (before training)
- Dimension: 768
- Source: model hidden states
- File: embeddings.pt (on disk)

**DensityAwareSampler embeddings:**

- Computed **online** during training (in kat_train_grpo.py)
- Dimension: 50,304 (vocab size)
- Source: model logits
- Not saved (computed on-the-fly)

### So What Gets Used for Sampling?

**DensityAwareSampler** (in kat_train_grpo.py) does the sampling:

```python
# During GRPO training initialization:

# 1. Load prompts from disk
prompts = load_prompts("prompts_all.jsonl")

# 2. Initialize DensityAwareSampler
sampler = DensityAwareSampler(
    prompts=prompts,
    base_model=base_model,  # Pre-trained base model
    tokenizer=tokenizer,
    device=device,
    k=10  # k-NN parameter
)

# Inside DensityAwareSampler:
#   - Computes embeddings from base_model
#   - Computes k-NN distances
#   - Computes inverse density weights
#   - Creates WeightedRandomSampler

# 3. Use sampler in DataLoader
dataloader = DataLoader(
    dataset,
    sampler=sampler,  # Uses computed weights
    batch_size=32
)

# Now training gets more rare prompts:
for batch in dataloader:
    # Batch biased towards rare prompts!
    loss = train_step(batch)
```

---

## Q4: What are the relationships between these embedding systems?

### They're Actually Separate Pipelines

```
┌─────────────────────────────────────────────┐
│           kat_embed.py                      │
│  (Offline: Extract & Save Embeddings)      │
│  Input: conversations.jsonl                │
│  Output: embeddings.pt [N, 768]            │
│  Uses: model hidden_states                 │
└─────────────────────────────────────────────┘
       ↓                              ↓
   For Analysis              For Sampling (Optional)
   - Retrieval           kat_inv_density_sample.py
   - Clustering              (Offline: Use embeddings
   - Visualization           to compute sample weights)

┌─────────────────────────────────────────────┐
│      kat_train_grpo.py                      │
│  (Online: Real-time Density Computation)   │
│  Has: DensityAwareSampler                  │
│  Computes: embeddings on-the-fly           │
│  Uses: model logits                        │
│  Independent: Doesn't use kat_embed.py     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│     kat_viz_embeddings.py                   │
│  (Visualization: 3D Prompt Space)          │
│  Input: pairs_all.jsonl                    │
│  Uses: sentence-transformers               │
│  Independent: Separate embedding model     │
└─────────────────────────────────────────────┘
```

### Connection: They All Measure Density

**What They Share:**

- Same concept: k-NN distance → density
- Same math: weight = 1 / density
- Different implementations and purposes

**What They Don't Share:**

- kat_embed.py outputs are NOT automatically used by kat_train_grpo.py
- Each computes its own embeddings
- Can use different models/dimensions

---

## Q5: Does kat_embed.py get used for anything?

### Yes, multiple things:

1. **Analysis & Clustering**

   ```bash
   python scripts/kat_embed.py \
     --ckpt_source sft \
     --data conversations.jsonl \
     --out embeddings.pt

   # Now you can cluster/analyze with embeddings.pt
   ```

2. **Retrieval-Augmented Generation (RAG)**

   ```python
   # Load embeddings
   embeddings = torch.load("embeddings.pt")

   # For each query, find similar prompts
   query = "What is NLP?"
   query_emb = compute_embedding(query)
   similarities = (embeddings["emb"] @ query_emb).topk(5)
   ```

3. **Offline Inverse-Density Sampling**

   ```bash
   python scripts/kat_inv_density_sample.py \
     --emb embeddings.pt \
     --budget 10000 \
     --out_indices sampled_ids.txt
   ```

4. **3D Visualization Input** (optional)
   ```bash
   # kat_viz_embeddings can also use pre-computed embeddings
   # Currently it computes its own with sentence-transformers
   # But could be modified to use kat_embed.py output
   ```

### But NOT Required for GRPO Training

kat_train_grpo.py **does not require** kat_embed.py output.

---

## Q6: So what's the real pipeline for training?

### Minimal Pipeline (Required)

```
1. Download pairs
   python -m scripts.kat_download_pairs
   → pairs_all.jsonl

2. Extract prompts
   python -m scripts.kat_make_prompts
   → prompts_all.jsonl

3. Train Reward Model
   python -m scripts.kat_train_rm
   → rm_checkpoints/model_000000.pt

4. Train GRPO (with built-in DensityAwareSampler)
   python -m scripts.kat_train_grpo
   → grpo_checkpoints/model_000000.pt

   # Inside: DensityAwareSampler automatically:
   # - Loads prompts
   # - Computes embeddings from base model
   # - Computes density weights
   # - Samples inversely proportional to density
```

### Optional: Analysis & Visualization

```
5. Extract embeddings for later use (optional)
   python -m scripts.kat_embed
   → embeddings.pt

6. Visualize prompt space (optional)
   python -m scripts.kat_viz_embeddings
   → embeddings_3d.json
```

---

## Q7: What about the `return_hidden_states=True` parameter?

### What It Does

```python
# Before (only returns logits):
logits = model(x)
# logits shape: [batch, seq_len, vocab_size]

# After (with new parameter):
output = model(x, return_hidden_states=True)
# Returns dict:
# {
#     'logits': [batch, seq_len, vocab_size],
#     'hidden_states': [batch, seq_len, n_embd]
# }
```

### Who Uses It?

**kat_embed.py** uses it to get high-quality embeddings:

```python
output = model(x, return_hidden_states=True)
hidden_states = output['hidden_states']  # [B, T, 768]
# Average pool → normalize → save
```

**DensityAwareSampler** doesn't use it (uses logits instead):

```python
# Just get logits (standard call)
logits = base_model(x)  # [B, T, 50304]
# Average pool → normalize
```

---

## Quick Comparison Table

| Question                                         | Answer                                                                                 |
| ------------------------------------------------ | -------------------------------------------------------------------------------------- |
| **Does kat_embed.py pair data with embeddings?** | ✅ Yes - outputs {"ids": [...], "emb": tensor}                                         |
| **Does that embedding get used for sampling?**   | ❌ No - sampling uses its own embeddings                                               |
| **What uses kat_embed.py output?**               | Analysis, retrieval, optional sampling, visualization                                  |
| **Does sampling require kat_embed.py?**          | ❌ No - DensityAwareSampler is self-contained                                          |
| **Are they related?**                            | 🤔 Same concept (density-aware), different implementations                             |
| **How many embedding systems?**                  | 4: kat_embed.py, DensityAwareSampler, kat_viz_embeddings.py, kat_inv_density_sample.py |
| **Do they interfere?**                           | ❌ No - completely independent                                                         |
| **Which is used for training?**                  | ✅ DensityAwareSampler (built into kat_train_grpo.py)                                  |

---

## The Key Insight

```
┌────────────────────────────────────────────────────────────┐
│  Density-Aware Sampling = Core Hypothesis                  │
│                                                            │
│  "Training on rare prompts more often reduces mode collapse"│
│                                                            │
│  Implementation:                                           │
│  1. Compute prompt embeddings                              │
│  2. Find k-NN neighbors for each prompt                    │
│  3. Compute local density (from k-NN distances)            │
│  4. Weight = 1 / density (rare = high weight)              │
│  5. Sample using these weights                             │
│  6. Train model on biased dataset                          │
│                                                            │
│  Test:                                                     │
│  Compare outputs with/without density-aware sampling       │
└────────────────────────────────────────────────────────────┘

kat_embed.py                   vs    DensityAwareSampler
├─ Offline tool                       ├─ Online mechanism
├─ Saves embeddings                   ├─ Computes on-the-fly
├─ For analysis/retrieval            ├─ For training biasing
└─ Optional                           └─ Core to experiment
```

---

## Bottom Line

1. ✅ **kat_embed.py** extracts and pairs embeddings (useful tool)
2. ✅ **DensityAwareSampler** uses embeddings for sampling (core mechanism)
3. ❌ They're **not directly connected** - each does its own thing
4. ✅ Both follow same density-aware principle
5. ✅ Pipeline works **without kat_embed.py** (it's optional)
