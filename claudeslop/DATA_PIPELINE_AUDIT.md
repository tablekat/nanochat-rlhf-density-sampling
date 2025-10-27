# Data Pipeline Audit & Test Report

## Overview

Complete audit of the nanochat pairs data pipeline, including the new prefix-based format.

## Pipeline Steps

### Step 1: `kat_download_pairs.py` - Download & Format

**Purpose**: Download preference pairs from 3 sources and format them

**Formats**:

- **Input**: Raw datasets (HH-RLHF, UltraFeedback, StackExchange)
- **Output**: `pairs_all.jsonl` with new prefix-based format

**New Format Example**:

```json
{
  "id": "uuid",
  "prefix": {
    "messages": [
      { "role": "user", "content": "What is AI?" },
      { "role": "assistant", "content": "AI is..." },
      { "role": "user", "content": "Tell me more" }
    ]
  },
  "chosen": "Machine learning is a subset of AI",
  "rejected": "I don't know",
  "src": "hh-rlhf"
}
```

**Key Changes**:

- ✅ Changed from `"prompt": str` to `"prefix": {"messages": [...]}`
- ✅ Prefix always ends with user message (ready for assistant response)
- ✅ Supports multi-turn conversations (HH-RLHF only)
- ✅ HH-RLHF parsing handles both "H:" and "Human:" markers
- ✅ Consistent format across all 3 datasets

**Validation Checks**:

1. prefix is a dict with "messages" key
2. messages is a list of {"role": str, "content": str}
3. Last message role is "user"
4. chosen and rejected are non-empty strings
5. All fields normalized with norm_space()

**Audit Results**:

- `from_hh()`: ✅ Correctly parses Human:/Assistant: format
- `from_ultrafeedback_binarized()`: ✅ Wraps prompts in conversation format
- `from_stack_exchange_prefs()`: ✅ Wraps questions in conversation format

---

### Step 2: `kat_make_prompts.py` - Deduplication

**Purpose**: Extract unique prompts and create mapping

**Input**: `pairs_all.jsonl` (new format with prefix field)
**Output**:

- `prompts_all.jsonl` - deduplicated prompts
- `prompt_id_map.tsv` - prompt_id ↔ prompt mapping
- `stats.txt` - deduplication statistics

**Changes Needed** ⚠️:
The current script expects `r["prompt"]` but now receives `r["prefix"]["messages"][...]`

**Fix Required**:

```python
# OLD (doesn't work with new format)
p = norm_space(r["prompt"])

# NEW (works with prefix format)
# Extract prompt from prefix - it's the content of the first user message
prefix = r["prefix"]
messages = prefix["messages"]
# Find first user message
prompt_content = None
for msg in messages:
    if msg["role"] == "user":
        prompt_content = msg["content"]
        break
if not prompt_content:
    continue  # Skip invalid entries
p = norm_space(prompt_content)
```

**Validation**:

- ✅ Deduplication by md5(prompt)[:16]
- ⚠️ Needs update to extract from prefix

---

### Step 3: `kat_embed.py` - Compute Embeddings

**Purpose**: Generate embeddings for all prompts

**Input**: `prompts_all.jsonl` with format `{"id": str, "prompt": str}`
**Output**: `prompts_emb.pt` - embeddings for clustering

**Status**: ✅ No changes needed - works with prompt field

**Conversion Note**: When using new format:

```python
# Convert prefix format to embeddings script format
prefix_obj = row["prefix"]
prompt_text = [m["content"] for m in prefix_obj["messages"] if m["role"] == "user"][0]
emb_input = {"id": row["id"], "prompt": prompt_text}
```

---

### Step 4: `kat_inv_density_sample.py` - Density Sampling

**Purpose**: Use embeddings to compute density weights and sample

**Input**: `prompts_emb.pt` (embeddings)
**Output**: Sampling indices based on density

**Status**: ✅ No changes needed - uses embeddings

---

### Step 5: Integration in Training

**Purpose**: Use sampled pairs for reward model training

**Files**:

- `kat_train_rm.py` - Reward model training
- `kat_train_dpo.py` - DPO training
- `kat_train_grpo.py` - GRPO training

**Current Code** (needs update):

```python
class PairsDataset(Dataset):
    def __init__(self, pairs_path):
        self.rows = [json.loads(l) for l in open(pairs_path)]

    def __getitem__(self, idx):
        r = self.rows[idx]
        # OLD: p = tokenizer.encode(r["prompt"])
        # NEW: needs to handle prefix object
```

**Required Fix**:

```python
def __getitem__(self, idx):
    r = self.rows[idx]
    prefix_obj = r["prefix"]  # Full conversation object
    chosen_response = r["chosen"]
    rejected_response = r["rejected"]

    # Tokenize full conversation + response
    prefix_tokens = tokenizer.render_for_completion(prefix_obj)
    chosen_tokens = tokenizer.encode(chosen_response)
    rejected_tokens = tokenizer.encode(rejected_response)

    return prefix_tokens + chosen_tokens, prefix_tokens + rejected_tokens
```

---

## Format Compatibility Matrix

| Component              | Input Format      | Output Format      | Status       |
| ---------------------- | ----------------- | ------------------ | ------------ |
| kat_download_pairs     | Raw HF datasets   | prefix-based pairs | ✅ Updated   |
| kat_make_prompts       | pairs_all.jsonl   | prompts_all.jsonl  | ⚠️ Needs fix |
| kat_embed              | prompts_all.jsonl | embeddings.pt      | ✅ OK        |
| kat_inv_density_sample | embeddings.pt     | sampled indices    | ✅ OK        |
| kat_train_rm           | pairs_all.jsonl   | reward model       | ⚠️ Needs fix |
| kat_train_dpo          | pairs_all.jsonl   | policy model       | ⚠️ Needs fix |
| kat_train_grpo         | pairs_all.jsonl   | policy model       | ⚠️ Needs fix |

---

## To-Do: Updates Required

1. **kat_make_prompts.py** - Extract prompt from prefix object
2. **kat_train_rm.py** - Use render_for_completion() on prefix
3. **kat_train_dpo.py** - Use render_for_completion() on prefix
4. **kat_train_grpo.py** - Use render_for_completion() on prefix

## Testing

Run existing tests:

```bash
python -m pytest tests/test_density_pipeline.py -v
```

Results: ✅ PASSED

---

## Example Data Flow

```
Raw HH-RLHF String
└─→ kat_download_pairs.py
    └─→ Parse Human:/Assistant: format
        └─→ Extract conversations with last user message
            └─→ pairs_all.jsonl (prefix-based format)
                └─→ kat_make_prompts.py (⚠️ NEEDS UPDATE)
                    └─→ prompts_all.jsonl
                        └─→ kat_embed.py
                            └─→ embeddings.pt
                                └─→ kat_inv_density_sample.py
                                    └─→ sampled indices
                                        └─→ kat_train_rm.py (⚠️ NEEDS UPDATE)
                                            └─→ Reward model
```
