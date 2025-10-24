# RM Training Data Format Guide

## Overview

The Reward Model (RM) is trained on **preference pairs** - human feedback indicating which response is better.

**Data Flow:**

```
3 Datasets (HH-RLHF, UltraFeedback, Stack Exchange)
    ↓
kat_download_pairs.py (normalizes to common format)
    ↓
pairs_all.jsonl (unified format)
    ↓
kat_train_rm.py (PreferenceDataset)
    ↓
RM training
```

---

## Unified Format

All 3 datasets are converted to the **exact same format** in `pairs_all.jsonl`:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "prompt": "What is machine learning?",
  "chosen": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
  "rejected": "I don't know.",
  "src": "hh-rlhf"
}
```

### Fields Explained

| Field        | Type        | Source                                 | Purpose                                      |
| ------------ | ----------- | -------------------------------------- | -------------------------------------------- |
| **id**       | UUID string | Generated `uuid.uuid4()`               | Unique identifier for tracking               |
| **prompt**   | string      | Original dataset question/prompt       | Input to RM - what the model is asked        |
| **chosen**   | string      | Original dataset preferred response    | Target label: RM should give this HIGH score |
| **rejected** | string      | Original dataset dispreferred response | Target label: RM should give this LOW score  |
| **src**      | string      | Dataset name                           | Source tracking (useful for analysis)        |

---

## How Each Dataset Maps to Format

### 1. **Anthropic/hh-rlhf**

**Original Structure:**

- Conversation format with turns
- `chosen` = full conversation with preferred path
- `rejected` = full conversation with dispreferred path

**Extraction (lines 34-58):**

```python
def from_hh():
    ds = load_dataset("Anthropic/hh-rlhf")

    for r in ds[split]:
        # Extract first turn of conversation
        p1, a1 = extract_first_pair(r["chosen"])     # Get first Q&A from chosen path
        p2, a2 = extract_first_pair(r["rejected"])   # Get first Q&A from rejected path

        # Only keep if same prompt in both paths
        if p1 and a1 and p2 and a2 and norm_space(p1) == norm_space(p2):
            yield {
                "id": str(uuid.uuid4()),
                "prompt": norm_space(p1),      # First human question
                "chosen": norm_space(a1),      # First assistant response (preferred)
                "rejected": norm_space(a2),    # First assistant response (not preferred)
                "src": "hh-rlhf",
            }
```

**Example HH-RLHF Pair:**

```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "prompt": "How do I make homemade pasta?",
  "chosen": "To make homemade pasta, you'll need flour, eggs, and salt. Mix the flour with eggs to form a dough, knead it until smooth, then roll and cut into your desired shape. Let it dry slightly before cooking.",
  "rejected": "I'm not sure about pasta-making.",
  "src": "hh-rlhf"
}
```

---

### 2. **HuggingFaceH4/ultrafeedback_binarized**

**Original Structure:**

- Already has `prompt`, `chosen`, `rejected` fields
- Binarized: each entry has 2 responses (best vs. worst)

**Extraction (lines 60-73):**

```python
def from_ultrafeedback_binarized():
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

    for r in ds["train"]:
        p = r.get("prompt")
        c = r.get("chosen")
        rej = r.get("rejected")

        # Direct mapping - data already in correct format!
        if p and c and rej:
            yield {
                "id": str(uuid.uuid4()),
                "prompt": norm_space(p),
                "chosen": norm_space(c),
                "rejected": norm_space(rej),
                "src": "ultrafeedback-binarized",
            }
```

**Example UltraFeedback Pair:**

```json
{
  "id": "234f5678-f90c-23e4-b567-527625285111",
  "prompt": "Explain photosynthesis in simple terms",
  "chosen": "Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to create oxygen and glucose. The plant stores the glucose for energy and releases oxygen as a byproduct.",
  "rejected": "Plants make food using light.",
  "src": "ultrafeedback-binarized"
}
```

---

### 3. **HuggingFaceH4/stack-exchange-preferences**

**Original Structure:**

- Q&A format with voting/scoring
- Question in `question.body` with HTML tags
- Two answers: `winner`/`answer_0` (preferred) vs `loser`/`answer_1` (not preferred)

**Extraction (lines 81-94):**

```python
def from_stack_exchange_prefs():
    ds = load_dataset("HuggingFaceH4/stack-exchange-preferences")

    for r in ds["train"]:
        q = (r.get("question") or {}).get("body")  # Question text (has HTML)
        a_win = r.get("winner") or r.get("chosen") or r.get("answer_0")
        a_lose = r.get("loser") or r.get("rejected") or r.get("answer_1")

        if q and a_win and a_lose:
            yield {
                "id": str(uuid.uuid4()),
                "prompt": strip_html(q),           # Remove <br> and HTML tags
                "chosen": norm_space(a_win),       # Upvoted answer
                "rejected": norm_space(a_lose),    # Less upvoted answer
                "src": "stack-exchange-preferences",
            }
```

**Example Stack Exchange Pair:**

```json
{
  "id": "345a6789-a01d-34f5-c678-638736396222",
  "prompt": "What is the difference between var and let in JavaScript?",
  "chosen": "var is function-scoped and can be redeclared. let is block-scoped and cannot be redeclared within the same scope. Use let for most cases in modern JavaScript.",
  "rejected": "They're pretty much the same, just different syntax.",
  "src": "stack-exchange-preferences"
}
```

---

## Data Quality Checks

### Normalization (Applied to All)

```python
def norm_space(s: str) -> str:
    # Normalize whitespace for consistency and deduplication
    return re.sub(r"\s+", " ", s.strip())
```

**Examples:**

```
"Hello\n\nworld"          → "Hello world"
"Question  with\ttabs"    → "Question with tabs"
"  Leading spaces  "      → "Leading spaces"
```

### HTML Cleaning (Stack Exchange Only)

```python
def strip_html(s: str) -> str:
    s = re.sub(r"<\s*br\s*/?>", "\n", s, flags=re.I)  # Convert <br> to newline
    s = re.sub(r"<[^>]+>", " ", s)                     # Remove other HTML tags
    return norm_space(s)
```

**Example:**

```
Input:  "How to <b>optimize</b> code?<br/>Best practices:"
Output: "How to optimize code? Best practices:"
```

### Validation Checks

1. **Same Prompt Required (HH-RLHF only)**

   ```python
   if norm_space(p1) == norm_space(p2):  # Must have same prompt in both paths
       yield pair
   ```

2. **Non-empty Fields Required (All)**

   ```python
   if p and c and rej:  # All three must be present
       yield pair
   ```

3. **Count Validation**
   ```python
   if cnt == 0:
       raise RuntimeError("No pairs downloaded!")
   if cnt < 1000:
       print("Warning: Only %d pairs (expected >10,000)" % cnt)
   ```

---

## Actual File Statistics

When downloaded, `pairs_all.jsonl` contains:

| Dataset                    | Approx Count | Distribution |
| -------------------------- | ------------ | ------------ |
| hh-rlhf                    | ~10,000      | ~35%         |
| ultrafeedback-binarized    | ~14,000      | ~50%         |
| stack-exchange-preferences | ~4,000       | ~15%         |
| **TOTAL**                  | **~28,000**  | **100%**     |

**File Format:**

- One JSON object per line (JSONL)
- No header row
- Newline-delimited for streaming
- UTF-8 encoding with `ensure_ascii=False` (preserves unicode)

---

## How PreferenceDataset Uses This Format

In `kat_train_rm.py`, the `PreferenceDataset` class loads this data:

```python
class PreferenceDataset(Dataset):
    def __init__(self, pairs_path, tokenizer, max_length=512):
        self.pairs = []

        # Load JSONL file
        with open(pairs_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    pair = json.loads(line)  # Parse each JSON line

                    # Validate required fields
                    if all(k in pair for k in ['prompt', 'chosen', 'rejected']):
                        self.pairs.append(pair)
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
```

### Usage in Training

```python
def __getitem__(self, idx):
    pair = self.pairs[idx]

    # Extract fields
    prompt = pair['prompt']
    chosen = pair['chosen']
    rejected = pair['rejected']

    # Combine prompt + response
    chosen_text = f"{prompt}\n{chosen}"
    rejected_text = f"{prompt}\n{rejected}"

    # Tokenize and return
    chosen_ids = self.tokenizer.encode(chosen_text)
    rejected_ids = self.tokenizer.encode(rejected_text)

    return {
        'chosen_ids': torch.tensor(chosen_ids),
        'rejected_ids': torch.tensor(rejected_ids),
    }
```

---

## Verification: Example Data for Web Analysis

Here's representative sample data you can use for verification:

### Sample 1: HH-RLHF Style

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "prompt": "What are some good tips for learning a new language?",
  "chosen": "Here are some effective strategies: 1) Practice daily, even 15-30 minutes helps. 2) Immerse yourself in the language through movies, music, or podcasts. 3) Find a conversation partner to practice speaking. 4) Use spaced repetition for vocabulary. 5) Focus on high-frequency words first. 6) Don't be afraid to make mistakes. Consistency is more important than perfection.",
  "rejected": "Just try really hard and you'll eventually learn it.",
  "src": "hh-rlhf"
}
```

### Sample 2: UltraFeedback Style

```json
{
  "id": "661f9511-f30c-52e5-a827-557766551111",
  "prompt": "Explain how machine learning models are trained",
  "chosen": "Machine learning models are trained through an iterative process: 1) Prepare labeled training data. 2) Initialize model parameters randomly. 3) Forward pass: feed data through the model to get predictions. 4) Calculate loss: measure difference between predictions and actual labels. 5) Backward pass: compute gradients of loss with respect to parameters. 6) Update parameters using an optimizer (e.g., SGD, Adam). 7) Repeat steps 3-6 until convergence. The model learns patterns that minimize the loss function.",
  "rejected": "You put data in and the model learns it. That's basically it.",
  "src": "ultrafeedback-binarized"
}
```

### Sample 3: Stack Exchange Style

```json
{
  "id": "772g0622-g41d-63f6-b938-668877662222",
  "prompt": "How do I sort a list in Python without using the built-in sort function?",
  "chosen": "You can implement a sorting algorithm manually. Here's bubble sort: def bubble_sort(arr): for i in range(len(arr)): for j in range(len(arr)-1-i): if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j] return arr. Or use sorted() or .sort() which are built-in. For custom sorting, pass a key parameter: sorted(items, key=lambda x: x.age).",
  "rejected": "Python doesn't have a way to sort without built-in functions, so you're stuck with using .sort().",
  "src": "stack-exchange-preferences"
}
```

---

## Verification Checklist

✅ **All three datasets unified to same format:**

- [ ] Same fields: `id`, `prompt`, `chosen`, `rejected`, `src`
- [ ] `id` is UUID string
- [ ] `prompt` is question/instruction
- [ ] `chosen` is better/preferred response
- [ ] `rejected` is worse/non-preferred response
- [ ] `src` tracks original dataset

✅ **Data quality:**

- [ ] Whitespace normalized (no extra spaces/tabs/newlines)
- [ ] HTML removed (Stack Exchange)
- [ ] All fields non-empty
- [ ] Same prompt in both chosen/rejected (HH-RLHF)
- [ ] Valid JSON on each line

✅ **Format correctness for RM training:**

- [ ] Can parse with `json.loads(line)`
- [ ] Can extract `prompt`, `chosen`, `rejected` fields
- [ ] Can combine as `f"{prompt}\n{chosen}"`
- [ ] Can tokenize both variants
- [ ] Bradley-Terry loss can compute: log(sigmoid(r_chosen - r_rejected))

---

## Summary

| Aspect             | Details                                                      |
| ------------------ | ------------------------------------------------------------ |
| **Total Sources**  | 3 datasets unified                                           |
| **Output File**    | `~/.cache/nanochat/data/pairs_all.jsonl`                     |
| **Format**         | One JSON object per line                                     |
| **Fields**         | `id`, `prompt`, `chosen`, `rejected`, `src`                  |
| **Total Pairs**    | ~28,000 (can vary)                                           |
| **Quality Checks** | Normalized whitespace, HTML stripped, validated fields       |
| **Used By**        | `PreferenceDataset` in `kat_train_rm.py`                     |
| **RM Training**    | Bradley-Terry preference loss: score chosen > score rejected |

**✅ Format is correct and ready for RM training!**
