# How to Use the Pairwise Preferences Data Pipeline

This guide provides exact commands to download, process, and evaluate the pairwise preferences data for RLHF training with density-aware sampling.

## Quick Start (TL;DR)

If you want to run everything end-to-end:

```bash
# 1. Download all datasets
python -m scripts.kat_download_pairs

# 2. Extract and deduplicate prompts
python -m scripts.kat_make_prompts

# 3. Evaluate the dataset
python -m scripts.kat_eval_pairs --output-report .cache/data/eval_report.txt
```

---

## Step-by-Step Guide

### Prerequisites

Make sure you're in the repo root and your Python environment is activated:

```bash
cd /path/to/nanochat-rlhf-density-sampling
source .venv/bin/activate  # or your venv activation command
```

Verify the scripts/ directory exists:

```bash
ls scripts/ | grep kat_
```

You should see output like:

```
kat_download_pairs.py
kat_eval_pairs.py
kat_make_prompts.py
```

---

## Stage 1: Download Pairwise Preferences Data

### 1.1 Download All Datasets (Default)

This downloads from three sources: HH-RLHF, UltraFeedback, and Stack Exchange.

```bash
python -m scripts.kat_download_pairs
```

**Expected output:**

```
Wrote .cache/data/pairs_all.jsonl with X pairs
```

Where X is typically around 200,000-280,000 pairs depending on dataset availability.

**Time estimate**: 20-40 minutes (depends on network and dataset sizes)

**Output file**: `.cache/data/pairs_all.jsonl` (~500 MB)

---

### 1.2 Download Only One Dataset (Optional)

If you want to test or avoid downloading everything:

**Only HH-RLHF:**

```bash
python -m scripts.kat_download_pairs --only hh
```

**Only UltraFeedback:**

```bash
python -m scripts.kat_download_pairs --only uf
```

**Only Stack Exchange:**

```bash
python -m scripts.kat_download_pairs --only se
```

---

### 1.3 Download Selectively (Optional)

Skip specific datasets if they're causing issues:

**Skip HH-RLHF:**

```bash
python -m scripts.kat_download_pairs --no-hh
```

**Skip UltraFeedback:**

```bash
python -m scripts.kat_download_pairs --no-uf
```

**Skip Stack Exchange:**

```bash
python -m scripts.kat_download_pairs --no-se
```

**Example: Download only HH and Stack Exchange:**

```bash
python -m scripts.kat_download_pairs --no-uf
```

---

## Stage 2: Extract and Deduplicate Prompts

### 2.1 Standard Processing

After stage 1 completes successfully, extract unique prompts:

```bash
python -m scripts.kat_make_prompts
```

**Expected output:**

```
Wrote .cache/data/prompts_all.jsonl, .cache/data/prompt_id_map.tsv, and .cache/data/stats.txt
```

**Output files**:

- `.cache/data/prompts_all.jsonl`: Deduplicated prompts (~50 MB)
- `.cache/data/prompt_id_map.tsv`: TSV mapping of prompt IDs to text
- `.cache/data/stats.txt`: Statistics breakdown

**Time estimate**: 2-5 minutes

---

### 2.2 Inspect the Results

View basic statistics:

```bash
cat .cache/data/stats.txt
```

Example output:

```
total_pairs        276543
unique_prompts     89234
src_hh-rlhf        160000
src_ultrafeedback-binarized  64000
src_stack-exchange-preferences  52543
```

Look at a few prompts:

```bash
head -5 .cache/data/prompts_all.jsonl
```

---

## Stage 3: Evaluate the Dataset

### 3.1 Print Evaluation Report to Console

```bash
python -m scripts.kat_eval_pairs
```

This will print a formatted report with:

- Dataset size and composition
- Source distribution
- Response length statistics
- Quality metrics

---

### 3.2 Save Evaluation Report to File

```bash
python -m scripts.kat_eval_pairs --output-report .cache/data/eval_report.txt
```

Then view it:

```bash
cat .cache/data/eval_report.txt
```

---

### 3.3 Evaluate Custom Input Files

If you processed data with custom paths:

```bash
python -m scripts.kat_eval_pairs \
  --pairs-file /path/to/pairs.jsonl \
  --prompts-file /path/to/prompts.jsonl \
  --output-report /path/to/report.txt
```

---

## Verification Checklist

After running all stages, verify everything succeeded:

```bash
# Check that all output files exist
ls -lh .cache/data/

# Expected files:
# pairs_all.jsonl          (should be ~500 MB)
# prompts_all.jsonl        (should be ~50 MB)
# prompt_id_map.tsv        (should be ~50 MB)
# stats.txt                (should be a few KB)
# eval_report.txt          (if you ran stage 3.2)
```

Count the number of pairs and prompts:

```bash
echo "Pairs:" && wc -l .cache/data/pairs_all.jsonl && \
echo "Prompts:" && wc -l .cache/data/prompts_all.jsonl
```

---

## Full Pipeline: One Command

If you want to run everything in sequence without stopping:

```bash
echo "=== Stage 1: Download ===" && \
python -m scripts.kat_download_pairs && \
echo "" && \
echo "=== Stage 2: Extract & Deduplicate ===" && \
python -m scripts.kat_make_prompts && \
echo "" && \
echo "=== Stage 3: Evaluate ===" && \
python -m scripts.kat_eval_pairs --output-report .cache/data/eval_report.txt && \
echo "" && \
echo "=== All Done! ===" && \
ls -lh .cache/data/
```

**Total time**: ~1-2 hours (mostly network I/O for downloads)

---

## What to Do With the Data

Once you have the processed data, the next steps for density-aware RLHF would be:

### Option 1: Quick Integration (Not Yet Implemented)

- Use `prompts_all.jsonl` directly in existing RLHF training
- Sampling would be uniform (no density weighting yet)

### Option 2: Density Sampling (Future Work)

- Compute embeddings for each prompt using a sentence encoder
- Build a k-NN graph to estimate local density
- Weight samples inversely by density
- Integrate into the RLHF training loop

### Option 3: Exploratory Analysis

- Use `eval_report.txt` to understand data characteristics
- Analyze response quality by source dataset
- Identify quality issues or biases

---

## Troubleshooting

### Problem: "No such file or directory" for pairs_all.jsonl

**Solution**: Stage 1 didn't complete or failed. Check:

```bash
python -m scripts.kat_download_pairs --only hh
```

Try downloading just one dataset first to debug network issues.

### Problem: Out of Memory

**Solution**: Datasets are large. Try:

1. Download one dataset at a time (not all three)
2. Run on a machine with more RAM
3. Process smaller batches

### Problem: Very Few Unique Prompts

**Solution**: This is actually expected! Many datasets have overlapping questions. This validates that deduplication is important. Example realistic output:

- 276k pairs → 89k unique prompts (68% duplicate)

### Problem: Dataset Not Found

**Solution**: HuggingFace datasets sometimes require authentication or may be temporarily unavailable.

- Try skipping that dataset: `python -m scripts.kat_download_pairs --no-uf`
- Try again later
- Check your HuggingFace token if needed: `huggingface-cli login`

### Problem: Script Import Errors

**Solution**: Make sure you're running from repo root and environment is activated:

```bash
cd /path/to/nanochat-rlhf-density-sampling
source .venv/bin/activate
```

---

## Output File Formats

### pairs_all.jsonl

Each line is a JSON object:

```json
{
  "id": "a1b2c3d4e5f6g7h8",
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "I'm not sure, maybe Lyon?",
  "src": "hh-rlhf"
}
```

### prompts_all.jsonl

Each line is a JSON object:

```json
{
  "id": "a1b2c3d4e5f6g7h8",
  "prompt": "What is the capital of France?"
}
```

### prompt_id_map.tsv

Tab-separated values:

```
a1b2c3d4e5f6g7h8	What is the capital of France?
b2c3d4e5f6g7h8i9	How do photosynthesis work?
...
```

### stats.txt

Key-value pairs:

```
total_pairs	276543
unique_prompts	89234
src_hh-rlhf	160000
src_ultrafeedback-binarized	64000
src_stack-exchange-preferences	52543
```

---

## Performance Notes

Approximate performance on a typical machine:

| Stage          | Time          | Notes                                      |
| -------------- | ------------- | ------------------------------------------ |
| Download (all) | 20-40 min     | Network dependent, large dataset downloads |
| Download (one) | 5-10 min      | Much faster if only doing one dataset      |
| Deduplicate    | 2-5 min       | Processes ~276k items, mostly I/O          |
| Evaluate       | <1 min        | Just reads files and computes stats        |
| **Total**      | **1-2 hours** | Mostly waiting for downloads               |

---

## Next Steps

After completing this pipeline:

1. **Understand your data**: Review `eval_report.txt` to understand distribution, quality, and characteristics

2. **Plan density sampling**: Design how to compute embeddings and estimate density

3. **Integrate with training**: Modify RLHF training loop to sample proportionally to `1/density`

4. **Run experiments**: Train with and without density sampling to compare diversity metrics

5. **Compare results**: Analyze if density sampling reduces mode collapse (em-dashes, repetitive patterns, etc.)

Good luck! 🚀

---

**Last updated**: October 2025
