# Claude's Additions: Pairwise Preferences Data Pipeline

This document describes the new data pipeline scripts added to support the density-sampling RLHF research in nanochat.

## Overview

The goal of this pipeline is to:

1. Download open pairwise preference datasets from HuggingFace
2. Consolidate them into a unified format
3. Extract and deduplicate prompts
4. Provide evaluation and analysis tools
5. Prepare data for density-aware sampling during RLHF training

The theory is that by sampling conversation pairs inversely proportional to prompt density (rather than uniformly), we can improve the diversity of the model's outputs and reduce mode collapse.

## New Scripts

All scripts follow the `kat_` naming convention (per your existing additions).

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
- Optional filtering by dataset (see Usage below)

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

# Custom output path
python -m scripts.kat_download_pairs --out /path/to/pairs.jsonl
```

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

**Why deduplication?**: The three datasets may have overlapping prompts. Deduplication helps:

- Avoid training bias from redundant prompts
- Prepare for density-based sampling (you only sample each unique prompt once)
- Reduce storage requirements

**Usage**:

```bash
# Process pairs to extract and deduplicate prompts
python -m scripts.kat_make_prompts
```

### 3. `scripts/kat_eval_pairs.py`

**Purpose**: Analyze and evaluate the quality and characteristics of the consolidated pairs dataset.

**Analysis includes**:

- **Basic statistics**: Total pairs, unique prompts, pair-to-prompt ratio
- **Source distribution**: Breakdown of pairs by source dataset
- **Length statistics**: Min, max, mean, median, p95 character counts for:
  - Prompts
  - Chosen responses
  - Rejected responses
- **Quality checks**:
  - Empty prompts/responses
  - Identical chosen/rejected responses
  - Missing fields

**Output**: Formatted report printed to stdout and optionally saved to file

**Usage**:

```bash
# Run evaluation on default paths
python -m scripts.kat_eval_pairs

# Use custom input paths
python -m scripts.kat_eval_pairs --pairs-file /path/to/pairs.jsonl --prompts-file /path/to/prompts.jsonl

# Save report to file
python -m scripts.kat_eval_pairs --output-report .cache/data/eval_report.txt
```

## Data Flow

```
[HuggingFace Datasets]
    |
    v
[kat_download_pairs.py]
    |
    v
.cache/data/pairs_all.jsonl
    |
    v
[kat_make_prompts.py]
    |
    +---> .cache/data/prompts_all.jsonl
    |
    +---> .cache/data/prompt_id_map.tsv
    |
    +---> .cache/data/stats.txt
    |
    v
[kat_eval_pairs.py]
    |
    v
Evaluation Report
```

## Storage Considerations

The three datasets are **large**:

- HH-RLHF: ~160k pairs
- UltraFeedback: ~64k pairs
- Stack Exchange: ~52k pairs

**Total**: ~276k pairs, but may be reduced significantly after deduplication.

The JSONL files are gzippable if needed. Typical size:

- `pairs_all.jsonl`: ~500 MB (uncompressed)
- `prompts_all.jsonl`: ~50 MB (uncompressed)

If storage is an issue, you can:

- Download only one dataset at a time: `python -m scripts.kat_download_pairs --only hh`
- Keep only one of the output files (e.g., just `prompts_all.jsonl` for downstream use)
- Compress with gzip after processing

## Integration with Density Sampling

These scripts prepare data for the density-aware RLHF approach mentioned in the main README. The next steps would be:

1. **Compute prompt embeddings**: Convert each prompt to a fixed-dimensional embedding using a pretrained model (e.g., sentence-transformers)

2. **Build embedding space**: Store embeddings in a database or cache for fast lookup

3. **Estimate local density**: For each prompt, estimate the local density of prompts in embedding space (e.g., using k-NN)

4. **Inverse density weighting**: When sampling during RLHF training, weight prompts inversely by their local density

   - High-density prompts (common/similar to others): lower sampling weight
   - Low-density prompts (rare/unique): higher sampling weight

5. **RLHF with weighted sampling**: Use the density weights during preference learning to encourage diversity

## Dependencies

These scripts require:

- `datasets` (HuggingFace datasets library)
- Standard library: `argparse`, `json`, `os`, `hashlib`, `re`, `collections`, `sys`, `pathlib`

All should be available in the existing nanochat environment.

## Troubleshooting

**Issue**: "HuggingFaceH4/ultrafeedback_binarized not found"

- **Cause**: Dataset may be temporarily unavailable or requires authentication
- **Solution**: Try again later, or skip with `--no-uf` flag

**Issue**: Out of memory during download

- **Cause**: Datasets are large and loaded into memory during processing
- **Solution**: Download only one dataset at a time, or run on a machine with more RAM

**Issue**: Very few unique prompts after deduplication

- **Cause**: The datasets have significant overlap
- **Solution**: This is expected! It validates the need for proper deduplication in downstream processing

## Future Enhancements

Potential improvements to this pipeline:

- Add semantic deduplication (beyond exact text matching) using embeddings
- Implement filtering by response quality (e.g., remove very short responses)
- Add source weighting to adjust the distribution of dataset contributions
- Support incremental updates (append new pairs without reprocessing everything)
- Add more datasets (e.g., OpenHermes, Orca, etc.)

---

**Created**: October 2025  
**Purpose**: Support density-aware RLHF research in nanochat
