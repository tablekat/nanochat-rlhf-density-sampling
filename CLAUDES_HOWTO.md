# How to Use the Pairwise Preferences Data Pipeline & GRPO Training

This guide provides exact commands to download, process, and train models with GRPO (Generalized Reward Policy Optimization) using density-aware sampling for improved diversity and reduced mode collapse.

## ⚡ Quick Start: Automated Complete Pipeline

If you want to run **everything automatically** from scratch (including tokenizer, pretraining, SFT):

```bash
# Option 1: Simple (logs to console)
bash kat_speedrun.sh

# Option 2: In a screen session (recommended, runs ~1-2 days)
screen -L -Logfile kat_speedrun.log -S kat_speedrun bash kat_speedrun.sh

# Option 3: With wandb logging
WANDB_RUN=my_experiment bash kat_speedrun.sh
```

**What it does** (all automatic):

1. ✅ Setup Python environment
2. ✅ Train tokenizer (if needed)
3. ✅ Pretrain base model (depth=20)
4. ✅ Mid-training
5. ✅ Supervised fine-tuning (SFT)
6. ✅ Download & deduplicate preference pairs
7. ✅ Train reward model
8. ✅ **GRPO with density sampling** (main experiment)
9. ✅ **GRPO baseline** (uniform sampling)
10. ✅ **Evaluate both** and generate report

**Output**:

- `outs/grpo_density/ckpt.pt` — Density-aware model
- `outs/grpo_baseline/ckpt.pt` — Baseline model
- `.cache/diversity_report.md` — Full evaluation report

**Time estimate**: ~1-2 days on 8xH100 GPU node

---

## Manual Step-by-Step (if you already have SFT)

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
kat_make_prompts.py
kat_train_rm.py
kat_train_grpo.py
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

---

## Stage 3: Train Reward Model

### 3.1 Train RM on SFT Checkpoint

Before running GRPO, you need a trained reward model. This script trains an RM head on top of your SFT checkpoint to predict preference scores:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_rm
```

**Expected behavior**:

- Loads the SFT checkpoint (e.g., `outs/sft/ckpt.pt`)
- Trains a scalar reward head using the pairs data
- Saves RM checkpoint to `outs/rm/ckpt.pt`

**Time estimate**: 2-4 hours on 8xH100

**Key hyperparameters** (see `kat_train_rm.py`):

- `--max_steps`: Number of training steps (default: reasonable default based on dataset size)
- `--learning_rate`: Learning rate for RM training
- `--batch_size_pairs`: How many pair comparisons per batch
- `--device_batch_size`: Per-device batch size

**Custom options**:

```bash
# Train with custom learning rate
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_rm -- --learning_rate=1e-4

# Train on fewer steps for testing
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_rm -- --max_steps=100

# Specify custom input/output paths
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_rm -- \
  --sft_ckpt_path outs/sft/ckpt.pt \
  --pairs_path .cache/data/pairs_all.jsonl \
  --out_path outs/rm/ckpt.pt
```

---

## Stage 4: Train with GRPO

### 4.1 Train with Density-Aware GRPO

After training the reward model, train the policy with GRPO using density-aware sampling:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo
```

**Expected behavior**:

- Loads the SFT checkpoint as policy
- Loads the RM checkpoint for reward scoring
- Samples preference pairs with density weighting (inversely proportional to prompt density)
- Optimizes policy using GRPO loss
- Saves checkpoints to `outs/grpo/`

**Typical Results**:

- Training loss should decrease gradually
- Sampling weight distribution should show higher variance with density weighting
- Training time: 4-8 hours on 8xH100

---

**Key hyperparameters**:

- `--max_steps`: Total training steps
- `--learning_rate`: Policy learning rate
- `--beta`: KL divergence penalty (higher = closer to SFT)
- `--density_aware`: Enable/disable density weighting (default: True)
- `--density_k`: k-NN parameter for local density estimation

**Custom options**:

```bash
# Train without density weighting (baseline for comparison)
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- --density_aware=False

# Adjust KL penalty strength
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- --beta=0.1

# Custom paths
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- \
  --sft_ckpt_path outs/sft/ckpt.pt \
  --rm_ckpt_path outs/rm/ckpt.pt \
  --pairs_path .cache/data/pairs_all.jsonl \
  --out_dir outs/grpo/
```

---

## Evaluation: Diversity & Mode Collapse Metrics

### Key Hypothesis

The goal is to test if density-aware sampling reduces mode collapse. Expected improvements:

- ✅ **Reduced em-dash frequency**: Model uses em-dashes less often
- ✅ **Better token diversity**: Token frequency distribution is flatter (less peaked)
- ✅ **Improved grammar**: Grammar mistakes reduced (especially in non-English contexts)
- ✅ **Better personality matching**: Model can adjust tone/style based on context
- ✅ **More varied writing**: Less repetitive phrases and patterns

### 4.1 Evaluate Diversity: Run Comparison

After training both with (`--density_aware=True`) and without (`--density_aware=False`), evaluate:

```bash
# Generate outputs from both models
python -m scripts.chat_cli --ckpt_path outs/grpo/ckpt.pt > outputs_grpo_density.txt
python -m scripts.chat_cli --ckpt_path outs/grpo_baseline/ckpt.pt > outputs_grpo_baseline.txt
```

### 4.2 Analyze Em-Dash Frequency

```bash
# Count em-dashes in both outputs
echo "GRPO with density sampling:" && grep -o "—" outputs_grpo_density.txt | wc -l
echo "GRPO baseline:" && grep -o "—" outputs_grpo_baseline.txt | wc -l
```

### 4.3 Token Frequency Analysis

Script to analyze vocabulary diversity (Gini coefficient):

```bash
python << 'EOF'
import re
from collections import Counter

def gini_coefficient(tokens):
    """Calculate Gini coefficient for token distribution (0=uniform, 1=all one token)"""
    counts = Counter(tokens)
    freqs = sorted(counts.values())
    n = sum(freqs)
    return sum((2 * i + 1) * f for i, f in enumerate(freqs)) / (n * len(freqs)) - (len(freqs) + 1) / len(freqs)

def analyze_file(filename):
    with open(filename, 'r') as f:
        text = f.read().lower()

    # Simple tokenization
    tokens = re.findall(r'\b\w+\b', text)

    gini = gini_coefficient(tokens)
    print(f"{filename}: Gini coefficient = {gini:.4f}")
    print(f"  (Lower = more diverse vocabulary)")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Unique tokens: {len(set(tokens))}")
    print(f"  Diversity ratio: {len(set(tokens)) / len(tokens):.4f}")

analyze_file('outputs_grpo_density.txt')
analyze_file('outputs_grpo_baseline.txt')
EOF
```

---

## Understanding the Loss Breakdown

During GRPO training, you'll see logs like:

```
Step 100/5000 - Loss: 0.4532
  Pref Loss: 0.3245, KL Loss: 0.1287, Total: 0.4532
```

**What each term means**:

- **Pref Loss** (Preference Loss): Bradley-Terry loss on preference pairs
  - Ideally decreases over time (model learns preferences)
  - Range: typically 0.3-0.8
- **KL Loss** (Kullback-Leibler Divergence): Divergence from SFT model
  - Measures how far policy drifted from reference
  - Higher KL = policy diverging more from SFT
  - With `--beta=0.1`: KL should be ~1/3 of total loss
- **Total Loss**: `pref_loss + beta * kl_loss`
  - Should decrease gradually over training
  - If stuck: adjust `--beta` (higher = stay closer to SFT)

**Adjusting the balance**:

```bash
# More reward optimization (less KL penalty)
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- --beta=0.01

# More conservative (higher KL penalty, closer to SFT)
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- --beta=0.5
```

---

### 5.1 Baseline Comparison (Without Density)

---

## Full Pipeline: One Command

If you want to run everything in sequence (warning: this takes ~1-2 days):

```bash
echo "=== Stage 1: Download ===" && \
python -m scripts.kat_download_pairs && \
echo "" && \
echo "=== Stage 2: Extract & Deduplicate ===" && \
python -m scripts.kat_make_prompts && \
echo "" && \
echo "=== Stage 3: Train Reward Model ===" && \
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_rm && \
echo "" && \
echo "=== Stage 4: Train GRPO (Density-Aware) ===" && \
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo && \
echo "" && \
echo "=== Stage 5: Train GRPO (Baseline) ===" && \
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- --density_aware=False && \
echo "" && \
echo "=== All Done! ==="
```

**Total time**: ~1-2 days on 8xH100 GPU (mostly training)

---

## Output Artifacts

### RM Training Outputs

- `outs/rm/ckpt.pt`: Trained reward model checkpoint
- `outs/rm/report.md`: Training report with loss curves

### GRPO Training Outputs

- `outs/grpo/ckpt.pt`: Policy trained with density sampling
- `outs/grpo/report.md`: Training report and diversity metrics
- `outs/grpo_baseline/ckpt.pt` (if running baseline): Baseline without density sampling
- `outs/grpo_baseline/report.md`: Baseline training report

---

## Troubleshooting

### Problem: "SFT checkpoint not found"

**Solution**: Make sure you've run SFT training first:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
```

### Problem: Out of Memory during RM training

**Solution**: Reduce batch size:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_rm -- --device_batch_size=8
```

### Problem: RM not improving (loss stays flat)

**Solution**:

1. Check that pairs data is valid: `head -5 .cache/data/pairs_all.jsonl`
2. Try higher learning rate: `--learning_rate=5e-4`
3. Check RM architecture parameters

### Problem: GRPO training very slow

**Solution**:

- Reduce `--max_steps` for testing
- Decrease `--device_batch_size`
- Disable density computation for initial testing: `--density_aware=False`

### Problem: Files not found

**Solution**: Make sure you're in repo root and all stages completed:

```bash
cd /path/to/nanochat-rlhf-density-sampling
ls -la outs/sft/ckpt.pt
ls -la .cache/data/pairs_all.jsonl
```

---

## Experimental Design: Testing the Hypothesis

### Baseline Comparison

To properly test the hypothesis, run both:

1. **With density sampling** (main experiment):

   ```bash
   torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo
   ```

2. **Without density sampling** (control):
   ```bash
   torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- --density_aware=False
   ```

### Metrics to Track

| Metric            | How to Measure                     | Expected Improvement |
| ----------------- | ---------------------------------- | -------------------- |
| Em-dash frequency | `grep -o "—" output.txt \| wc -l`  | 20-40% reduction     |
| Token Gini coeff  | Gini script above                  | Lower = more diverse |
| Vocab diversity   | Unique tokens / total              | 5-15% improvement    |
| Grammar errors    | Manual review or language model    | 10-20% reduction     |
| Task performance  | Benchmark scores (ARC, GSM8K, etc) | Should not degrade   |

### Analysis Steps

1. Generate N prompts from both models
2. Measure frequency of target artifacts
3. Compare token distributions
4. Run standard benchmarks
5. Compare diversity metrics

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

---

## Performance Notes

Approximate performance on 8xH100 GPU node:

| Stage                   | Time          | Notes                             |
| ----------------------- | ------------- | --------------------------------- |
| Download (all)          | 20-40 min     | Network dependent                 |
| Extract & deduplicate   | 2-5 min       | Deterministic prompt IDs          |
| Train RM                | 2-4 hours     | On ~276k pairs                    |
| Train GRPO (w/ density) | 4-8 hours     | Includes density computation      |
| Train GRPO (baseline)   | 3-6 hours     | Faster, no density weighting      |
| **Total**               | **~1-2 days** | Most time in RM and GRPO training |

---

## Automated Evaluation: Diversity Report

After training both models, use the evaluation script to automatically test the hypothesis:

```bash
# Generate comprehensive diversity report
python -m scripts.kat_eval_diversity \
    --density_model_path outs/grpo_density/ckpt.pt \
    --baseline_model_path outs/grpo_baseline/ckpt.pt \
    --output_report .cache/diversity_report.md \
    --num_prompts 100
```

### What the Report Measures

The evaluation script analyzes:

1. **Em-dash Frequency**

   - Count of `—` and `–` characters
   - Indicator of repetitive punctuation patterns (mode collapse)
   - **Expected**: Density model has fewer em-dashes

2. **Gini Coefficient** (token diversity)

   - 0 = perfectly uniform distribution (ideal, all tokens equal)
   - 1 = complete collapse (only one token used)
   - **Expected**: Density model closer to 0 (more diverse)

3. **Vocabulary Ratio**

   - Unique tokens / total tokens
   - **Expected**: Density model has higher ratio (more word variety)

4. **Response Statistics**

   - Sentence length, word count, character count
   - **Expected**: Less repetition, more varied responses

5. **Repetition Patterns**
   - Counts phrases like "not just X but Y"
   - **Expected**: Fewer in density model

### Report Output

The script generates a markdown report with:

- ✅/❌ colored interpretation of each metric
- Percentage improvement scores
- Key findings and recommendations
- Methodology explanation

**Example output**:

```
✅ Em-dash Frequency: 45.2% reduction (GOOD)
   - Density sampling reduces repetitive punctuation patterns

✅ Gini Coefficient: 12.8% improvement (GOOD)
   - Token distribution is more uniform/diverse with density sampling

✅ Vocabulary Ratio: 8.5% improvement (GOOD)
   - Model uses wider variety of words
```

### View the Report

```bash
# Display report
cat .cache/diversity_report.md

# Or open in editor
nano .cache/diversity_report.md
```

---

## Next Steps

After completing this pipeline:

1. **Compare models**: Generate outputs and analyze for mode collapse reduction
2. **Measure diversity**: Use token frequency and artifact detection
3. **Track performance**: Ensure quality metrics don't degrade
4. **Iterate hyperparameters**: Adjust `--beta`, `--density_k`, etc.
5. **Write results**: Document findings for hypothesis validation

Good luck! 🚀

---

**Last updated**: October 2025
**Focus**: Density-aware GRPO for reducing mode collapse
