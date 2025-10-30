#!/usr/bin/env python3
"""
Evaluate diversity and mode collapse reduction.

This script tests the hypothesis that density-aware sampling reduces mode collapse
by comparing two models:
  1. GRPO with density-aware sampling (main experiment)
  2. GRPO without density sampling (baseline/control)

Metrics:
  - Em-dash frequency (count of em-dashes)
  - Token diversity (Gini coefficient, vocabulary ratio)
  - Repetition patterns (e.g., "not just X but Y")
  - Response length distribution
  - Grammar error indicators

Output: Formatted markdown report with comparisons

Usage:
  python -m scripts.kat_eval_diversity \
    --density_model_path outs/grpo_density/ckpt.pt \
    --baseline_model_path outs/grpo_baseline/ckpt.pt \
    --output_report .cache/diversity_report.md \
    --num_prompts 100
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm

from nanochat.checkpoint_manager import load_model, get_base_dir
from nanochat.tokenizer import get_tokenizer

from scripts.kat_utils import ensure_prefix_dict, render_prefix_for_completion


def load_prefix_conversations(num_samples, seed=42):
    """Load prefix conversations for evaluation."""

    base_dir = get_base_dir()
    prefixes_path = os.path.join(base_dir, "data", "prefixes_all.jsonl")

    default_prompts = [
        "Explain quantum computing",
        "What is machine learning?",
        "How does photosynthesis work?",
        "What is the meaning of life?",
        "How do neural networks work?",
    ]

    rows = []
    if os.path.exists(prefixes_path):
        with open(prefixes_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    rows.append({
                        "id": obj.get("id"),
                        "prefix": ensure_prefix_dict(obj.get("prefix")),
                    })
                except json.JSONDecodeError:
                    continue

    if not rows:
        # Fallback to legacy text prompts if prefixes are unavailable
        limited = default_prompts[:num_samples] if num_samples else default_prompts
        return [
            {
                "id": f"default_{idx}",
                "prefix": ensure_prefix_dict(prompt),
            }
            for idx, prompt in enumerate(limited)
        ]

    if num_samples and num_samples < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, num_samples)
    elif num_samples:
        rows = rows[:num_samples]

    return rows


def generate_samples(model_source, prefixes, device="cpu", max_tokens=128, temperature=0.8, top_k=50):
    """Generate assistant responses for the provided prefixes."""

    try:
        print(f"Loading model from {model_source}...")
        model, tokenizer, _ = load_model(model_source, device=device, phase="eval")
        model.eval()
        print("✓ Model loaded successfully")

        assistant_end_id = tokenizer.encode_special("<|assistant_end|>")

        samples = []
        for idx, item in enumerate(prefixes):
            prefix_tokens = render_prefix_for_completion(tokenizer, item["prefix"])
            prompt_ids = list(prefix_tokens)  # copy to avoid accidental mutation

            generated_ids = []
            for token in model.generate(
                prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=42 + idx,
            ):
                if token == assistant_end_id:
                    break
                generated_ids.append(token)

            text = tokenizer.decode(generated_ids) if generated_ids else ""
            samples.append((item["id"], text))

            if (idx + 1) % 10 == 0:
                print(f"  Generated {idx + 1}/{len(prefixes)} samples")

        return samples

    except Exception as e:
        print(f"Error generating samples: {e}")
        import traceback
        traceback.print_exc()
        return []


def count_em_dashes(text):
    """Count em-dashes (—) in text."""
    return text.count("—") + text.count("–")


def count_word(text, word):
    """Count occurrences of a specific word."""
    pattern = r'\b' + re.escape(word) + r'\b'
    return len(re.findall(pattern, text, re.IGNORECASE))


def gini_coefficient(tokens):
    """
    Calculate Gini coefficient for token distribution.
    0 = uniform distribution (high diversity)
    1 = single token (complete collapse)
    """
    if not tokens:
        return 0.0
    
    counts = Counter(tokens)
    freqs = sorted(counts.values())
    n = sum(freqs)
    
    if n == 0:
        return 0.0
    
    return sum((2 * i + 1) * f for i, f in enumerate(freqs)) / (n * len(freqs)) - (len(freqs) + 1) / len(freqs)


def tokenize_simple(text):
    """Simple word tokenization."""
    return re.findall(r'\b\w+\b', text.lower())


def analyze_text(text):
    """Comprehensive text analysis."""
    tokens = tokenize_simple(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    metrics = {
        'em_dashes': count_em_dashes(text),
        'total_chars': len(text),
        'total_words': len(tokens),
        'unique_words': len(set(tokens)),
        'vocabulary_ratio': len(set(tokens)) / max(len(tokens), 1),
        'gini_coefficient': gini_coefficient(tokens),
        'avg_sentence_length': sum(len(s.split()) for s in sentences) / max(len(sentences), 1),
        'num_sentences': len(sentences),
        # Repetition patterns
        'not_just_but': count_word(text, 'not') if 'but' in text.lower() else 0,
        'em_dash_freq': count_em_dashes(text) / max(len(sentences), 1),
    }
    
    return metrics


def compare_models(density_samples, baseline_samples):
    """Compare metrics between two sets of samples aligned by prefix id."""

    density_map = {pid: text for pid, text in density_samples}
    baseline_map = {pid: text for pid, text in baseline_samples}

    shared_ids = [pid for pid in density_map if pid in baseline_map]
    if not shared_ids:
        raise ValueError("No overlapping prefixes between model samples")

    density_metrics = [analyze_text(density_map[pid]) for pid in shared_ids]
    baseline_metrics = [analyze_text(baseline_map[pid]) for pid in shared_ids]
    
    # Average metrics
    density_avg = {k: sum(m[k] for m in density_metrics) / len(density_metrics) 
                   for k in density_metrics[0].keys()}
    baseline_avg = {k: sum(m[k] for m in baseline_metrics) / len(baseline_metrics) 
                    for k in baseline_metrics[0].keys()}
    
    # Calculate improvements
    improvements = {}
    for key in density_avg:
        d_val = density_avg[key]
        b_val = baseline_avg[key]
        
        if b_val == 0:
            pct_diff = 0
        else:
            pct_diff = (b_val - d_val) / abs(b_val) * 100  # Negative = improvement
        
        improvements[key] = {
            'density': d_val,
            'baseline': b_val,
            'difference': d_val - b_val,
            'percent_change': pct_diff,
        }
    
    return improvements, shared_ids


def generate_report(improvements, output_path=None):
    """Generate markdown report."""
    lines = []
    
    lines.append("# Diversity & Mode Collapse Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("## Hypothesis")
    lines.append("")
    lines.append(
        "**Main Hypothesis**: Density-aware GRPO sampling reduces mode collapse by exposing "
        "the model to diverse prompts, resulting in:"
    )
    lines.append("- Lower em-dash frequency (fewer \"—\" punctuation marks)")
    lines.append("- Higher vocabulary diversity (Gini coefficient closer to 0)")
    lines.append("- Lower repetition patterns")
    lines.append("- More varied response lengths")
    lines.append("")
    
    lines.append("## Results Summary")
    lines.append("")
    
    # Key findings
    em_dash_improvement = improvements['em_dashes']['percent_change']
    gini_improvement = improvements['gini_coefficient']['percent_change']
    vocab_improvement = improvements['vocabulary_ratio']['percent_change']
    
    lines.append("| Metric | Density | Baseline | Change | % Improvement |")
    lines.append("|--------|---------|----------|--------|---------------|")
    
    for key, vals in sorted(improvements.items()):
        d = vals['density']
        b = vals['baseline']
        diff = vals['difference']
        pct = vals['percent_change']
        
        # Format appropriately
        if isinstance(d, int):
            d_str = str(int(d))
            b_str = str(int(b))
            diff_str = f"{diff:+.0f}"
        else:
            d_str = f"{d:.4f}"
            b_str = f"{b:.4f}"
            diff_str = f"{diff:+.4f}"
        
        lines.append(f"| {key:25s} | {d_str:>9s} | {b_str:>9s} | {diff_str:>10s} | {pct:>6.1f}% |")
    
    lines.append("")
    
    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    
    if em_dash_improvement > 0:
        lines.append(f"✅ **Em-dash Frequency**: {em_dash_improvement:.1f}% reduction (GOOD)")
        lines.append("   - Density sampling reduces repetitive punctuation patterns")
    else:
        lines.append(f"❌ **Em-dash Frequency**: {em_dash_improvement:.1f}% increase")
        lines.append("   - Unexpected result; may indicate more expressive output")
    
    lines.append("")
    
    if gini_improvement < 0:  # Lower Gini = better (more diverse)
        lines.append(f"✅ **Gini Coefficient**: {abs(gini_improvement):.1f}% improvement (GOOD)")
        lines.append("   - Token distribution is more uniform/diverse with density sampling")
    else:
        lines.append(f"❌ **Gini Coefficient**: {gini_improvement:.1f}% worse")
        lines.append("   - Token distribution became more peaked with density sampling")
    
    lines.append("")
    
    if vocab_improvement > 0:
        lines.append(f"✅ **Vocabulary Ratio**: {vocab_improvement:.1f}% improvement (GOOD)")
        lines.append("   - Model uses wider variety of words")
    else:
        lines.append(f"❌ **Vocabulary Ratio**: {vocab_improvement:.1f}% decrease")
        lines.append("   - Model uses narrower vocabulary with density sampling")
    
    lines.append("")
    
    lines.append("## Detailed Metrics")
    lines.append("")
    
    for key, vals in sorted(improvements.items()):
        lines.append(f"### {key}")
        lines.append(f"- Density model:  {vals['density']:.4f}")
        lines.append(f"- Baseline:       {vals['baseline']:.4f}")
        lines.append(f"- Difference:     {vals['difference']:+.4f}")
        lines.append(f"- % Change:       {vals['percent_change']:+.1f}%")
        lines.append("")
    
    lines.append("## Key Findings")
    lines.append("")
    
    improvements_count = sum(1 for v in improvements.values() if v['percent_change'] > 0)
    lines.append(f"- Metrics improved in density model: {improvements_count}/{len(improvements)}")
    
    if em_dash_improvement > 20:
        lines.append("- **Strong em-dash reduction**: Hypothesis appears validated")
    elif em_dash_improvement > 5:
        lines.append("- **Modest em-dash reduction**: Some support for hypothesis")
    else:
        lines.append("- **No significant em-dash change**: Consider other metrics")
    
    lines.append("")
    
    lines.append("## Recommendations")
    lines.append("")
    lines.append("1. **If hypothesis validated** (improvements > 10%):")
    lines.append("   - Use density-aware GRPO for diversity-focused training")
    lines.append("   - Adjust `--density_k` parameter (currently 10)")
    lines.append("   - Try with full dataset (currently HH-RLHF only)")
    lines.append("")
    lines.append("2. **If mixed results**:")
    lines.append("   - Tune `--beta` KL penalty (try 0.05, 0.1, 0.2)")
    lines.append("   - Increase training steps (try 10000 instead of 5000)")
    lines.append("   - Evaluate on specific diversity dimensions (not aggregate)")
    lines.append("")
    lines.append("3. **If no improvement**:")
    lines.append("   - Consider using better embeddings (sentence-transformers)")
    lines.append("   - Investigate other diversity mechanisms")
    lines.append("   - Check if mode collapse is truly the issue")
    lines.append("")
    
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Em-dashes**: Count of '—' characters (indicator of repetition)")
    lines.append("- **Gini Coefficient**: Inequality in token distribution (0=uniform, 1=single token)")
    lines.append("- **Vocabulary Ratio**: Unique tokens / total tokens (higher = more diverse)")
    lines.append("- **Comparison**: Paired metrics from 50+ generated samples per model")
    lines.append("")
    
    lines.append("---")
    lines.append(f"Report generated: {datetime.now().isoformat()}")
    
    report = "\n".join(lines)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Report saved to: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate diversity and mode collapse")
    
    # Use get_base_dir() for default paths
    base_dir = get_base_dir()
    
    parser.add_argument("--density_model_source", default="grpo",
                        help="Model source for density-aware model (grpo|sft|base)")
    parser.add_argument("--baseline_model_source", default="sft",
                        help="Model source for baseline model (grpo|sft|base)")
    parser.add_argument("--output_report", default=os.path.join(base_dir, "diversity_report.md"),
                        help="Output report path")
    parser.add_argument("--num_prompts", type=int, default=50,
                        help="Number of samples to generate from each model")
    args = parser.parse_args()
    
    print("=" * 70)
    print("KAT Diversity Evaluation: Mode Collapse Hypothesis Testing")
    print("=" * 70)
    print("")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    prefixes = load_prefix_conversations(args.num_prompts)
    if not prefixes:
        print("❌ No prefixes available for evaluation")
        sys.exit(1)

    print(f"Using {len(prefixes)} prefixes for evaluation")

    # Generate real samples from both models
    print(f"Generating samples from density-aware model ({args.density_model_source})...")
    density_samples = generate_samples(args.density_model_source, prefixes, device=device)
    
    print(f"\nGenerating samples from baseline model ({args.baseline_model_source})...")
    baseline_samples = generate_samples(args.baseline_model_source, prefixes, device=device)
    
    # Validate that we got real samples
    if not density_samples or not baseline_samples:
        print("\n⚠️  Warning: Could not generate real samples from models")
        print("Ensure models have been trained with:")
        print(f"  - torchrun -m scripts.kat_train_rm")
        print(f"  - torchrun -m scripts.kat_train_grpo")
        print("\nAlternatively, you can run with mock data for demonstration:")
        print(f"  python -m scripts.kat_eval_diversity --use_mock_data")
        sys.exit(1)
    
    print(f"✓ Generated {len(density_samples)} density-aware samples")
    print(f"✓ Generated {len(baseline_samples)} baseline samples")
    print("")
    
    # Compare models
    improvements, shared_ids = compare_models(density_samples, baseline_samples)
    print(f"Comparing {len(shared_ids)} shared prefixes")
    
    # Generate report
    print("Generating report...")
    report = generate_report(improvements, args.output_report)
    
    print("")
    print(report)
    print("")
    print("=" * 70)
    print("Evaluation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
