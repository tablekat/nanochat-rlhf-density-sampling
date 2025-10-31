#!/usr/bin/env python3
"""Evaluate reward models on preference pairs.

This compares one or more trained reward heads on a shared set of
preference pairs, reporting accuracy, weighted accuracy (using inverse
density weights when available), and reward margins per dataset source
and density bin.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from nanochat.checkpoint_manager import get_base_dir, load_model
from scripts.kat_utils import ensure_prefix_dict, prefix_id_from_prefix, render_prefix_for_completion


# -----------------------------------------------------------------------------
# Shared helpers (mirrors kat_train_rm)


def truncate_two(p: List[int], r: List[int], max_len: int, min_prompt: int) -> Tuple[List[int], List[int]]:
    """Trim prompt from left, response from right, preserving response priority."""
    if len(p) + len(r) <= max_len:
        return p, r
    resp_budget = max_len - min(len(p), min_prompt)
    resp_budget = max(resp_budget, 1)
    r = r[:resp_budget]
    over = (len(p) + len(r)) - max_len
    if over > 0:
        p = p[over:]
    return p, r


def pad_sequences(seqs: List[List[int]], max_len: int, pad_id: int, device: torch.device) -> torch.Tensor:
    seqs = [seq[:max_len] for seq in seqs]
    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in seqs]
    return torch.tensor(padded, dtype=torch.long, device=device)


def build_dual_sequences(
    rows: List[PairRow],
    tokenizer,
    max_len: int,
    min_prompt: int,
    device: torch.device,
    pad_id: int,
    rating_prompt_ids: List[int],
    response_a_prefix_ids: List[int],
    response_a_suffix_ids: List[int],
    response_b_prefix_ids: List[int],
    response_b_suffix_ids: List[int],
    shuffle_seed: int,
    preferred_token_id: int,
    rejected_token_id: int,
) -> Tuple[torch.Tensor, List[int], List[int], List[int], List[int], List[float]]:
    random.seed(shuffle_seed)
    rating_prompt_len = len(rating_prompt_ids)
    effective_max_len = max(1, max_len - rating_prompt_len)

    sequences: List[List[int]] = []
    digit1_indices: List[int] = []
    digit2_indices: List[int] = []
    digit1_tokens: List[int] = []
    digit2_tokens: List[int] = []
    weights: List[float] = []

    for row in rows:
        try:
            prefix_ids = render_prefix_for_completion(tokenizer, row.prefix)
        except ValueError:
            prefix_ids = render_prefix_for_completion(tokenizer, None)

        chosen_ids = tokenizer.encode(row.chosen)
        rejected_ids = tokenizer.encode(row.rejected)

        prompt_chosen, chosen_trimmed = truncate_two(prefix_ids, chosen_ids, effective_max_len, min_prompt)
        prompt_rejected, rejected_trimmed = truncate_two(prefix_ids, rejected_ids, effective_max_len, min_prompt)

        chosen_seq = prompt_chosen + chosen_trimmed
        rejected_seq = prompt_rejected + rejected_trimmed

        if random.random() < 0.5:
            first_response = chosen_seq
            second_response = rejected_seq
            first_digit = preferred_token_id
            second_digit = rejected_token_id
        else:
            first_response = rejected_seq
            second_response = chosen_seq
            first_digit = rejected_token_id
            second_digit = preferred_token_id

        assembled = (
            prefix_ids
            + response_a_prefix_ids
            + first_response
            + response_a_suffix_ids
            + response_b_prefix_ids
            + second_response
            + response_b_suffix_ids
            + rating_prompt_ids
        )

        digit1_idx = len(assembled)
        assembled.append(first_digit)
        digit2_idx = len(assembled)
        assembled.append(second_digit)

        assert len(assembled) <= max_len, "Sequence exceeded max_len after appending digits"

        sequences.append(assembled)
        digit1_indices.append(digit1_idx)
        digit2_indices.append(digit2_idx)
        digit1_tokens.append(first_digit)
        digit2_tokens.append(second_digit)
        weights.append(row.weight)

    padded = pad_sequences(sequences, max_len, pad_id, device)
    return padded, digit1_indices, digit2_indices, digit1_tokens, digit2_tokens, weights


# -----------------------------------------------------------------------------
# Data loading


@dataclass
class PairRow:
    prefix: dict
    chosen: str
    rejected: str
    src: str
    prefix_id: Optional[str]
    weight: float
    weight_bin: Optional[str]


def load_density_mapping(prefixes_path: Path, weights_path: Path) -> Optional[Dict[str, float]]:
    if not prefixes_path.exists() or not weights_path.exists():
        return None
    ids: List[str] = []
    with prefixes_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                ids.append(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    weights = np.load(weights_path)
    if len(ids) != len(weights):
        raise RuntimeError("prefixes_all.jsonl and density_weights.npy misaligned")
    return {pid: float(w) for pid, w in zip(ids, weights.tolist())}


def assign_weight_bins(weights: List[float], num_bins: int = 5) -> Tuple[List[str], List[float]]:
    if not weights:
        return [], []
    quantiles = np.linspace(0, 1, num_bins + 1)
    edges = np.quantile(np.array(weights), quantiles)
    labels = []
    for i in range(num_bins):
        lo = edges[i]
        hi = edges[i + 1]
        labels.append(f"q{i+1}:{lo:.4f}-{hi:.4f}")
    return labels, edges.tolist()


def load_pairs(
    pairs_path: Path,
    max_examples: Optional[int],
    seed: int,
    density_map: Optional[Dict[str, float]],
    weight_bins: Optional[Tuple[List[str], List[float]]],
    filter_sources: Optional[Iterable[str]] = None,
) -> List[PairRow]:
    rows: List[PairRow] = []
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            prefix = ensure_prefix_dict(ex.get("prefix"))
            chosen = ex.get("chosen")
            rejected = ex.get("rejected")
            src = ex.get("src", "unknown")
            if not chosen or not rejected:
                continue
            if filter_sources and src not in filter_sources:
                continue

            prefix_id = ex.get("prefix_id") or prefix_id_from_prefix(prefix)
            weight = 1.0
            if density_map and prefix_id:
                weight = density_map.get(prefix_id, 1.0)

            bin_label = None
            if weight_bins and weight is not None:
                labels, edges = weight_bins
                for idx in range(len(labels)):
                    lo, hi = edges[idx], edges[idx + 1]
                    # include upper edge on last bin
                    if (weight >= lo and (weight < hi or (idx == len(labels) - 1 and weight <= hi))):
                        bin_label = labels[idx]
                        break

            rows.append(PairRow(
                prefix=prefix,
                chosen=chosen,
                rejected=rejected,
                src=src,
                prefix_id=prefix_id,
                weight=weight,
                weight_bin=bin_label,
            ))

    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_examples is not None and len(rows) > max_examples:
        rows = rows[:max_examples]
    return rows


# -----------------------------------------------------------------------------
# Metrics aggregation


class MetricBucket:
    def __init__(self) -> None:
        self.total = 0
        self.correct = 0
        self.weight_sum = 0.0
        self.weight_correct = 0.0
        self.margins: List[float] = []

    def add(self, correct: bool, margin: float, weight: float) -> None:
        self.total += 1
        self.correct += int(correct)
        self.weight_sum += weight
        self.weight_correct += weight * int(correct)
        self.margins.append(float(margin))

    def summary(self) -> Dict[str, float]:
        if self.total == 0:
            return {}
        acc = self.correct / self.total
        weighted_acc = self.weight_correct / self.weight_sum if self.weight_sum > 0 else float("nan")
        margins = np.array(self.margins, dtype=np.float32)
        return {
            "count": self.total,
            "accuracy": acc,
            "weighted_accuracy": weighted_acc,
            "margin_mean": float(margins.mean()),
            "margin_std": float(margins.std()),
            "margin_p25": float(np.percentile(margins, 25)),
            "margin_p50": float(np.percentile(margins, 50)),
            "margin_p75": float(np.percentile(margins, 75)),
        }


class Metrics:
    def __init__(self, name: str) -> None:
        self.name = name
        self.total = MetricBucket()
        self.by_src: Dict[str, MetricBucket] = defaultdict(MetricBucket)
        self.by_weight_bin: Dict[str, MetricBucket] = defaultdict(MetricBucket)

    def add(self, src: str, weight_bin: Optional[str], correct: bool, margin: float, weight: float) -> None:
        self.total.add(correct, margin, weight)
        self.by_src[src].add(correct, margin, weight)
        if weight_bin is not None:
            self.by_weight_bin[weight_bin].add(correct, margin, weight)

    def summarize(self) -> Dict[str, Dict[str, float]]:
        return {
            "overall": self.total.summary(),
            "by_src": {k: v.summary() for k, v in sorted(self.by_src.items())},
            "by_weight_bin": {k: v.summary() for k, v in sorted(self.by_weight_bin.items())},
        }


# -----------------------------------------------------------------------------
# RM loading


def find_latest_checkpoint(directory: Path) -> Path:
    ckpts = glob.glob(str(directory / "model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {directory}")
    return max(ckpts, key=lambda p: int(Path(p).stem.split("_")[1]))


def load_reward_configs(
    sources: List[str],
) -> List[Tuple[str, Path, Dict[str, object], List[int], Dict[str, torch.Tensor]]]:
    base = Path(get_base_dir())
    mapping = {
        "rm": base / "rm_checkpoints" / "uniform" / "d20_dual",
        "rm_density": base / "rm_checkpoints" / "density" / "d20_dual",
    }
    configs = []
    for source in sources:
        if source not in mapping:
            raise ValueError(f"Unknown rm_source '{source}'")
        directory = mapping[source]
        ckpt_path = find_latest_checkpoint(directory)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        meta = ckpt.get("meta", {})
        blocks_state = ckpt.get("backbone_blocks_state_dict")
        if blocks_state is None:
            # Backwards compatibility with single-block checkpoints
            single_state = ckpt.get("backbone_block_state_dict")
            block_idx = meta.get("backbone_block_index", -1)
            if single_state is not None:
                blocks_state = {str(block_idx): single_state}
            block_indices = [block_idx]
        else:
            block_indices = meta.get("backbone_block_indices", [int(k) for k in blocks_state.keys()])
        configs.append((source, Path(ckpt_path), meta, block_indices, blocks_state or {}))
    return configs


# -----------------------------------------------------------------------------
# Evaluation loop


def evaluate(args):
    device = torch.device(args.device)

    # Load density weights
    base_dir = Path(get_base_dir())
    density_map = None
    weight_bins = None
    if not args.skip_density_weights:
        prefixes_path = args.prefixes_path or (base_dir / "data" / "prefixes_all.jsonl")
        weights_path = args.density_weights_path or (base_dir / "data" / "embeddings_offline" / "density_weights.npy")
        density_map = load_density_mapping(Path(prefixes_path), Path(weights_path))
        if density_map:
            weight_bins = assign_weight_bins(list(density_map.values()), num_bins=args.density_bins)

    pairs_path = Path(args.pairs_path or (base_dir / "data" / "pairs_all.jsonl"))
    rows = load_pairs(
        pairs_path,
        max_examples=args.max_examples,
        seed=args.seed,
        density_map=density_map,
        weight_bins=weight_bins,
        filter_sources=set(args.filter_sources.split(",")) if args.filter_sources else None,
    )
    if not rows:
        raise RuntimeError("No evaluation pairs loaded")

    print(f"Loaded {len(rows)} evaluation pairs")

    configs = load_reward_configs(args.rm_sources.split(","))

    batch_size = args.batch_size
    max_len = args.max_len
    min_prompt = args.min_prompt

    summaries = {}
    for name, ckpt_path, meta, block_indices, blocks_state in configs:
        print(f"Evaluating reward model '{name}' from {ckpt_path}")
        backbone, tokenizer, _ = load_model("sft", device=device, phase="eval")
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad_(False)
        if blocks_state:
            for idx in block_indices:
                state = blocks_state.get(str(idx))
                if state is None:
                    continue
                backbone.transformer.h[idx].load_state_dict(state)
        pad_id = tokenizer.encode_special("<|assistant_end|>")

        rating_prompt = meta.get("rating_prompt", "\nRating (Response A first, Response B second):")
        preferred_digit = meta.get("preferred_digit", "7")
        rejected_digit = meta.get("rejected_digit", "1")

        response_a_prefix_ids = tokenizer.encode("\n### Response A ###\n")
        response_a_suffix_ids = tokenizer.encode("\n### End Response A ###\n")
        response_b_prefix_ids = tokenizer.encode("\n### Response B ###\n")
        response_b_suffix_ids = tokenizer.encode("\n### End Response B ###\n")
        rating_prompt_ids = tokenizer.encode(rating_prompt)
        if not rating_prompt_ids:
            raise ValueError("rating_prompt must tokenize to at least one token")

        preferred_token_ids = tokenizer.encode(preferred_digit)
        rejected_token_ids = tokenizer.encode(rejected_digit)
        if len(preferred_token_ids) != 1 or len(rejected_token_ids) != 1:
            raise ValueError("Likert digits must map to single tokens")
        preferred_token_id = preferred_token_ids[0]
        rejected_token_id = rejected_token_ids[0]

        metric = Metrics(name)

        for start in tqdm(range(0, len(rows), batch_size), desc=f"{name} eval"):
            batch = rows[start:start + batch_size]
            sources = [row.src for row in batch]
            weights = [row.weight for row in batch]
            bins = [row.weight_bin for row in batch]

            inputs, digit1_idx, digit2_idx, digit1_tokens, digit2_tokens, _ = build_dual_sequences(
                batch,
                tokenizer,
                max_len,
                min_prompt,
                device,
                pad_id,
                rating_prompt_ids,
                response_a_prefix_ids,
                response_a_suffix_ids,
                response_b_prefix_ids,
                response_b_suffix_ids,
                shuffle_seed=0,
                preferred_token_id=preferred_token_id,
                rejected_token_id=rejected_token_id,
            )

            digit1_idx_tensor = torch.tensor(digit1_idx, dtype=torch.long, device=device)
            digit2_idx_tensor = torch.tensor(digit2_idx, dtype=torch.long, device=device)
            digit1_tokens_tensor = torch.tensor(digit1_tokens, dtype=torch.long, device=device)
            digit2_tokens_tensor = torch.tensor(digit2_tokens, dtype=torch.long, device=device)
            batch_idx = torch.arange(inputs.size(0), device=device)

            with torch.no_grad():
                logits = backbone(inputs)

            logits_digit1 = logits[batch_idx, digit1_idx_tensor - 1, :]
            logits_digit2 = logits[batch_idx, digit2_idx_tensor - 1, :]
            reward_first = logits_digit1.gather(1, digit1_tokens_tensor.unsqueeze(1)).squeeze(1)
            reward_second = logits_digit2.gather(1, digit2_tokens_tensor.unsqueeze(1)).squeeze(1)

            first_is_preferred = digit1_tokens_tensor == preferred_token_id
            reward_chosen = torch.where(first_is_preferred, reward_first, reward_second)
            reward_rejected = torch.where(first_is_preferred, reward_second, reward_first)
            margins = (reward_chosen - reward_rejected).detach().cpu().numpy()

            correct = margins > 0
            for src, bin_label, ok, margin_val, weight_val in zip(sources, bins, correct, margins, weights):
                metric.add(src, bin_label, bool(ok), float(margin_val), float(weight_val))

        summaries[name] = metric.summarize()

    if args.output_json:
        os.makedirs(Path(args.output_json).parent, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"Saved metrics to {args.output_json}")

    for name, summary in summaries.items():
        print("=" * 80)
        print(f"Reward model: {name}")
        if summary["overall"]:
            overall = summary["overall"]
            print(f"  Accuracy:        {overall['accuracy']*100:6.2f}%")
            if not np.isnan(overall['weighted_accuracy']):
                print(f"  Weighted Acc.:   {overall['weighted_accuracy']*100:6.2f}%")
            print(f"  Margin mean:     {overall['margin_mean']:.4f} ± {overall['margin_std']:.4f}")
            print(f"  Margin quartiles:{overall['margin_p25']:.4f} | {overall['margin_p50']:.4f} | {overall['margin_p75']:.4f}")

        if summary["by_src"]:
            print("  By source:")
            for src, stats in summary["by_src"].items():
                print(f"    {src:25s} -> acc {stats['accuracy']*100:6.2f}% | margin {stats['margin_mean']:.4f}")

        if summary["by_weight_bin"]:
            print("  By density quantile:")
            for bin_label, stats in summary["by_weight_bin"].items():
                print(f"    {bin_label:20s} -> acc {stats['accuracy']*100:6.2f}% | margin {stats['margin_mean']:.4f}")

    return summaries


# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reward models on preference data")
    parser.add_argument("--rm-sources", default="rm,rm_density", help="Comma-separated rm sources (rm|rm_density)")
    parser.add_argument("--pairs-path", default=None, help="Path to pairs JSONL (default: data/pairs_all.jsonl)")
    parser.add_argument("--prefixes-path", default=None, help="Path to prefixes_all.jsonl for density weights")
    parser.add_argument("--density-weights-path", default=None, help="Path to density_weights.npy")
    parser.add_argument("--skip-density-weights", action="store_true", help="Disable weighted metrics")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap evaluation set size")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--min-prompt", type=int, default=128)
    parser.add_argument("--density-bins", type=int, default=5, help="Number of quantile bins for density stats")
    parser.add_argument("--filter-sources", default=None, help="Comma-separated src values to keep")
    parser.add_argument("--output-json", default=None, help="Optional path to dump metrics as JSON")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())


