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
import torch.nn as nn
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


@torch.no_grad()
def extract_features(backbone, x: torch.Tensor, pad_id: int) -> torch.Tensor:
    attn = (x != pad_id)
    out = backbone(x, return_hidden_states=True)
    hidden = out["hidden_states"]
    idx = attn.long().sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(x.size(0), device=x.device)
    return hidden[batch_idx, idx, :]


class RewardHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


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


def load_reward_heads(
    sources: List[str],
    device: torch.device,
) -> List[Tuple[str, RewardHead, Dict[str, torch.Tensor], Path, List[int], Dict[str, torch.Tensor]]]:
    base = Path(get_base_dir())
    mapping = {
        "rm": base / "rm_checkpoints" / "uniform" / "d20",
        "rm_density": base / "rm_checkpoints" / "density" / "d20",
    }
    heads = []
    for source in sources:
        if source not in mapping:
            raise ValueError(f"Unknown rm_source '{source}'")
        directory = mapping[source]
        ckpt_path = find_latest_checkpoint(directory)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        meta = ckpt.get("meta", {})
        feat_dim = int(meta.get("features_dim"))
        head = RewardHead(feat_dim).to(device)
        head.load_state_dict(ckpt["rm_head_state_dict"])
        head.eval()
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
        heads.append((source, head, meta, Path(ckpt_path), block_indices, blocks_state or {}))
    return heads


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

    heads = load_reward_heads(args.rm_sources.split(","), device)

    batch_size = args.batch_size
    max_len = args.max_len
    min_prompt = args.min_prompt

    summaries = {}
    for name, head, meta, ckpt_path, block_indices, blocks_state in heads:
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

        metric = Metrics(name)

        for start in tqdm(range(0, len(rows), batch_size), desc=f"{name} eval"):
            batch = rows[start:start + batch_size]
            pcs, prs = [], []
            sources, weights, bins = [], [], []

            for row in batch:
                try:
                    prompt_tokens = render_prefix_for_completion(tokenizer, row.prefix)
                except ValueError:
                    prompt_tokens = render_prefix_for_completion(tokenizer, None)
                chosen_ids = tokenizer.encode(row.chosen)
                rejected_ids = tokenizer.encode(row.rejected)
                p1, c1 = truncate_two(prompt_tokens, chosen_ids, max_len, min_prompt)
                p2, r2 = truncate_two(prompt_tokens, rejected_ids, max_len, min_prompt)
                pcs.append(p1 + c1)
                prs.append(p2 + r2)
                sources.append(row.src)
                weights.append(row.weight)
                bins.append(row.weight_bin)

            if not pcs:
                continue

            def pad_sequences(seq_list: List[List[int]]) -> torch.Tensor:
                seq_list = [seq[:max_len] for seq in seq_list]
                max_seq = max(len(seq) for seq in seq_list)
                padded = [seq + [pad_id] * (max_seq - len(seq)) for seq in seq_list]
                return torch.tensor(padded, dtype=torch.long, device=device)

            x_c = pad_sequences(pcs)
            x_r = pad_sequences(prs)

            with torch.no_grad():
                feat_c = extract_features(backbone, x_c, pad_id)
                feat_r = extract_features(backbone, x_r, pad_id)
                rc = head(feat_c)
                rr = head(feat_r)
                margin = (rc - rr).detach().cpu().numpy()

            correct = margin > 0
            for src, bin_label, ok, m, w in zip(sources, bins, correct, margin, weights):
                metric.add(src, bin_label, bool(ok), float(m), float(w))

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


