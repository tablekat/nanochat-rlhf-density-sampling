#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward Model (RM) training — loss-weighted (inverse-density) instead of sampler-weighted.

Design notes
- Keeps "nanochat-y" single-file style: argparse, simple dataset, clear training loop.
- Applies prompt inverse-density as a PER-EXAMPLE LOSS WEIGHT (not WeightedRandomSampler).
- Freezes the SFT backbone; trains a tiny linear RewardHead on the last-token features.
- Truncation prioritizes response tokens while preserving a minimum prompt context.
- DDP-friendly: rank-0 logging/saving; optional DistributedSampler.

Assumptions
- pairs_all.jsonl lines: {"id", "prompt", "chosen", "rejected", "src"}
- prompts_all.jsonl lines: {"id": md5_16(prompt), "prompt": "..."} (for joining density)
- density_weights.npy aligns with prompts_all.jsonl order.
- Tokenizer provides .encode(str) -> List[int].
- Backbone "policy" returns logits (B,T,V); if hidden states are available, see TODO near extract_features().

If your tokenizer/model names differ, adjust the imports in `build_tokenizer()` and `build_backbone()`.
"""

from __future__ import annotations
import os, json, math, time, argparse, hashlib, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed import is_initialized as dist_is_init, get_rank as dist_rank, get_world_size as dist_ws, init_process_group, barrier
from tqdm import tqdm


# -------------------
# Paths & utilities
# -------------------

def get_base_dir() -> Path:
    # mirrors the doc you wrote; overridable via env
    return Path(os.environ.get("NANOCHAT_HOME", "~/.cache/nanochat")).expanduser()

def md5_16(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

def is_main_process() -> bool:
    return (not dist_is_init()) or (dist_rank() == 0)

def setup_ddp_if_needed():
    # Initialize DDP if torchrun used
    if "RANK" in os.environ and not dist_is_init():
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


# -------------------
# Tokenizer
# -------------------

class TokenizerWrapper:
    """
    Thin wrapper so we don't depend on any single tokenizer implementation.
    Tries nanochat RustBPE first; falls back to HF if requested.
    """
    def __init__(self, tokenizer_path: Optional[str], hf_fallback: Optional[str]):
        self.impl = None
        self.kind = None
        if tokenizer_path is not None:
            try:
                # nanochat/rustbpe build (most likely in your repo)
                from rustbpe import Tokenizer as RustTokenizer  # type: ignore
                self.impl = RustTokenizer.from_file(tokenizer_path)
                self.kind = "rustbpe"
            except Exception as e:
                if hf_fallback is None:
                    raise RuntimeError(
                        f"Failed to load rustbpe tokenizer at {tokenizer_path}. "
                        f"Install/compile rustbpe or pass --hf_fallback."
                    ) from e

        if self.impl is None:
            # Use HF fallback
            from transformers import AutoTokenizer
            self.impl = AutoTokenizer.from_pretrained(hf_fallback, use_fast=True)
            # make sure pad exists for collate convenience
            if self.impl.pad_token_id is None:
                self.impl.pad_token = self.impl.eos_token or "<|pad|>"
            self.kind = "hf"

    @property
    def pad_id(self) -> int:
        if self.kind == "rustbpe":
            # RustBPE in nanochat may not define a special pad token.
            # We rely on attention masks to ignore pads; use 0 as inert pad id.
            return 0
        else:
            return int(self.impl.pad_token_id)

    def encode(self, s: str) -> List[int]:
        if self.kind == "rustbpe":
            return self.impl.encode(s)
        else:
            return self.impl.encode(s, add_special_tokens=False)

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]


# -------------------
# Backbone model
# -------------------

def build_backbone(device: torch.device, dtype: torch.dtype):
    """
    Create the frozen SFT backbone used to extract features for RM.

    By default we try to import the nanochat Transformer checkpoint loader.
    If your loader utility has a different name, adjust here.

    Expected forward: logits = model(input_ids) -> (B,T,V)
    """
    # Try to import a canonical nanochat model builder if available
    model = None
    try:
        # THE FOLLOWING IS A SOFT HOOK: adapt to your repo if needed.
        # E.g., from nanochat.engine import build_sft_backbone
        # model = build_sft_backbone().to(device=device, dtype=dtype).eval()
        from nanochat.tasks.infer import load_model  # common in nanochat forks
        model = load_model("sft")  # or "base" depending on your tree
        model = model.to(device=device, dtype=dtype).eval()
    except Exception:
        pass

    if model is None:
        raise RuntimeError(
            "Couldn't import a nanochat backbone. Replace build_backbone() body with "
            "your local loader that returns a (B,T,V) logits model in eval mode."
        )

    for p in model.parameters():
        p.requires_grad_(False)
    return model


class RewardHead(nn.Module):
    """
    Minimal linear head mapping last-token features to a scalar.
    Default features are the final-token LOGITS vector (V-dim), which we always have.
    If you prefer hidden states, see extract_features() TODO.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> (B,)
        return self.fc(x).squeeze(-1)


# -------------------
# Data
# -------------------

@dataclass
class PairRow:
    prompt: str
    chosen: str
    rejected: str
    weight: float  # loss weight

class PairsDataset(Dataset):
    def __init__(self, pairs_path: Path, density: Optional[Dict[str, float]]):
        self.rows: List[PairRow] = []
        with pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                w = 1.0
                if density is not None:
                    w = float(density.get(md5_16(ex["prompt"]), 1.0))
                self.rows.append(PairRow(
                    prompt=ex["prompt"], chosen=ex["chosen"],
                    rejected=ex["rejected"], weight=w
                ))

    def __len__(self): return len(self.rows)
    def __getitem__(self, idx: int) -> PairRow: return self.rows[idx]


def load_density_mapping(
    prompts_path: Path, weights_path: Path
) -> Dict[str, float]:
    """
    Build: prompt_md5_16 -> inverse-density weight
    """
    # read prompts_all.jsonl to preserve order
    ids: List[str] = []
    with prompts_path.open("r", encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)["id"])
    weights = np.load(weights_path)
    assert len(ids) == len(weights), "prompts_all.jsonl and density_weights.npy misaligned"
    return {pid: float(w) for pid, w in zip(ids, weights.tolist())}


def truncate_two(
    tok_ids_prompt: List[int],
    tok_ids_resp: List[int],
    max_len: int,
    min_prompt: int,
) -> Tuple[List[int], List[int]]:
    """
    Trim from left of prompt and right of response to fit max_len.
    Reserve `min_prompt` tokens for prompt if possible, and let response use the rest.
    """
    p, r = tok_ids_prompt, tok_ids_resp
    # If already fit:
    if len(p) + len(r) <= max_len:
        return p, r

    # budget
    resp_budget = max_len - min(len(p), min_prompt)
    resp_budget = max(resp_budget, 1)

    # trim response from the right
    r = r[:max(0, resp_budget)]

    # trim prompt from left if still too long
    over = (len(p) + len(r)) - max_len
    if over > 0:
        p = p[over:]

    return p, r


def make_batch(
    tok: TokenizerWrapper,
    rows: List[PairRow],
    max_len: int,
    min_prompt: int,
    device: torch.device,
):
    """
    Build tensors for (prompt+chosen) and (prompt+rejected).
    Labels mask out the prompt tokens (-100) so losses/logprobs are over the response only.
    """
    B = len(rows)
    enc_p = tok.batch_encode([r.prompt for r in rows])
    enc_c = tok.batch_encode([r.chosen for r in rows])
    enc_r = tok.batch_encode([r.rejected for r in rows])

    pcs, prs = [], []
    ccs, c_labels = [], []
    rrs, r_labels = [], []
    weights = torch.tensor([r.weight for r in rows], dtype=torch.float32, device=device)

    for p_ids, c_ids, r_ids in zip(enc_p, enc_c, enc_r):
        p_trim, c_trim = truncate_two(p_ids, c_ids, max_len, min_prompt)
        _,      r_trim = truncate_two(p_ids, r_ids, max_len, min_prompt)
        # concat
        pc = p_trim + c_trim
        pr = p_trim + r_trim
        pcs.append(pc); prs.append(pr)
        # labels mask prompt
        lc = [-100]*len(p_trim) + c_trim
        lr = [-100]*len(p_trim) + r_trim
        ccs.append(lc); rrs.append(lr)

    # pad to max length in batch
    maxT_c = max(len(x) for x in pcs)
    maxT_r = max(len(x) for x in prs)

    def pad_batch(xs, pad_id):
        pad_x = [x + [pad_id]*(max_len - len(x)) if len(x) < max_len else x for x in xs]
        return torch.tensor(pad_x, dtype=torch.long, device=device)

    def pad_labels(ls):
        pad_l = [l + [-100]*(max_len - len(l)) if len(l) < max_len else l for l in ls]
        return torch.tensor(pad_l, dtype=torch.long, device=device)

    # enforce hard cap max_len
    pcs = [x[:max_len] for x in pcs]
    prs = [x[:max_len] for x in prs]
    ccs = [x[:max_len] for x in ccs]
    rrs = [x[:max_len] for x in rrs]

    x_c = pad_batch(pcs, tok.pad_id)
    x_r = pad_batch(prs, tok.pad_id)
    y_c = pad_labels(ccs)
    y_r = pad_labels(rrs)

    return x_c, y_c, x_r, y_r, weights


# -------------------
# RM loss
# -------------------

@torch.no_grad()
def extract_features(backbone, x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Returns a (B, D) feature per sequence. Default uses last-token LOGITS vector.

    If your backbone can return hidden states, you may switch to hidden features:
    - Example: logits, h = backbone(x, return_hidden=True); take last non-pad row from `h`.
    - Then change `features_dim = logits.size(-1)` to `h.size(-1)` below.
    """
    logits = backbone(x)  # (B,T,V)
    # compute last non-pad index per row
    with torch.no_grad():
        mask = (x != pad_id).to(x.dtype)  # (B,T)
        lengths = mask.sum(dim=1).clamp(min=1)  # (B,)
        idx = (lengths - 1).long()  # last token index
        out = logits[torch.arange(x.size(0), device=x.device), idx]  # (B,V)
    return out  # features dim == V


def bt_loss(reward_ch: torch.Tensor, reward_rj: torch.Tensor) -> torch.Tensor:
    """Bradley–Terry pairwise loss per example: -logsigmoid(r_c - r_r)."""
    return F.softplus(-(reward_ch - reward_rj))  # numerically stable


def apply_weights(loss_per_ex: torch.Tensor, w: torch.Tensor, mode: str, cap: Optional[float]) -> torch.Tensor:
    """
    Weight the per-example losses:
    - mode='mean': divide by mean(w) => keeps LR scale stable when weights vary
    - mode='sum': normalize w to sum == batch_size
    - mode='none': use raw w (not recommended)
    - cap: optional clip of w to [0, cap] to reduce variance
    """
    if cap is not None:
        w = torch.clamp(w, max=cap)
    if mode == "mean":
        wn = w / (w.mean() + 1e-12)
    elif mode == "sum":
        wn = w * (w.numel() / (w.sum() + 1e-12))
    else:
        wn = w
    return (wn * loss_per_ex).mean()


# -------------------
# Train
# -------------------

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    base = get_base_dir()
    p.add_argument("--pairs_path", type=str, default=str(base / "data" / "pairs_all.jsonl"))
    p.add_argument("--prompts_path", type=str, default=str(base / "data" / "prompts_all.jsonl"))
    p.add_argument("--density_weights_path", type=str, default=str(base / "data" / "embeddings_offline" / "density_weights.npy"))
    p.add_argument("--density_aware", action="store_true", help="if set, load density weights and apply as loss weights")
    p.add_argument("--weight_mode", type=str, choices=["mean","sum","none"], default="mean",
                   help="how to normalize per-example weights inside the loss")
    p.add_argument("--weight_cap", type=float, default=None, help="optional upper clip for per-example weights")
    p.add_argument("--tokenizer_path", type=str, default="tokenizer.model", help="nanochat RustBPE tokenizer file")
    p.add_argument("--hf_fallback", type=str, default=None, help="HF tokenizer name (e.g. 'gpt2') if RustBPE not available")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32","bfloat16","float16"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--min_prompt", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--save_dir", type=str, default=str(base / "rm_checkpoints" / "d20"))
    args = p.parse_args()

    setup_ddp_if_needed()
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    # tokenizer
    tok = TokenizerWrapper(args.tokenizer_path, args.hf_fallback)

    # density map (prompt_id -> weight)
    density = None
    if args.density_aware:
        density = load_density_mapping(Path(args.prompts_path), Path(args.density_weights_path))

    # dataset & loader
    ds = PairsDataset(Path(args.pairs_path), density)
    sampler = DistributedSampler(ds, shuffle=True) if dist_is_init() else None
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                    num_workers=2, pin_memory=True, drop_last=True)

    # backbone (frozen)
    backbone = build_backbone(device, dtype)
    # build a dummy batch to determine feature dim
    tmp_rows = [ds[i] for i in range(min(len(ds), 2))]
    x_c, y_c, _, _, _ = make_batch(tok, tmp_rows, args.max_len, args.min_prompt, device)
    with torch.no_grad():
        feats = extract_features(backbone, x_c, tok.pad_id)  # (B, Dfeat)
    head = RewardHead(in_dim=feats.size(-1)).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)

    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"RM training: {len(ds)} pairs | density_aware={args.density_aware} weight_mode={args.weight_mode}")

    step = 0
    t0 = time.time()
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for _rows in dl:
            step += 1
            if step > args.max_steps:
                break

            x_c, y_c, x_r, y_r, w = make_batch(tok, _rows, args.max_len, args.min_prompt, device)
            # features
            with torch.no_grad():
                fc = extract_features(backbone, x_c, tok.pad_id)  # (B,Df)
                fr = extract_features(backbone, x_r, tok.pad_id)  # (B,Df)
            # reward
            rc = head(fc.to(dtype))
            rr = head(fr.to(dtype))
            # loss
            loss_vec = bt_loss(rc, rr)  # (B,)
            loss = apply_weights(loss_vec, w, args.weight_mode, args.weight_cap)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            if is_main_process() and (step % args.log_every == 0):
                dt = time.time() - t0
                with torch.no_grad():
                    margin = (rc - rr).mean().item()
                    lw_mean = loss_vec.mean().item()
                    w_mean = w.mean().item()
                print(f"step {step:06d} | loss {loss.item():.4f} (ex {lw_mean:.4f}) | "
                      f"margin {margin:+.4f} | w_mean {w_mean:.4g} | dt {dt:.1f}s")
                t0 = time.time()

    # save (rank 0 only)
    if is_main_process():
        ckpt = {
            "rm_head_state_dict": head.state_dict(),
            "meta": {
                "features_dim": feats.size(-1),
                "weight_mode": args.weight_mode,
                "weight_cap": args.weight_cap,
                "density_aware": args.density_aware,
                "tokenizer_kind": tok.kind,
                "pad_id": int(tok.pad_id),
                "max_len": args.max_len,
                "min_prompt": args.min_prompt,
                "dtype": args.dtype,
            }
        }
        out_path = Path(args.save_dir) / f"model_{int(time.time())}.pt"
        torch.save(ckpt, out_path)
        print(f"Saved RM head to {out_path}")

    if dist_is_init():
        barrier()


if __name__ == "__main__":
    main()
