#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO training over offline preference pairs with:
- Response-only logprob accounting
- Reference model KL penalty (reverse KL)
- Adaptive target-KL controller for stable updates
- Optional advantage standardization

Sampling is uniform. Rewards come from frozen backbone features + RM head you trained in kat_train_rm.py.
"""

from __future__ import annotations
import os, json, time, argparse, math, hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed import is_initialized as dist_is_init, get_rank as dist_rank, init_process_group, barrier
from tqdm import tqdm


# -------------
# utils
# -------------

def md5_16(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

def is_main() -> bool:
    return (not dist_is_init()) or (dist_rank() == 0)

def setup_ddp_if_needed():
    if "RANK" in os.environ and not dist_is_init():
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def get_base_dir() -> Path:
    return Path(os.environ.get("NANOCHAT_HOME", "~/.cache/nanochat")).expanduser()


# -------------
# tokenizer
# -------------

class TokenizerWrapper:
    def __init__(self, tokenizer_path: Optional[str], hf_fallback: Optional[str]):
        self.kind = None
        self.impl = None
        if tokenizer_path is not None:
            try:
                from rustbpe import Tokenizer as RustTokenizer  # type: ignore
                self.impl = RustTokenizer.from_file(tokenizer_path)
                self.kind = "rustbpe"
            except Exception:
                pass
        if self.impl is None:
            from transformers import AutoTokenizer
            self.impl = AutoTokenizer.from_pretrained(hf_fallback or "gpt2", use_fast=True)
            if self.impl.pad_token_id is None:
                self.impl.pad_token = self.impl.eos_token or "<|pad|>"
            self.kind = "hf"

    @property
    def pad_id(self) -> int:
        if self.kind == "rustbpe":
            return 0
        return int(self.impl.pad_token_id)

    def encode(self, s: str) -> List[int]:
        if self.kind == "rustbpe":
            return self.impl.encode(s)
        return self.impl.encode(s, add_special_tokens=False)


# -------------
# models
# -------------

def load_backbone(tag: str, device, dtype):
    """
    Build the policy or reference model from your nanochat checkpoint utility.
    Replace body with your local loader if needed. Must return logits (B,T,V).
    """
    model = None
    try:
        from nanochat.tasks.infer import load_model
        model = load_model(tag)  # e.g., "sft"
        model = model.to(device=device, dtype=dtype).eval()
    except Exception:
        raise RuntimeError("Replace load_backbone() with your local model loader.")
    return model


class RewardHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=True)
    def forward(self, x): return self.fc(x).squeeze(-1)


@torch.no_grad()
def last_features(backbone, x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Return last-token LOGITS vector (B,V) as RM features (always available)."""
    logits = backbone(x)
    mask = (x != pad_id).to(x.dtype)
    idx = (mask.sum(dim=1).clamp(min=1) - 1).long()
    return logits[torch.arange(x.size(0), device=x.device), idx]


def sum_logprobs(model, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Sum of log p(y_t | x_{<t}) over response tokens (labels != -100).
    Uses teacher forcing. Returns (B,)
    """
    logits = model(x)  # (B,T,V)
    logp = logits.log_softmax(dim=-1)
    # shift for next-token prediction
    tgt = labels[:, 1:].contiguous()           # (B,T-1)
    logp = logp[:, :-1].contiguous()           # (B,T-1,V)
    mask = (tgt != -100)
    gathered = logp.gather(2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=1)


def sum_kl(policy, reference, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Sum over response tokens of KL(policy || reference).
    """
    with torch.no_grad():
        ref_logits = reference(x)
        ref_logp = ref_logits.log_softmax(dim=-1)
    pol_logits = policy(x)
    pol_logp = pol_logits.log_softmax(dim=-1)

    tgt = labels[:, 1:].contiguous()
    pol_logp = pol_logp[:, :-1].contiguous()
    ref_logp = ref_logp[:, :-1].contiguous()
    mask = (tgt != -100)

    # KL = sum_i p_i (log p_i - log q_i) ; but we only need token logprob at target indices.
    # However in GRPO offline form with teacher forcing on the reference sequence tokens,
    # we use the token realized under reference labels; equivalent per-pair margin when differenced.
    # Here compute logprob margin at the *same* labels:
    kl_token = (pol_logp - ref_logp).gather(2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (kl_token * mask).sum(dim=1)


# -------------
# data
# -------------

class PairRow:
    __slots__ = ("prompt","chosen","rejected")
    def __init__(self, d: dict):
        self.prompt = d["prompt"]; self.chosen = d["chosen"]; self.rejected = d["rejected"]

class Pairs(Dataset):
    def __init__(self, path: Path):
        xs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f: xs.append(PairRow(json.loads(line)))
        self.xs = xs
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return self.xs[i]

def truncate_two(p: List[int], r: List[int], max_len: int, min_prompt: int):
    if len(p) + len(r) <= max_len: return p, r
    resp_budget = max_len - min(len(p), min_prompt)
    resp_budget = max(resp_budget, 1)
    r = r[:resp_budget]
    over = (len(p) + len(r)) - max_len
    if over > 0: p = p[over:]
    return p, r

def collate(rows: List[PairRow], tok: TokenizerWrapper, max_len: int, min_prompt: int, device):
    pcs, prs, ccs, rrs = [], [], [], []
    for r in rows:
        p = tok.encode(r.prompt)
        c = tok.encode(r.chosen)
        j = tok.encode(r.rejected)
        p1, c1 = truncate_two(p, c, max_len, min_prompt)
        p2, j2 = truncate_two(p, j, max_len, min_prompt)
        pc = p1 + c1; pj = p2 + j2
        lc = [-100]*len(p1) + c1; lj = [-100]*len(p2) + j2
        pcs.append(pc); prs.append(pj); ccs.append(lc); rrs.append(lj)

    def pad_to(xs, pad_id):
        xs = [x[:max_len] for x in xs]
        return torch.tensor([x + [pad_id]*(max_len - len(x)) for x in xs], dtype=torch.long, device=device)
    def pad_lab(ls):
        ls = [l[:max_len] for l in ls]
        return torch.tensor([l + [-100]*(max_len - len(l)) for l in ls], dtype=torch.long, device=device)

    x_c = pad_to(pcs, tok.pad_id); x_r = pad_to(prs, tok.pad_id)
    y_c = pad_lab(ccs); y_r = pad_lab(rrs)
    return x_c, y_c, x_r, y_r


# -------------
# train
# -------------

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    base = get_base_dir()
    p.add_argument("--pairs_path", type=str, default=str(base / "data" / "pairs_all.jsonl"))
    p.add_argument("--rm_head_ckpt", type=str, required=True, help="path saved by kat_train_rm.py")
    p.add_argument("--tokenizer_path", type=str, default="tokenizer.model")
    p.add_argument("--hf_fallback", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, choices=["float32","bfloat16","float16"], default="bfloat16")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--min_prompt", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.01, help="initial KL weight")
    p.add_argument("--target_kl", type=float, default=6.0, help="target per-sample KL (chosen minus rejected)")
    p.add_argument("--beta_gain", type=float, default=0.05, help="update speed of KL controller")
    p.add_argument("--std_adv", action="store_true", help="standardize (rc-rr) per batch to stabilize")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--save_dir", type=str, default=str(base / "grpo_checkpoints" / "d20"))
    args = p.parse_args()

    setup_ddp_if_needed()
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    tok = TokenizerWrapper(args.tokenizer_path, args.hf_fallback)

    # data
    ds = Pairs(Path(args.pairs_path))
    sampler = DistributedSampler(ds, shuffle=True) if dist_is_init() else None
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                    num_workers=2, pin_memory=True, drop_last=True)

    # models
    policy = load_backbone("sft", device, dtype).train()   # policy gets updated
    reference = load_backbone("sft", device, dtype).eval() # ref frozen
    for p in reference.parameters(): p.requires_grad_(False)

    # RM head
    rm = torch.load(args.rm_head_ckpt, map_location="cpu")
    head = RewardHead(in_dim=rm["meta"]["features_dim"]).to(device=device, dtype=dtype)
    head.load_state_dict(rm["rm_head_state_dict"])
    for p_ in head.parameters(): p_.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.wd)

    # KL controller
    beta = args.beta

    if is_main():
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"GRPO: {len(ds)} pairs | target_kl={args.target_kl} | std_adv={args.std_adv}")

    step = 0
    t0 = time.time()
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for rows in dl:
            step += 1
            if step > args.max_steps: break

            x_c, y_c, x_r, y_r = collate(rows, tok, args.max_len, args.min_prompt, device)

            # logprob sums (response-only)
            lp_c = sum_logprobs(policy, x_c, y_c)
            lp_r = sum_logprobs(policy, x_r, y_r)

            # KL sums w.r.t. reference
            kl_c = sum_kl(policy, reference, x_c, y_c)
            kl_r = sum_kl(policy, reference, x_r, y_r)

            # rewards via frozen backbone + RM head
            with torch.no_grad():
                fc = last_features(reference, x_c, tok.pad_id)  # frozen feats
                fr = last_features(reference, x_r, tok.pad_id)
                rc = head(fc.to(dtype))
                rr = head(fr.to(dtype))

            # advantage per pair
            dr = (rc - rr)  # reward margin
            if args.std_adv:
                dr = (dr - dr.mean()) / (dr.std(unbiased=False) + 1e-6)

            dkl = (kl_c - kl_r)
            A = dr - beta * dkl

            # GRPO loss (maximize margin in direction of A)
            loss = -(A.detach() * (lp_c - lp_r)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            opt.step()

            # KL controller (simple proportional control on per-sample mean)
            with torch.no_grad():
                kl_now = dkl.mean().item()
                beta *= math.exp(args.beta_gain * (kl_now - args.target_kl))

            if is_main() and (step % args.log_every == 0):
                with torch.no_grad():
                    stats = dict(
                        loss=loss.item(),
                        lp_margin=(lp_c - lp_r).mean().item(),
                        r_margin=dr.mean().item(),
                        kl_margin=dkl.mean().item(),
                        beta=beta,
                    )
                dt = time.time() - t0
                print(f"step {step:06d} | "
                      f"loss {stats['loss']:.4f} | "
                      f"dlogp {stats['lp_margin']:+.4f} | "
                      f"dr {stats['r_margin']:+.4f} | "
                      f"dkl {stats['kl_margin']:+.4f} | "
                      f"beta {stats['beta']:.4g} | dt {dt:.1f}s")
                t0 = time.time()

    if is_main():
        out_path = Path(args.save_dir) / f"model_{int(time.time())}.pt"
        # Save only policy weights (same convention as upstream, adjust if you have a manager)
        torch.save({"policy_state_dict": policy.state_dict()}, out_path)
        print(f"Saved GRPO policy to {out_path}")

    if dist_is_init():
        barrier()


if __name__ == "__main__":
    main()
