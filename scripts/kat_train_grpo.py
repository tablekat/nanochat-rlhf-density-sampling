#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training over offline preference pairs.

Run as:
  python -m scripts.kat_train_grpo

Or distributed:
  torchrun --nproc_per_node=8 -m scripts.kat_train_grpo --rm_source rm --grpo_source grpo

Design notes:
- Response-only logprob accounting
- Reference model KL penalty (reverse KL)
- Adaptive target-KL controller for stable updates
- Frozen backbone features + RM head for reward computation
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, time, math, hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import wandb
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, autodetect_device_type, get_base_dir, print_banner
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer

from scripts.kat_utils import render_prefix_for_completion

print_banner()

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

run = "dummy"  # wandb run name ("dummy" = no wandb logging)
device_type = ""  # cuda|cpu|mps (empty => autodetect)
rm_source = "rm"  # rm|rm_density
grpo_source = "grpo"  # grpo|grpo_density
batch_size = 32
learning_rate = 1e-5
weight_decay = 0.0
grad_clip = 1.0
beta = 0.01  # initial KL weight
target_kl = 6.0  # target per-sample KL
beta_gain = 0.05  # KL controller update speed
std_adv = False  # standardize advantage
max_steps = 5000
log_every = 25
eval_every = -1  # -1 = disable

# Data
pairs_path = None  # Auto-resolved
max_len = 512
min_prompt = 128

# Tokenizer
tokenizer_path = "tokenizer.model"
hf_fallback = None

# Config override via CLI
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# ═════════════════════════════════════════════════════════════════════════════
# Distributed setup
# ═════════════════════════════════════════════════════════════════════════════

device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# WandB logging
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-grpo", name=run, config=user_config)

# ═════════════════════════════════════════════════════════════════════════════
# Utilities & Data structures
# ═════════════════════════════════════════════════════════════════════════════

def md5_16(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

class PairRow:
    __slots__ = ("prefix", "chosen", "rejected")
    def __init__(self, d: dict):
        self.prefix = d.get("prefix", {"messages": []})  # NEW: Full conversation object
        self.chosen = d["chosen"]
        self.rejected = d["rejected"]

class Pairs(Dataset):
    def __init__(self, path: Path):
        xs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                xs.append(PairRow(json.loads(line)))
        self.xs = xs
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, i):
        return self.xs[i]

class RewardHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=True)
    
    def forward(self, x):
        return self.fc(x).squeeze(-1)

# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def truncate_two(p: List[int], r: List[int], max_len: int, min_prompt: int):
    """Trim prompt from left, response from right."""
    if len(p) + len(r) <= max_len:
        return p, r
    resp_budget = max_len - min(len(p), min_prompt)
    resp_budget = max(resp_budget, 1)
    r = r[:resp_budget]
    over = (len(p) + len(r)) - max_len
    if over > 0:
        p = p[over:]
    return p, r

def collate(rows: List[PairRow], tokenizer, max_len: int, min_prompt: int, device):
    """Collate preference pairs into batch tensors."""
    pcs, prs, ccs, rrs = [], [], [], []
    
    for r in rows:
        # NEW: Handle prefix conversation object using render_for_completion
        try:
            p = render_prefix_for_completion(tokenizer, r.prefix)
        except ValueError:
            p = render_prefix_for_completion(tokenizer, None)
        
        c = tokenizer.encode(r.chosen)
        j = tokenizer.encode(r.rejected)
        
        p1, c1 = truncate_two(p, c, max_len, min_prompt)
        p2, j2 = truncate_two(p, j, max_len, min_prompt)
        
        pc = p1 + c1
        pj = p2 + j2
        lc = [-100]*len(p1) + c1
        lj = [-100]*len(p2) + j2
        
        pcs.append(pc)
        prs.append(pj)
        ccs.append(lc)
        rrs.append(lj)
    
    pad_id = tokenizer.encode_special("<|assistant_end|>")
    
    def pad_to(xs):
        xs = [x[:max_len] for x in xs]
        return torch.tensor([x + [pad_id]*(max_len - len(x)) for x in xs], dtype=torch.long, device=device)
    
    def pad_lab(ls):
        ls = [l[:max_len] for l in ls]
        return torch.tensor([l + [-100]*(max_len - len(l)) for l in ls], dtype=torch.long, device=device)
    
    return pad_to(pcs), pad_lab(ccs), pad_to(prs), pad_lab(rrs)

@torch.no_grad()
def last_features(backbone, x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Extract last non-pad hidden state [B,H] as RM features."""
    attn = (x != pad_id)
    out = backbone(x, return_hidden_states=True)
    H = out["hidden_states"]            # [B,T,H]
    idx = attn.long().sum(dim=1).clamp(min=1) - 1
    b = torch.arange(x.size(0), device=x.device)
    return H[b, idx, :]                 # [B,H]

def sum_logprobs(model, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log-probs over response tokens (teacher forcing)."""
    logits = model(x)
    logp = logits.log_softmax(dim=-1)
    tgt = labels[:, 1:].contiguous()
    logp = logp[:, :-1].contiguous()
    mask = (tgt != -100)
    gathered = logp.gather(2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=1)

def sum_kl(policy, reference, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """True KL(policy || reference) over response tokens (sum over vocab)."""
    with torch.no_grad():
        ref_logits = reference(x)
    pol_logits = policy(x)
    logp = F.log_softmax(pol_logits[:, :-1, :], dim=-1)  # [B,T-1,V]
    logq = F.log_softmax(ref_logits[:, :-1, :], dim=-1)  # [B,T-1,V]
    p = logp.exp()
    kl_tok = (p * (logp - logq)).sum(dim=-1)             # [B,T-1]
    resp_mask = (labels[:, 1:] != -100).float()          # [B,T-1]
    return (kl_tok * resp_mask).sum(dim=1)               # [B]

# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

# Resolve paths
base = get_base_dir()
rm_ckpt_mapping = {
    "rm": os.path.join(base, "rm_checkpoints", "uniform", "d20"),
    "rm_density": os.path.join(base, "rm_checkpoints", "density", "d20"),
}
grpo_out_mapping = {
    "grpo": os.path.join(base, "grpo_checkpoints", "uniform", "d20"),
    "grpo_density": os.path.join(base, "grpo_checkpoints", "density", "d20"),
}

rm_ckpt_dir = rm_ckpt_mapping[rm_source]
save_dir = grpo_out_mapping[grpo_source]

if pairs_path is None:
    pairs_path = os.path.join(base, "data", "pairs_all.jsonl")

# Load models
print0(f"Loading models...")
policy,   _, _ = load_model(source="sft", device=device, phase="train")
reference, _, _ = load_model(source="sft", device=device, phase="eval")
reference.eval()
for p in reference.parameters():
    p.requires_grad_(False)

# RM head
print0(f"Loading RM head from {rm_ckpt_dir}...")
rm_ckpt_files = glob.glob(os.path.join(rm_ckpt_dir, "model_*.pt"))
if not rm_ckpt_files:
    raise FileNotFoundError(f"No RM checkpoint found in {rm_ckpt_dir}")
rm_head_path = max(rm_ckpt_files, key=lambda x: int(Path(x).stem.split("_")[1]))
print0(f"Using RM checkpoint: {rm_head_path}")

rm = torch.load(rm_head_path, map_location="cpu")
head = RewardHead(in_dim=rm["meta"]["features_dim"]).to(device)
head.load_state_dict(rm["rm_head_state_dict"])
for p_ in head.parameters():
    p_.requires_grad_(False)

# If RM training also fine-tuned transformer blocks, load them into policy/reference
blocks_state = rm.get("backbone_blocks_state_dict")
if blocks_state is not None:
    indices = rm.get("meta", {}).get("backbone_block_indices")
    if indices is None:
        # Backwards compatibility with single-block checkpoints
        block_idx = rm.get("meta", {}).get("backbone_block_index", -1)
        indices = [block_idx]
        blocks_state = {str(block_idx): rm.get("backbone_block_state_dict")}
    print0(f"Applying reward-model backbone blocks {indices} to policy/reference")
    for idx in indices:
        state = blocks_state.get(str(idx))
        if state is None:
            continue
        policy.transformer.h[idx].load_state_dict(state)
        reference.transformer.h[idx].load_state_dict(state)

# Dataset
print0(f"Loading preference dataset...")
tokenizer = get_tokenizer()
ds = Pairs(Path(pairs_path))
pad_id = tokenizer.encode_special("<|assistant_end|>")

sampler = DistributedSampler(ds, shuffle=True) if ddp else None
dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=(sampler is None),
    sampler=sampler,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    collate_fn=lambda batch: batch,
)

# Optimizer and KL controller
opt = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
kl_beta = beta

# Mkdir
if master_process:
    os.makedirs(save_dir, exist_ok=True)

# Training loop
print0(f"Starting GRPO training: {max_steps} steps, batch_size={batch_size}, rm_source={rm_source}, grpo_source={grpo_source}")
step = 0
t0 = time.time()

while step < max_steps:
    if sampler is not None:
        sampler.set_epoch(step)
    
    for rows in dl:
        step += 1
        if step > max_steps:
            break
        
        x_c, y_c, x_r, y_r = collate(rows, tokenizer, max_len, min_prompt, device)
        
        # Log-probs (response-only)
        lp_c = sum_logprobs(policy, x_c, y_c)
        lp_r = sum_logprobs(policy, x_r, y_r)
        
        # KL sums
        kl_c = sum_kl(policy, reference, x_c, y_c)
        kl_r = sum_kl(policy, reference, x_r, y_r)
        
        # Rewards (frozen backbone + RM head)
        with torch.no_grad():
            fc = last_features(reference, x_c, pad_id)
            fr = last_features(reference, x_r, pad_id)
            rc = head(fc)
            rr = head(fr)
        
        # Advantage and loss
        dr = (rc - rr)
        if std_adv:
            dr = (dr - dr.mean()) / (dr.std(unbiased=False) + 1e-6)
        
        dkl = (kl_c - kl_r)
        A = dr - kl_beta * dkl
        
        loss = -(A.detach() * (lp_c - lp_r)).mean()
        
        # Backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        opt.step()
        
        # KL controller: target absolute KL (chosen) for stability
        with torch.no_grad():
            kl_now = kl_c.mean().item()
            kl_beta *= math.exp(beta_gain * (kl_now - target_kl))
        
        # Logging
        if step % log_every == 0:
            with torch.no_grad():
                stats = dict(
                    loss=loss.item(),
                    lp_margin=(lp_c - lp_r).mean().item(),
                    r_margin=dr.mean().item(),
                    kl_margin=dkl.mean().item(),
                    beta=kl_beta,
                )
            
            print0(f"step {step:06d}/{max_steps:06d} | "
                  f"loss {stats['loss']:.4f} | "
                  f"dlogp {stats['lp_margin']:+.4f} | "
                  f"dr {stats['r_margin']:+.4f} | "
                  f"dkl {stats['kl_margin']:+.4f} | "
                  f"beta {stats['beta']:.4g}")
            
            wandb_run.log({
                "step": step,
                "loss": stats["loss"],
                "logprob_margin": stats["lp_margin"],
                "reward_margin": stats["r_margin"],
                "kl_margin": stats["kl_margin"],
                "beta": stats["beta"],
            })

# Checkpoint
if master_process:
    out_path = Path(save_dir) / f"model_{int(time.time())}.pt"
    torch.save({"policy_state_dict": policy.state_dict()}, out_path)
    print0(f"✓ Saved GRPO policy to {out_path}")

wandb_run.finish()
compute_cleanup()
