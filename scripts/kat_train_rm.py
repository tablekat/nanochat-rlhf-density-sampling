#!/usr/bin/env python3
"""
Reward Model (RM) training with inverse-density loss weighting.

Run as:
  python -m scripts.kat_train_rm

Or distributed:
  torchrun --nproc_per_node=8 -m scripts.kat_train_rm --rm_source rm

Design notes:
- Per-example loss weighting (inverse-density) instead of sampler-weighted
- Freezes SFT backbone; trains tiny linear RewardHead on last-token features
- DDP-friendly with rank-0 logging/saving
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, math, time, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import wandb
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, autodetect_device_type, get_base_dir, print_banner
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer

print_banner()

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

run = "dummy"  # wandb run name ("dummy" = no wandb logging)
device_type = ""  # cuda|cpu|mps (empty => autodetect)
rm_source = "rm"  # rm|rm_density
batch_size = 64
learning_rate = 5e-4
weight_decay = 0.0
max_steps = 1000
log_every = 25
eval_every = -1  # -1 = disable

# Data paths and processing
pairs_path = None  # Auto-resolved
density_weights_path = None  # Auto-resolved
weight_mode = "mean"  # mean|sum|none
weight_cap = None  # clip weights to max value
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
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rm", name=run, config=user_config)

# ═════════════════════════════════════════════════════════════════════════════
# Utilities & Data structures
# ═════════════════════════════════════════════════════════════════════════════

def md5_16(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

@dataclass
class PairRow:
    prompt: str
    chosen: str
    rejected: str
    weight: float

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
    
    def __len__(self): 
        return len(self.rows)
    
    def __getitem__(self, idx: int) -> PairRow: 
        return self.rows[idx]

class RewardHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)

# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def load_density_mapping(prompts_path: Path, weights_path: Path) -> Dict[str, float]:
    """Build prompt_id -> inverse-density weight mapping."""
    ids: List[str] = []
    with prompts_path.open("r", encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)["id"])
    weights = np.load(weights_path)
    assert len(ids) == len(weights), "prompts_all.jsonl and density_weights.npy misaligned"
    return {pid: float(w) for pid, w in zip(ids, weights.tolist())}

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

def make_batch(rows: List[PairRow], tokenizer, max_len: int, min_prompt: int, device: torch.device, pad_id: int):
    """Prepare batch tensors with per-example weights."""
    pcs, prs, ccs, rrs = [], [], [], []
    weights = []
    
    for row in rows:
        p = tokenizer.encode(row.prompt)
        c = tokenizer.encode(row.chosen)
        r = tokenizer.encode(row.rejected)
        
        p1, c1 = truncate_two(p, c, max_len, min_prompt)
        p2, r2 = truncate_two(p, r, max_len, min_prompt)
        
        pc, pr = p1 + c1, p2 + r2
        lc = [-100]*len(p1) + c1
        lr = [-100]*len(p2) + r2
        
        pcs.append(pc); prs.append(pr)
        ccs.append(lc); rrs.append(lr)
        weights.append(row.weight)
    
    def pad_batch(xs):
        xs = [x[:max_len] for x in xs]
        return torch.tensor([x + [pad_id]*(max_len - len(x)) for x in xs], dtype=torch.long, device=device)
    
    def pad_labels(ls):
        ls = [l[:max_len] for l in ls]
        return torch.tensor([l + [-100]*(max_len - len(l)) for l in ls], dtype=torch.long, device=device)
    
    return (pad_batch(pcs), pad_labels(ccs), pad_batch(prs), pad_labels(rrs), 
            torch.tensor(weights, dtype=torch.float32, device=device))

@torch.no_grad()
def extract_features(backbone, x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Extract last non-pad HIDDEN STATE as features: [B, H].
    Assumes backbone(..., return_hidden_states=True) returns {'hidden_states': [B,T,H]}.
    """
    attn = (x != pad_id)
    out = backbone(x, return_hidden_states=True)
    H = out["hidden_states"]            # [B,T,H]
    idx = attn.long().sum(dim=1).clamp(min=1) - 1
    b = torch.arange(x.size(0), device=x.device)
    return H[b, idx, :]                 # [B,H]

def bt_loss(reward_ch: torch.Tensor, reward_rj: torch.Tensor) -> torch.Tensor:
    """Bradley–Terry pairwise loss."""
    return F.softplus(-(reward_ch - reward_rj))

def apply_weights(loss_per_ex: torch.Tensor, w: torch.Tensor, mode: str, cap: Optional[float]) -> torch.Tensor:
    """Apply per-example weight normalization."""
    if cap is not None:
        w = torch.clamp(w, max=cap)
    if mode == "mean":
        wn = w / (w.mean() + 1e-12)
    elif mode == "sum":
        wn = w * (w.numel() / (w.sum() + 1e-12))
    else:
        wn = w
    return (wn * loss_per_ex).mean()

# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

# Resolve paths and config
base = get_base_dir()
save_dir_mapping = {
    "rm": os.path.join(base, "rm_checkpoints", "uniform", "d20"),
    "rm_density": os.path.join(base, "rm_checkpoints", "density", "d20"),
}
save_dir = save_dir_mapping[rm_source]
density_aware = rm_source == "rm_density"

if pairs_path is None:
    pairs_path = os.path.join(base, "data", "pairs_all.jsonl")
if density_weights_path is None:
    density_weights_path = os.path.join(base, "data", "embeddings_offline", "density_weights.npy")

# Load backbone
print0(f"Loading SFT backbone...")
backbone, tokenizer, _ = load_model(source="sft", device=device, phase="eval")
backbone.eval()
for p in backbone.parameters():
    p.requires_grad_(False)

hidden_size = getattr(getattr(backbone, "config", None), "n_embd", None)
assert hidden_size is not None, "model.config.n_embd not found"

pad_id = tokenizer.encode_special("<|assistant_end|>")

# Load or build dataset
print0(f"Loading preference dataset from {pairs_path}...")
density = None
if density_aware:
    print0(f"Loading density weights from {density_weights_path}...")
    prompts_path = os.path.join(base, "data", "prompts_all.jsonl")
    density = load_density_mapping(Path(prompts_path), Path(density_weights_path))

ds = PairsDataset(Path(pairs_path), density)
print0(f"Dataset size: {len(ds)} pairs | density_aware={density_aware}")

# DataLoader with DistributedSampler
sampler = DistributedSampler(ds, shuffle=True) if ddp else None
dl = DataLoader(ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
                num_workers=2, pin_memory=True, drop_last=True)

# Reward head
print0(f"Building RewardHead with input_dim={hidden_size}...")
head = RewardHead(in_dim=hidden_size).to(device)
opt = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Mkdir
if master_process:
    os.makedirs(save_dir, exist_ok=True)

# Training loop
print0(f"Starting training: {max_steps} steps, batch_size={batch_size}")
step = 0
t0 = time.time()

while step < max_steps:
    if sampler is not None:
        sampler.set_epoch(step)
    
    for _rows in dl:
        step += 1
        if step > max_steps:
            break
        
        x_c, y_c, x_r, y_r, w = make_batch(_rows, tokenizer, max_len, min_prompt, device, pad_id)
        
        # Forward
        with torch.no_grad():
            fc = extract_features(backbone, x_c, pad_id)
            fr = extract_features(backbone, x_r, pad_id)
        
        rc = head(fc)
        rr = head(fr)
        loss_vec = bt_loss(rc, rr)
        loss = apply_weights(loss_vec, w, weight_mode, weight_cap)
        
        # Backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        
        # Logging
        if step % log_every == 0:
            with torch.no_grad():
                margin = (rc - rr).mean().item()
                lw_mean = loss_vec.mean().item()
                w_mean = w.mean().item()
            
            print0(f"step {step:06d}/{max_steps:06d} | loss {loss.item():.4f} (ex {lw_mean:.4f}) | "
                  f"margin {margin:+.4f} | w_mean {w_mean:.4g}")
            
            wandb_run.log({
                "step": step,
                "loss": loss.item(),
                "loss_ex_mean": lw_mean,
                "reward_margin": margin,
                "weight_mean": w_mean,
            })

# Checkpoint
if master_process:
    ckpt = {
        "rm_head_state_dict": head.state_dict(),
        "meta": {
            "features_dim": hidden_size,  # <-- matches hidden states
            "weight_mode": weight_mode,
            "weight_cap": weight_cap,
            "density_aware": density_aware,
            "pad_id": int(pad_id),
            "max_len": max_len,
            "min_prompt": min_prompt,
        }
    }
    out_path = Path(save_dir) / f"model_{int(time.time())}.pt"
    torch.save(ckpt, out_path)
    print0(f"✓ Saved RM head to {out_path}")

wandb_run.finish()
compute_cleanup()
