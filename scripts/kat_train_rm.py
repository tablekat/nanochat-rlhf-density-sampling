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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import wandb
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, autodetect_device_type, get_base_dir, print_banner
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer

from scripts.kat_utils import (
    ensure_prefix_dict,
    prefix_id_from_prefix,
    render_prefix_for_completion,
)

print_banner()

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

run = "dummy"  # wandb run name ("dummy" = no wandb logging)
device_type = ""  # cuda|cpu|mps (empty => autodetect)
rm_source = "rm"  # rm|rm_density
batch_size = 64
learning_rate = 5e-4
backbone_lr = 5e-5
weight_decay = 0.0
max_steps = 1000
log_every = 25
eval_every = 100  # steps between validation metrics (-1 = disable)
val_ratio = 0.05  # fraction of pairs routed to validation split
val_seed = 123
likert_token = "7"  # token whose logit scores the completion
likert_prompt = "\nRating (1-7):"  # appended prompt before scoring the logit

# Data paths and processing
pairs_path = None  # Auto-resolved
density_weights_path = None  # Auto-resolved
weight_mode = "mean"  # mean|sum|none
weight_cap = None  # clip weights to max value
max_len = 512 # TODO: try 4096
min_prompt = 128 # TODO: try 256

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

@dataclass
class PairRow:
    prefix: dict  # NEW: Full conversation object
    chosen: str
    rejected: str
    weight: float
    prefix_id: Optional[str] = None

def build_pair_row(ex: Dict, density: Optional[Dict[str, float]]) -> Optional[PairRow]:
    chosen = ex.get("chosen")
    rejected = ex.get("rejected")
    if not chosen or not rejected:
        return None

    prefix = ensure_prefix_dict(ex.get("prefix"))
    prefix_id = ex.get("prefix_id") or prefix_id_from_prefix(prefix)
    weight = 1.0
    if density is not None and prefix_id:
        weight = float(density.get(prefix_id, 1.0))

    return PairRow(
        prefix=prefix,
        chosen=chosen,
        rejected=rejected,
        weight=weight,
        prefix_id=prefix_id,
    )


class PairsDataset(Dataset):
    def __init__(self, pairs_path: Path, density: Optional[Dict[str, float]], rows: Optional[List[PairRow]] = None):
        if rows is not None:
            self.rows = rows
            return

        self.rows: List[PairRow] = []
        with pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row = build_pair_row(ex, density)
                if row is not None:
                    self.rows.append(row)
    
    def __len__(self): 
        return len(self.rows)
    
    def __getitem__(self, idx: int) -> PairRow: 
        return self.rows[idx]

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
    assert len(ids) == len(weights), "prefixes_all.jsonl and density_weights.npy misaligned"
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

def make_batch(
    rows: List[PairRow],
    tokenizer,
    max_len: int,
    min_prompt: int,
    device: torch.device,
    pad_id: int,
):
    """Prepare tensors for a minibatch of preference pairs."""

    chosen_token_ids: List[List[int]] = []      # prompt + chosen completion + rating prompt
    rejected_token_ids: List[List[int]] = []    # prompt + rejected completion + rating prompt
    example_weights: List[float] = []

    rating_prompt_len = len(tokenizer.encode(likert_prompt))
    effective_max_len = max(1, max_len - rating_prompt_len)

    for row in rows:
        try:
            prompt_ids = render_prefix_for_completion(tokenizer, row.prefix)
        except ValueError:
            prompt_ids = render_prefix_for_completion(tokenizer, None)

        chosen_ids = tokenizer.encode(row.chosen)
        rejected_ids = tokenizer.encode(row.rejected)

        prompt_chosen_ids, chosen_trimmed = truncate_two(prompt_ids, chosen_ids, effective_max_len, min_prompt)
        prompt_rejected_ids, rejected_trimmed = truncate_two(prompt_ids, rejected_ids, effective_max_len, min_prompt)

        chosen_sequence = prompt_chosen_ids + chosen_trimmed + tokenizer.encode(likert_prompt)
        rejected_sequence = prompt_rejected_ids + rejected_trimmed + tokenizer.encode(likert_prompt)

        assert len(chosen_sequence) <= max_len, "Chosen sequence exceeds max_len after rating prompt append"
        assert len(rejected_sequence) <= max_len, "Rejected sequence exceeds max_len after rating prompt append"

        chosen_token_ids.append(chosen_sequence)
        rejected_token_ids.append(rejected_sequence)
        example_weights.append(row.weight)

    def pad_sequences(seqs: List[List[int]]) -> torch.Tensor:
        seqs = [seq[:max_len] for seq in seqs]
        padded = [seq + [pad_id] * (max_len - len(seq)) for seq in seqs]
        return torch.tensor(padded, dtype=torch.long, device=device)

    return (
        pad_sequences(chosen_token_ids),
        pad_sequences(rejected_token_ids),
        torch.tensor(example_weights, dtype=torch.float32, device=device),
    )

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


def split_pair_rows(
    pairs_path: Path,
    density: Optional[Dict[str, float]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[PairRow], List[PairRow]]:
    if val_ratio <= 0.0:
        return [], []

    threshold = int(val_ratio * 1_000_000)
    threshold = max(0, min(threshold, 1_000_000))
    if threshold == 0:
        return [], []

    train_rows: List[PairRow] = []
    val_rows: List[PairRow] = []

    with pairs_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            row = build_pair_row(ex, density)
            if row is None:
                continue

            key = row.prefix_id or f"{seed}_{idx}"
            digest_input = f"{key}_{seed}".encode("utf-8")
            digest = int(hashlib.md5(digest_input).hexdigest(), 16) % 1_000_000
            if digest < threshold:
                val_rows.append(row)
            else:
                train_rows.append(row)

    return train_rows, val_rows


def evaluate_reward_model(
    backbone,
    loader,
    tokenizer,
    pad_id: int,
    weight_mode: str,
    weight_cap: Optional[float],
    autocast_ctx,
    max_len: int,
    min_prompt: int,
    device: torch.device,
    label_token_id: int,
) -> Tuple[float, float]:
    losses: List[float] = []
    margins: List[float] = []

    backbone_was_training = backbone.training
    backbone.eval()

    with torch.no_grad():
        for rows in loader:
            chosen_input_ids, rejected_input_ids, example_weights = make_batch(
                rows, tokenizer, max_len, min_prompt, device, pad_id
            )
            with autocast_ctx:
                chosen_logits = backbone(chosen_input_ids)[:, -1, :]
                rejected_logits = backbone(rejected_input_ids)[:, -1, :]
                chosen_rewards = chosen_logits[:, label_token_id]
                rejected_rewards = rejected_logits[:, label_token_id]
            reward_margins = chosen_rewards - rejected_rewards
            loss_vec = F.softplus(-reward_margins)
            loss = apply_weights(loss_vec, example_weights, weight_mode, weight_cap)
            losses.append(loss.float().item())
            margins.append(reward_margins.float().mean().item())

    if backbone_was_training:
        backbone.train()

    if not losses:
        return float("nan"), float("nan")

    return float(np.mean(losses)), float(np.mean(margins))

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
backbone.train()
for p in backbone.parameters():
    p.requires_grad_(False)

likert_token_ids = tokenizer.encode(likert_token)
assert len(likert_token_ids) == 1, f"Likert token '{likert_token}' must map to a single token"
label_token_id = likert_token_ids[0]
rating_prompt_ids = tokenizer.encode(likert_prompt)
assert len(rating_prompt_ids) > 0, "Likert rating prompt must produce at least one token"

# Unfreeze the final transformer blocks for joint training with the RM head
trainable_block_indices = [-2, -1]
trainable_blocks = []
for block_idx in trainable_block_indices:
    block = backbone.transformer.h[block_idx]
    for p in block.parameters():
        p.requires_grad_(True)
    block.train()
    trainable_blocks.append((block_idx, block))
print0(f"Unfreezing transformer blocks {trainable_block_indices} for reward model fine-tuning")

hidden_size = getattr(getattr(backbone, "config", None), "n_embd", None)
assert hidden_size is not None, "model.config.n_embd not found"

pad_id = tokenizer.encode_special("<|assistant_end|>")

# Load or build dataset
print0(f"Loading preference dataset from {pairs_path}...")
density = None
if density_aware:
    print0(f"Loading density weights from {density_weights_path}...")
    prefixes_path = os.path.join(base, "data", "prefixes_all.jsonl")
    density = load_density_mapping(Path(prefixes_path), Path(density_weights_path))

train_rows: List[PairRow] = []
val_rows: List[PairRow] = []
if val_ratio > 0.0:
    train_rows, val_rows = split_pair_rows(Path(pairs_path), density, val_ratio, val_seed)
    print0(f"Train/val split: {len(train_rows)} train | {len(val_rows)} val (ratio={val_ratio})")

ds = PairsDataset(Path(pairs_path), density, rows=train_rows or None)
print0(f"Dataset size: {len(ds)} pairs | density_aware={density_aware}")

val_ds = PairsDataset(Path(pairs_path), density, rows=val_rows or None) if val_rows else None

# DataLoader with DistributedSampler
sampler = DistributedSampler(ds, shuffle=True) if ddp else None
dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=(sampler is None),
    sampler=sampler,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
    collate_fn=lambda batch: batch,
)

val_dl = None
if val_ds is not None:
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda batch: batch,
    )

# Optimizer (policy head will be LM logit, so we only optimize unfrozen blocks)
param_groups = []
for _, block in trainable_blocks:
    param_groups.append({"params": block.parameters(), "lr": backbone_lr})
opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)

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
        
        chosen_input_ids, rejected_input_ids, example_weights = make_batch(
            _rows, tokenizer, max_len, min_prompt, device, pad_id
        )
        
        # Forward
        with autocast_ctx:
            chosen_logits = backbone(chosen_input_ids)[:, -1, :]
            rejected_logits = backbone(rejected_input_ids)[:, -1, :]

            chosen_rewards = chosen_logits[:, label_token_id]
            rejected_rewards = rejected_logits[:, label_token_id]

            reward_margins = chosen_rewards - rejected_rewards
            loss_vec = F.softplus(-reward_margins)
            loss = apply_weights(loss_vec, example_weights, weight_mode, weight_cap)
        loss = loss.float()
        
        # Backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_params = []
        for _, block in trainable_blocks:
            clip_params.extend(block.parameters())
        if clip_params:
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
        opt.step()
        
        # Logging
        if step % log_every == 0:
            with torch.no_grad():
                margin_mean = reward_margins.mean().item()
                loss_mean = loss_vec.mean().item()
                weight_mean = example_weights.mean().item()

            print0(
                "step {}/{} | loss {:.4f} (per-ex {:.4f}) | margin {:+.4f} | weight_mean {:.4g}".format(
                    step,
                    max_steps,
                    loss.item(),
                    loss_mean,
                    margin_mean,
                    weight_mean,
                )
            )
            
            wandb_run.log({
                "step": step,
                "loss": loss.item(),
                "loss_ex_mean": loss_mean,
                "reward_margin": margin_mean,
                "weight_mean": weight_mean,
            })

        if val_dl is not None and eval_every > 0 and (step % eval_every == 0 or step == max_steps):
            val_loss, val_margin = evaluate_reward_model(
                backbone,
                val_dl,
                tokenizer,
                pad_id,
                weight_mode,
                weight_cap,
                autocast_ctx,
                max_len,
                min_prompt,
                device,
                label_token_id,
            )
            wandb_run.log({
                "step": step,
                "val/loss": val_loss,
                "val/reward_margin": val_margin,
            })
            print0(f"    [val] loss {val_loss:.4f} | reward_margin {val_margin:+.4f}")

# Checkpoint
if master_process:
    ckpt = {
        "backbone_blocks_state_dict": {str(idx): block.state_dict() for idx, block in trainable_blocks},
        "meta": {
            "weight_mode": weight_mode,
            "weight_cap": weight_cap,
            "density_aware": density_aware,
            "pad_id": int(pad_id),
            "max_len": max_len,
            "min_prompt": min_prompt,
            "backbone_block_indices": trainable_block_indices,
            "backbone_lr": backbone_lr,
            "learning_rate": learning_rate,
            "val_ratio": val_ratio,
            "val_seed": val_seed,
            "likert_token": likert_token,
            "likert_prompt": likert_prompt,
        }
    }
    out_path = Path(save_dir) / f"model_{int(time.time())}.pt"
    torch.save(ckpt, out_path)
    print0(f"✓ Saved reward checkpoint to {out_path}")

wandb_run.finish()
compute_cleanup()
