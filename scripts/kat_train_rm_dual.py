#!/usr/bin/env python3
"""
Dual-response reward model training using Likert token logits.

Each preference pair is converted into a single sequence:

    <prefix>
    ### Response A ###
    <response>
    ### End Response A ###
    ### Response B ###
    <response>
    ### End Response B ###
    Rating (Response A first, Response B second):

Where the two responses are randomly assigned to slots A/B every step. The
model is encouraged to output "7 1" (preferred, rejected) or vice versa. We use
the logits for the digits to compute a Bradley–Terry style margin.
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import math
import time
import hashlib
import random
from contextlib import nullcontext
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

from scripts.kat_utils import ensure_prefix_dict, prefix_id_from_prefix, render_prefix_for_completion

print_banner()

# ----------------------------------------------------------------------------
# Configuration (overridable via configurator)
# ----------------------------------------------------------------------------

run = "dummy"
device_type = ""
rm_source = "rm"
batch_size = 64
learning_rate = 5e-5
backbone_lr = 5e-6
weight_decay = 0.0
max_steps = 1000
log_every = 25
eval_every = 100
val_ratio = 0.05
val_seed = 123

rating_prompt = "\nRating (Response A first, Response B second):"
preferred_digit = "7"
rejected_digit = "1"

pairs_path = None
density_weights_path = None
weight_mode = "mean"
weight_cap = None
max_len = 512
min_prompt = 128

config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join("nanochat", "configurator.py")).read())
user_config = {k: globals()[k] for k in config_keys}

device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rm", name=run, config=user_config)

# ----------------------------------------------------------------------------
# Dataset helpers
# ----------------------------------------------------------------------------

@dataclass
class PairExample:
    prefix: dict
    chosen: str
    rejected: str
    weight: float
    prefix_id: Optional[str]


def build_pair_example(example: Dict, density: Optional[Dict[str, float]]) -> Optional[PairExample]:
    chosen = example.get("chosen")
    rejected = example.get("rejected")
    if not chosen or not rejected:
        return None

    prefix = ensure_prefix_dict(example.get("prefix"))
    prefix_id = example.get("prefix_id") or prefix_id_from_prefix(prefix)
    weight = 1.0
    if density is not None and prefix_id:
        weight = float(density.get(prefix_id, 1.0))

    return PairExample(prefix, chosen, rejected, weight, prefix_id)


class PairDataset(Dataset):
    def __init__(self, rows: List[PairExample]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def load_pairs_with_split(
    path: Path,
    density: Optional[Dict[str, float]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[PairExample], List[PairExample]]:
    train_rows: List[PairExample] = []
    val_rows: List[PairExample] = []

    threshold = int(max(0.0, min(1.0, val_ratio)) * 1_000_000)

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            row = build_pair_example(ex, density)
            if row is None:
                continue

            if threshold == 0:
                train_rows.append(row)
                continue

            key = row.prefix_id or f"{seed}_{idx}"
            digest = int(hashlib.md5(f"{key}_{seed}".encode("utf-8")).hexdigest(), 16) % 1_000_000
            if digest < threshold:
                val_rows.append(row)
            else:
                train_rows.append(row)

    return train_rows, val_rows


def truncate_two(p: List[int], r: List[int], max_len: int, min_prompt: int) -> Tuple[List[int], List[int]]:
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
    rows: List[PairExample],
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
    fixed_overhead = (
        len(response_a_prefix_ids)
        + len(response_a_suffix_ids)
        + len(response_b_prefix_ids)
        + len(response_b_suffix_ids)
        + rating_prompt_len
        + 2  # digits
    )

    if fixed_overhead >= max_len:
        raise ValueError("max_len too small for dual-response format")

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

        # Determine available prefix budget (reserve at least one token per completion)
        min_completion_tokens = 2
        prefix_budget = max_len - fixed_overhead - min_completion_tokens
        if prefix_budget < 0:
            prefix_budget = 0

        if len(prefix_ids) > prefix_budget:
            prefix_ids = prefix_ids[-prefix_budget:]

        remaining = max_len - (len(prefix_ids) + fixed_overhead)
        if remaining < min_completion_tokens:
            remaining = min_completion_tokens

        # Allocate completion budgets proportional to their lengths
        len_chosen = max(1, len(chosen_ids))
        len_rejected = max(1, len(rejected_ids))
        total_len = len_chosen + len_rejected
        budget_chosen = min(len(chosen_ids), max(1, remaining * len_chosen // total_len))
        budget_rejected = remaining - budget_chosen
        if budget_rejected < 1:
            budget_rejected = 1
            if budget_chosen > 1:
                budget_chosen -= 1
        if budget_chosen < 1:
            budget_chosen = 1
        if budget_chosen + budget_rejected > remaining:
            budget_rejected = remaining - budget_chosen
            if budget_rejected < 1:
                budget_rejected = 1
                budget_chosen = max(1, remaining - budget_rejected)

        chosen_trimmed = chosen_ids[:budget_chosen]
        rejected_trimmed = rejected_ids[:budget_rejected]

        if random.random() < 0.5:
            first_response = chosen_trimmed
            second_response = rejected_trimmed
            first_digit = preferred_token_id
            second_digit = rejected_token_id
        else:
            first_response = rejected_trimmed
            second_response = chosen_trimmed
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


def apply_weights(loss_per_ex: torch.Tensor, weights: torch.Tensor, mode: str, cap: Optional[float]) -> torch.Tensor:
    if cap is not None:
        weights = torch.clamp(weights, max=cap)
    if mode == "mean":
        norm_weights = weights / (weights.mean() + 1e-12)
    elif mode == "sum":
        norm_weights = weights * (weights.numel() / (weights.sum() + 1e-12))
    else:
        norm_weights = weights
    return (norm_weights * loss_per_ex).mean()


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
    rating_prompt_ids: List[int],
    response_a_prefix_ids: List[int],
    response_a_suffix_ids: List[int],
    response_b_prefix_ids: List[int],
    response_b_suffix_ids: List[int],
    preferred_token_id: int,
    rejected_token_id: int,
) -> Tuple[float, float]:
    losses: List[float] = []
    margins: List[float] = []

    backbone_was_training = backbone.training
    backbone.eval()

    with torch.no_grad():
        for rows in loader:
            inputs, digit1_idx, digit2_idx, digit1_tokens, digit2_tokens, weights = build_dual_sequences(
                rows,
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

            with autocast_ctx:
                logits = backbone(inputs)

            batch_size_cur = inputs.size(0)
            batch_idx = torch.arange(batch_size_cur, device=device)
            digit1_idx_tensor = torch.tensor(digit1_idx, dtype=torch.long, device=device)
            digit2_idx_tensor = torch.tensor(digit2_idx, dtype=torch.long, device=device)
            digit1_tokens_tensor = torch.tensor(digit1_tokens, dtype=torch.long, device=device)
            digit2_tokens_tensor = torch.tensor(digit2_tokens, dtype=torch.long, device=device)
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

            logits_digit1 = logits[batch_idx, digit1_idx_tensor - 1, :]
            logits_digit2 = logits[batch_idx, digit2_idx_tensor - 1, :]
            reward_first = logits_digit1.gather(1, digit1_tokens_tensor.unsqueeze(1)).squeeze(1)
            reward_second = logits_digit2.gather(1, digit2_tokens_tensor.unsqueeze(1)).squeeze(1)

            first_is_preferred = digit1_tokens_tensor == preferred_token_id
            reward_chosen = torch.where(first_is_preferred, reward_first, reward_second)
            reward_rejected = torch.where(first_is_preferred, reward_second, reward_first)
            rewards = reward_chosen - reward_rejected

            loss_vec = F.softplus(-rewards)
            loss = apply_weights(loss_vec, weights_tensor, weight_mode, weight_cap)

            losses.append(loss.float().item())
            margins.append(rewards.float().mean().item())

    if backbone_was_training:
        backbone.train()

    if not losses:
        return float("nan"), float("nan")

    return float(np.mean(losses)), float(np.mean(margins))


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

base = get_base_dir()
save_dir_mapping = {
    "rm": os.path.join(base, "rm_checkpoints", "uniform", "d20_dual"),
    "rm_density": os.path.join(base, "rm_checkpoints", "density", "d20_dual"),
}
save_dir = save_dir_mapping[rm_source]
density_aware = rm_source == "rm_density"

if pairs_path is None:
    pairs_path = os.path.join(base, "data", "pairs_all.jsonl")
if density_weights_path is None:
    density_weights_path = os.path.join(base, "data", "embeddings_offline", "density_weights.npy")

backbone, tokenizer, _ = load_model(source="sft", device=device, phase="eval")
backbone.train()
for p in backbone.parameters():
    p.requires_grad_(False)

response_a_prefix_ids = tokenizer.encode("\n### Response A ###\n")
response_a_suffix_ids = tokenizer.encode("\n### End Response A ###\n")
response_b_prefix_ids = tokenizer.encode("\n### Response B ###\n")
response_b_suffix_ids = tokenizer.encode("\n### End Response B ###\n")
rating_prompt_ids = tokenizer.encode(rating_prompt)
assert rating_prompt_ids, "rating_prompt must provide at least one token"

preferred_token_ids = tokenizer.encode(preferred_digit)
rejected_token_ids = tokenizer.encode(rejected_digit)
if len(preferred_token_ids) != 1 or len(rejected_token_ids) != 1:
    raise ValueError("Likert digits must map to single tokens")
preferred_token_id = preferred_token_ids[0]
rejected_token_id = rejected_token_ids[0]

trainable_block_indices = [-2, -1]
trainable_blocks = []
for idx in trainable_block_indices:
    block = backbone.transformer.h[idx]
    for param in block.parameters():
        param.requires_grad_(True)
    block.train()
    trainable_blocks.append((idx, block))

pad_id = tokenizer.encode_special("<|assistant_end|>")

density_map = None
if density_aware:
    prefixes_path = os.path.join(base, "data", "prefixes_all.jsonl")
    ids = []
    with open(prefixes_path, "r", encoding="utf-8") as handle:
        for line in handle:
            ids.append(json.loads(line)["id"])
    weights = np.load(density_weights_path)
    assert len(ids) == len(weights)
    density_map = {pid: float(w) for pid, w in zip(ids, weights.tolist())}

train_rows, val_rows = load_pairs_with_split(Path(pairs_path), density_map, val_ratio, val_seed)
train_dataset = PairDataset(train_rows)
val_dataset = PairDataset(val_rows) if val_rows else None

print0(f"Loaded {len(train_dataset)} train pairs | {len(val_dataset) if val_dataset else 0} val pairs")

sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=(sampler is None),
    sampler=sampler,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
    collate_fn=lambda batch: batch,
)

val_loader = None
if val_dataset:
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda batch: batch,
    )

optimizer = torch.optim.AdamW(
    [{"params": block.parameters(), "lr": backbone_lr} for _, block in trainable_blocks],
    weight_decay=weight_decay,
)

autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

if master_process:
    os.makedirs(save_dir, exist_ok=True)

step = 0
while step < max_steps:
    if sampler is not None:
        sampler.set_epoch(step)

    for batch_rows in train_loader:
        step += 1
        if step > max_steps:
            break

        inputs, digit1_idx, digit2_idx, digit1_tokens, digit2_tokens, weights = build_dual_sequences(
            batch_rows,
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
            shuffle_seed=step,
            preferred_token_id=preferred_token_id,
            rejected_token_id=rejected_token_id,
        )

        digit1_idx_tensor = torch.tensor(digit1_idx, dtype=torch.long, device=device)
        digit2_idx_tensor = torch.tensor(digit2_idx, dtype=torch.long, device=device)
        digit1_tokens_tensor = torch.tensor(digit1_tokens, dtype=torch.long, device=device)
        digit2_tokens_tensor = torch.tensor(digit2_tokens, dtype=torch.long, device=device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        batch_indices = torch.arange(inputs.size(0), device=device)

        with autocast_ctx:
            logits = backbone(inputs)

        logits_digit1 = logits[batch_indices, digit1_idx_tensor - 1, :]
        logits_digit2 = logits[batch_indices, digit2_idx_tensor - 1, :]
        reward_first = logits_digit1.gather(1, digit1_tokens_tensor.unsqueeze(1)).squeeze(1)
        reward_second = logits_digit2.gather(1, digit2_tokens_tensor.unsqueeze(1)).squeeze(1)

        first_is_preferred = digit1_tokens_tensor == preferred_token_id
        reward_chosen = torch.where(first_is_preferred, reward_first, reward_second)
        reward_rejected = torch.where(first_is_preferred, reward_second, reward_first)
        rewards = reward_chosen - reward_rejected

        loss_vec = F.softplus(-rewards)
        loss = apply_weights(loss_vec, weights_tensor, weight_mode, weight_cap)
        loss = loss.float()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = []
        for _, block in trainable_blocks:
            params.extend(block.parameters())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if step % log_every == 0:
            wandb_run.log({
                "step": step,
                "loss": loss.float().item(),
                "reward_margin": rewards.float().mean().item(),
            })
            print0(f"step {step:06d}/{max_steps:06d} | loss {loss.float().item():.4f} | margin {rewards.float().mean().item():+.4f}")

        if val_loader is not None and eval_every > 0 and (step % eval_every == 0 or step == max_steps):
            val_loss, val_margin = evaluate_reward_model(
                backbone,
                val_loader,
                tokenizer,
                pad_id,
                weight_mode,
                weight_cap,
                autocast_ctx,
                max_len,
                min_prompt,
                device,
                rating_prompt_ids,
                response_a_prefix_ids,
                response_a_suffix_ids,
                response_b_prefix_ids,
                response_b_suffix_ids,
                rejected_token_id, # swappeD!!!!!!!!
                preferred_token_id,
            )
            wandb_run.log({
                "step": step,
                "val/loss": val_loss,
                "val/reward_margin": val_margin,
            })
            print0(f"    [val] loss {val_loss:.4f} | reward_margin {val_margin:+.4f}")

if master_process:
    ckpt = {
        "backbone_blocks_state_dict": {str(idx): block.state_dict() for idx, block in trainable_blocks},
        "meta": {
            "density_aware": density_aware,
            "pad_id": int(pad_id),
            "max_len": max_len,
            "min_prompt": min_prompt,
            "backbone_block_indices": trainable_block_indices,
            "backbone_lr": backbone_lr,
            "learning_rate": learning_rate,
            "val_ratio": val_ratio,
            "val_seed": val_seed,
            "rating_prompt": rating_prompt,
            "preferred_digit": preferred_digit,
            "rejected_digit": rejected_digit,
        }
    }
    out_path = Path(save_dir) / f"model_{int(time.time())}.pt"
    torch.save(ckpt, out_path)
    print0(f"✓ Saved dual reward checkpoint to {out_path}")

wandb_run.finish()
compute_cleanup()


