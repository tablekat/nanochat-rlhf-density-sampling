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
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import glob

import numpy as np
import torch
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
batch_size = 16
learning_rate = 5e-6
weight_decay = 0.0
grad_clip = 0.5
beta = 0.02  # initial KL weight
target_kl = 0.05  # target per-sample KL
beta_gain = 0.02  # KL controller update speed
std_adv = True   # standardize advantage
margin_scale = 0.1  # rescales RM margin before GRPO
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

# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def autocast_if_cuda():
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


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


def pad_sequences(seqs: List[List[int]], max_len: int, pad_id: int, device) -> torch.Tensor:
    seqs = [seq[:max_len] for seq in seqs]
    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in seqs]
    return torch.tensor(padded, dtype=torch.long, device=device)


def build_dual_sequences(
    rows: List[PairRow],
    tokenizer,
    max_len: int,
    min_prompt: int,
    device,
    pad_id: int,
    response_a_prefix_ids: List[int],
    response_a_suffix_ids: List[int],
    response_b_prefix_ids: List[int],
    response_b_suffix_ids: List[int],
    rating_prompt_ids: List[int],
    preferred_token_id: int,
    rejected_token_id: int,
):
    rating_prompt_len = len(rating_prompt_ids)
    fixed_overhead = (
        len(response_a_prefix_ids)
        + len(response_a_suffix_ids)
        + len(response_b_prefix_ids)
        + len(response_b_suffix_ids)
        + rating_prompt_len
        + 2  # reserve space for digits once
    )

    if fixed_overhead >= max_len:
        raise ValueError("max_len too small for dual sequence in GRPO")

    sequences = []
    digit1_indices = []
    digit2_indices = []
    digit1_tokens = []
    digit2_tokens = []

    for row in rows:
        try:
            prefix_ids = render_prefix_for_completion(tokenizer, row.prefix)
        except ValueError:
            prefix_ids = render_prefix_for_completion(tokenizer, None)

        chosen_ids = tokenizer.encode(row.chosen)
        rejected_ids = tokenizer.encode(row.rejected)

        min_completion_tokens = 2
        prefix_budget = max_len - fixed_overhead - min_completion_tokens
        if prefix_budget < 0:
            prefix_budget = 0

        if len(prefix_ids) > prefix_budget:
            prefix_ids = prefix_ids[-prefix_budget:]

        remaining = max_len - (len(prefix_ids) + fixed_overhead)
        if remaining < min_completion_tokens:
            remaining = min_completion_tokens

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

        assembled = (
            prefix_ids
            + response_a_prefix_ids
            + chosen_trimmed
            + response_a_suffix_ids
            + response_b_prefix_ids
            + rejected_trimmed
            + response_b_suffix_ids
            + rating_prompt_ids
        )

        digit1_idx = len(assembled)
        assembled.append(preferred_token_id)
        digit2_idx = len(assembled)
        assembled.append(rejected_token_id)
        if len(assembled) > max_len:
            raise RuntimeError("assembled dual sequence exceeded max_len; adjust budget logic")

        sequences.append(assembled)
        digit1_indices.append(digit1_idx)
        digit2_indices.append(digit2_idx)
        digit1_tokens.append(preferred_token_id)
        digit2_tokens.append(rejected_token_id)

    return pad_sequences(sequences, max_len, pad_id, device), digit1_indices, digit2_indices, digit1_tokens, digit2_tokens

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

def sum_logprobs(model, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log-probs over response tokens (teacher forcing)."""
    with autocast_if_cuda():
        logits = model(x)
    logp = logits.float().log_softmax(dim=-1)
    tgt = labels[:, 1:].contiguous()
    logp = logp[:, :-1].contiguous()
    mask = (tgt != -100)
    gathered = logp.gather(2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=1)

def sum_kl(policy, reference, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """KL(policy || reference) over response tokens, length-normalized."""
    with autocast_if_cuda():
        with torch.no_grad():
            ref_logits = reference(x)
        pol_logits = policy(x)
    ref_logits = ref_logits.float()
    pol_logits = pol_logits.float()
    logp = F.log_softmax(pol_logits[:, :-1, :], dim=-1)
    logq = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
    p = logp.exp()
    kl_tok = (p * (logp - logq)).sum(dim=-1)
    resp_mask = (labels[:, 1:] != -100).float()
    kl_sum = (kl_tok * resp_mask).sum(dim=1)
    resp_len = resp_mask.sum(dim=1).clamp_min(1.0)
    return kl_sum / resp_len

# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

# Resolve paths
base = get_base_dir()
mapping = {
    "rm": os.path.join(base, "rm_checkpoints", "uniform", "d20_dual"),
    "rm_density": os.path.join(base, "rm_checkpoints", "density", "d20_dual"),
}
grpo_out_mapping = {
    "grpo": os.path.join(base, "grpo_checkpoints", "uniform", "d20"),
    "grpo_density": os.path.join(base, "grpo_checkpoints", "density", "d20"),
}

rm_ckpt_dir = mapping[rm_source]
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
print0(f"Loading reward model from {rm_ckpt_dir}...")
rm_ckpt_files = glob.glob(os.path.join(rm_ckpt_dir, "model_*.pt"))
if not rm_ckpt_files:
    raise FileNotFoundError(f"No RM checkpoint found in {rm_ckpt_dir}")
rm_head_path = max(rm_ckpt_files, key=lambda x: int(Path(x).stem.split("_")[1]))
print0(f"Using RM checkpoint: {rm_head_path}")

rm = torch.load(rm_head_path, map_location="cpu")
# head = RewardHead(in_dim=rm["meta"]["features_dim"]).to(device) # Removed RewardHead
# head.load_state_dict(rm["rm_head_state_dict"]) # Removed RewardHead
# for p_ in head.parameters(): # Removed RewardHead
#     p_.requires_grad_(False) # Removed RewardHead

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

        def _cast_state(block, state_dict):
            target_dtype = next(block.parameters()).dtype
            return {k: v.to(target_dtype) for k, v in state_dict.items()}

        policy_state = _cast_state(policy.transformer.h[idx], state)
        reference_state = _cast_state(reference.transformer.h[idx], state)
        policy.transformer.h[idx].load_state_dict(policy_state)
        reference.transformer.h[idx].load_state_dict(reference_state)

meta = rm.get("meta", {})
rating_prompt = meta.get("rating_prompt", "\nRating (Response A first, Response B second):")
preferred_digit = meta.get("preferred_digit", "7")
rejected_digit = meta.get("rejected_digit", "1")

tokenizer = get_tokenizer()
ds = Pairs(Path(pairs_path))
pad_id = tokenizer.encode_special("<|assistant_end|>")

response_a_prefix_ids = tokenizer.encode("\n### Response A ###\n")
response_a_suffix_ids = tokenizer.encode("\n### End Response A ###\n")
response_b_prefix_ids = tokenizer.encode("\n### Response B ###\n")
response_b_suffix_ids = tokenizer.encode("\n### End Response B ###\n")
rating_prompt_ids = tokenizer.encode(rating_prompt)
assert rating_prompt_ids, "rating_prompt must yield tokens"

preferred_token_ids = tokenizer.encode(preferred_digit)
rejected_token_ids = tokenizer.encode(rejected_digit)
if len(preferred_token_ids) != 1 or len(rejected_token_ids) != 1:
    raise ValueError("Likert digits must map to single tokens")
preferred_token_id = preferred_token_ids[0]
rejected_token_id = rejected_token_ids[0]

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

# Optimizer and KL controller
opt = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
beta_min, beta_max = 1e-5, 0.2
kl_ema = None
kl_ema_momentum = 0.9
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
        rm_inputs, digit1_idx, digit2_idx, digit1_tokens, digit2_tokens = build_dual_sequences(
            rows,
            tokenizer,
            max_len,
            min_prompt,
            device,
            pad_id,
            response_a_prefix_ids,
            response_a_suffix_ids,
            response_b_prefix_ids,
            response_b_suffix_ids,
            rating_prompt_ids,
            preferred_token_id,
            rejected_token_id,
        )

        # Log-probs (response-only)
        lp_c = sum_logprobs(policy, x_c, y_c)
        lp_r = sum_logprobs(policy, x_r, y_r)
        
        # KL sums
        kl_c = sum_kl(policy, reference, x_c, y_c)
        kl_r = sum_kl(policy, reference, x_r, y_r)
        
        # Rewards (frozen backbone + RM logits)
        with torch.no_grad():
            with autocast_if_cuda():
                logits = reference(rm_inputs)

        batch_sz = rm_inputs.size(0)
        batch_idx = torch.arange(batch_sz, device=device)
        digit1_idx_tensor = torch.tensor(digit1_idx, dtype=torch.long, device=device)
        digit2_idx_tensor = torch.tensor(digit2_idx, dtype=torch.long, device=device)
        
        # Logits at the positions that PREDICT the two digits
        logits_digit1 = logits[batch_idx, digit1_idx_tensor - 1, :].float()  # [B, V]
        logits_digit2 = logits[batch_idx, digit2_idx_tensor - 1, :].float()  # [B, V]

        # Full-vocab log-probs per slot
        logp1 = F.log_softmax(logits_digit1, dim=-1)  # [B, V]
        logp2 = F.log_softmax(logits_digit2, dim=-1)  # [B, V]

        # Token IDs for pref/rej (scalar IDs)
        pref_id = torch.tensor([preferred_token_id], device=device)
        rej_id  = torch.tensor([rejected_token_id],  device=device)

        # Gather four log-probs (two tokens at each slot)
        lp_pref1 = logp1.gather(1, pref_id.expand(B,1)).squeeze(1)   # log p(pref | slot1)
        lp_rej1  = logp1.gather(1, rej_id .expand(B,1)).squeeze(1)   # log p(rej  | slot1)
        lp_pref2 = logp2.gather(1, pref_id.expand(B,1)).squeeze(1)   # log p(pref | slot2)
        lp_rej2  = logp2.gather(1, rej_id .expand(B,1)).squeeze(1)   # log p(rej  | slot2)

        # Symmetric pairwise margin
        margin = (lp_pref1 + lp_rej2) - (lp_rej1 + lp_pref2)
        margin = margin.clamp(-10, 10)  # optional safety

        rc = 0.5 * margin
        rr = -0.5 * margin
        dr = margin   # no extra scaling needed; drop margin_scale

        # Advantage and loss
        dkl = kl_c - kl_r
        A = dr - kl_beta * dkl
        if std_adv:
            A = (A - A.mean()) / (A.std(unbiased=False) + 1e-6)
        A = A.clamp(-5.0, 5.0)  # optional safety

        loss = -(A.detach() * (lp_c - lp_r)).mean().float()
        
        # Backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        for p_ in policy.parameters():
            if p_.grad is not None:
                p_.grad.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
        opt.step()
        
        # KL controller: target absolute KL (chosen) for stability
        with torch.no_grad():
            kl_now = kl_c.mean().item()
            if kl_ema is None:
                kl_ema = kl_now
            else:
                kl_ema = kl_ema_momentum * kl_ema + (1 - kl_ema_momentum) * kl_now
            step_raw = beta_gain * (kl_ema - target_kl)
            step_clipped = max(-0.2, min(0.2, step_raw))
            kl_beta = max(beta_min, min(beta_max, kl_beta * math.exp(step_clipped)))
        
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
