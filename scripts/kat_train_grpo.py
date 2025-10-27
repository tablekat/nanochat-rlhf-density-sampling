#!/usr/bin/env python3
"""
GRPO training with a clean RM interface.

Policy gradient is driven by pairwise advantage-weighted log-prob differences
over response tokens only; RM provides fixed rewards computed on a frozen SFT encoder.

Loss per batch:
    A = (r_c - r_r) - beta * (KL_c - KL_r)
    L = - A * ( logp_c - logp_r )

Contracts (required) match kat_train_rm.py:
- SFT model forward returns dict with 'logits':[B,T,V], 'hidden_states':[B,T,H]
- RewardHead maps last hidden [B,H] -> [B]
"""

from __future__ import annotations
import argparse, json, os, sys
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from nanochat.checkpoint_manager import load_model, get_base_dir
from nanochat.tokenizer import get_tokenizer

# -------------------------
# Data
# -------------------------

class PreferenceDataset(Dataset):
    def __init__(self, pairs_path: str):
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(pairs_path)
        self.rows: List[Dict[str,str]] = []
        with open(pairs_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                obj = json.loads(line)
                for k in ("prompt","chosen","rejected"):
                    if k not in obj:
                        raise KeyError(f"Missing '{k}' on line {ln}")
                self.rows.append({"prompt": obj["prompt"], "chosen": obj["chosen"], "rejected": obj["rejected"]})
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self, idx: int) -> Dict[str,str]: return self.rows[idx]

# -------------------------
# Reward head (same as RM script)
# -------------------------

class RewardHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, last_hidden: torch.Tensor) -> torch.Tensor:
        assert last_hidden.dim()==2, f"last_hidden must be [B,H], got {last_hidden.shape}"
        return self.linear(last_hidden).squeeze(-1)

# -------------------------
# Collation
# -------------------------

def _encode_concat(tokenizer, prompt: str, resp: str, pad_id: int, max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      input_ids [T], attention_mask [T], response_mask [T]  (True only on response tokens)
    Truncates from left of prompt first to preserve response.
    """
    p_ids = tokenizer.encode(prompt)
    r_ids = tokenizer.encode(resp)
    max_len = int(max_len)
    # Keep response; shrink prompt from left
    if len(p_ids) + len(r_ids) > max_len:
        keep_r = min(len(r_ids), max_len // 2 if len(r_ids) > 0 else 0)
        keep_p = max_len - keep_r
        if keep_p < 0: keep_p = 0
        p_ids = p_ids[-keep_p:]
        r_ids = r_ids[: (max_len - len(p_ids))]

    ids = p_ids + r_ids
    T = len(ids)
    attn = [1] * T
    resp_mask = [0]*len(p_ids) + [1]*len(r_ids)

    if T < max_len:
        pad = [pad_id] * (max_len - T)
        padm = [0] * (max_len - T)
        ids = ids + pad
        attn = attn + padm
        resp_mask = resp_mask + padm

    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.bool),
        torch.tensor(resp_mask, dtype=torch.bool),
    )

def make_collate_grpo(tokenizer, max_len: int):
    pad_id = tokenizer.encode_special("<|assistant_end|>")
    def _collate(batch: List[Dict[str,str]]):
        ci, ca, cr = [], [], []
        ri, ra, rr = [], [], []
        for row in batch:
            i,a,r = _encode_concat(tokenizer, row["prompt"], row["chosen"], pad_id, max_len)
            ci.append(i); ca.append(a); cr.append(r)
            i,a,r = _encode_concat(tokenizer, row["prompt"], row["rejected"], pad_id, max_len)
            ri.append(i); ra.append(a); rr.append(r)
        return {
            "chosen_input_ids": torch.stack(ci,0),  # [B,T]
            "chosen_attn": torch.stack(ca,0),       # [B,T]
            "chosen_resp": torch.stack(cr,0),       # [B,T]
            "rejected_input_ids": torch.stack(ri,0),
            "rejected_attn": torch.stack(ra,0),
            "rejected_resp": torch.stack(rr,0),
        }
    return _collate

# -------------------------
# Math helpers
# -------------------------

def seq_logprob_over_response(logits: torch.Tensor, input_ids: torch.Tensor, resp_mask: torch.Tensor) -> torch.Tensor:
    """
    logits: [B,T,V] over tokens 0..T-1
    We use teacher-forcing next-token log-prob:
        use logits[:, :-1] to predict labels = input_ids[:, 1:]
    Only sum on positions where resp_mask for the *label* token is True.
    Returns [B] sum of token log-probs over response labels.
    """
    assert logits.dim()==3 and input_ids.dim()==2 and resp_mask.dim()==2
    B,T,V = logits.shape
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)          # [B,T-1,V]
    labels = input_ids[:, 1:]                                 # [B,T-1]
    token_lp = logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B,T-1]

    resp_mask_labels = resp_mask[:, 1:]                       # [B,T-1]
    token_lp = token_lp * resp_mask_labels.float()
    return token_lp.sum(dim=1)                                # [B]

def seq_kl_over_response(pol_logits: torch.Tensor, ref_logits: torch.Tensor, resp_mask: torch.Tensor) -> torch.Tensor:
    """
    KL(policy || reference) per sequence, summed over response label positions.
    """
    assert pol_logits.shape == ref_logits.shape
    logp = F.log_softmax(pol_logits[:, :-1, :], dim=-1)      # [B,T-1,V]
    logq = F.log_softmax(ref_logits[:, :-1, :], dim=-1)      # [B,T-1,V]
    p = logp.exp()
    kl_tok = (p * (logp - logq)).sum(dim=-1)                 # [B,T-1]
    resp_mask_labels = resp_mask[:, 1:]                      # [B,T-1]
    kl_tok = kl_tok * resp_mask_labels.float()
    return kl_tok.sum(dim=1)                                 # [B]

def last_nonpad_hidden(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    assert hidden.dim()==3 and attn.dim()==2
    B,T,H = hidden.shape
    idx = attn.long().sum(dim=1).clamp(min=1) - 1            # [B]
    batch = torch.arange(B, device=hidden.device)
    return hidden[batch, idx, :]                             # [B,H]

# -------------------------
# Load RM head ckpt
# -------------------------

def load_rm_head(ckpt_dir: str, hidden_size: int, device: torch.device) -> RewardHead:
    path = os.path.join(ckpt_dir, "model_000000.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"RM checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu")
    sd = data.get("rm_head_state_dict", None)
    if sd is None:
        # backward compat
        sd = data.get("model_state_dict", None)
    if sd is None:
        raise KeyError(f"'rm_head_state_dict' not found in {path}")
    head = RewardHead(hidden_size)
    head.load_state_dict(sd)
    return head.to(device)

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO training with clean RM usage")
    default_pairs = os.path.join(get_base_dir(), "data", "pairs_all.jsonl")
    parser.add_argument("--pairs_path", default=default_pairs)
    parser.add_argument("--sft_source", default="sft")            # policy + reference init
    parser.add_argument("--rm_source", default="rm")              # selects RM ckpt dir
    parser.add_argument("--grpo_source", default="grpo")          # selects output dir layout
    parser.add_argument("--out_dir", default=None)

    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)        # KL penalty coefficient
    parser.add_argument("--device_batch_size", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    args = parser.parse_args()

    # Resolve dirs
    if args.out_dir is None:
        mapping = {
            "grpo": os.path.join(get_base_dir(), "grpo_checkpoints", "uniform", "d20"),
            "grpo_density": os.path.join(get_base_dir(), "grpo_checkpoints", "density", "d20"),
        }
        if args.grpo_source not in mapping:
            raise ValueError(f"Unknown grpo_source: {args.grpo_source}")
        args.out_dir = mapping[args.grpo_source]
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer + models
    # Policy (trainable)
    policy, tokenizer, _ = load_model(source=args.sft_source, device=device, phase="train")
    # Reference (frozen)
    reference, _, _ = load_model(source=args.sft_source, device=device, phase="eval")
    reference.eval()
    for p in reference.parameters(): p.requires_grad_(False)
    # Frozen encoder for RM (use the same SFT backbone as RM training)
    rm_encoder, _, _ = load_model(source=args.sft_source, device=device, phase="eval")
    rm_encoder.eval()
    for p in rm_encoder.parameters(): p.requires_grad_(False)

    hidden_size = getattr(getattr(policy, "config", None), "n_embd", None)
    if hidden_size is None:
        raise RuntimeError("policy.config.n_embd not found")

    # RM head
    rm_ckpt_dir = {
        "rm": os.path.join(get_base_dir(), "rm_checkpoints", "uniform", "d20"),
        "rm_density": os.path.join(get_base_dir(), "rm_checkpoints", "density", "d20"),
    }.get(args.rm_source, None)
    if rm_ckpt_dir is None:
        raise ValueError(f"Unknown rm_source: {args.rm_source}")
    rm_head = load_rm_head(rm_ckpt_dir, hidden_size=hidden_size, device=device)
    rm_head.eval()
    for p in rm_head.parameters(): p.requires_grad_(False)

    # Data
    dataset = PreferenceDataset(args.pairs_path)
    collate = make_collate_grpo(tokenizer, args.max_seq_len)
    loader = DataLoader(dataset,
                        batch_size=args.device_batch_size,
                        shuffle=True,
                        num_workers=0,
                        collate_fn=collate)

    # Optimizer / scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))

    step = 0
    policy.train()
    reference.eval()
    rm_encoder.eval()

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps: break

            # Move to device
            ci = batch["chosen_input_ids"].to(device)
            ca = batch["chosen_attn"].to(device)
            cr = batch["chosen_resp"].to(device)

            ri = batch["rejected_input_ids"].to(device)
            ra = batch["rejected_attn"].to(device)
            rr = batch["rejected_resp"].to(device)

            # Forward policy + reference (for LL and KL, response tokens only)
            out_pc = policy(input_ids=ci, attention_mask=ca, return_hidden_states=True)
            out_pr = policy(input_ids=ri, attention_mask=ra, return_hidden_states=True)
            out_rc = reference(input_ids=ci, attention_mask=ca, return_hidden_states=True)
            out_rr = reference(input_ids=ri, attention_mask=ra, return_hidden_states=True)

            # Strict contract checks (fail fast with clear messages)
            for name, out in (("policy:chosen", out_pc), ("policy:rejected", out_pr),
                              ("ref:chosen", out_rc), ("ref:rejected", out_rr)):
                assert isinstance(out, dict), f"{name}: model must return dict"
                assert "logits" in out, f"{name}: missing logits"
                assert out["logits"].dim()==3, f"{name}: logits must be [B,T,V], got {out['logits'].shape}"

            # Log-prob sums over response labels
            logp_c = seq_logprob_over_response(out_pc["logits"], ci, cr)  # [B]
            logp_r = seq_logprob_over_response(out_pr["logits"], ri, rr)  # [B]

            # KL sums over response labels
            kl_c = seq_kl_over_response(out_pc["logits"], out_rc["logits"], cr)  # [B]
            kl_r = seq_kl_over_response(out_pr["logits"], out_rr["logits"], rr)  # [B]

            # Rewards from RM (frozen encoder + frozen head), computed on FULL sequences
            with torch.no_grad():
                enc_c = rm_encoder(input_ids=ci, attention_mask=ca, return_hidden_states=True)
                enc_r = rm_encoder(input_ids=ri, attention_mask=ra, return_hidden_states=True)
                for name, out in (("rm:chosen", enc_c), ("rm:rejected", enc_r)):
                    assert isinstance(out, dict) and "hidden_states" in out and out["hidden_states"].dim()==3, \
                        f"{name}: need hidden_states [B,T,H]"
                last_c = last_nonpad_hidden(enc_c["hidden_states"], ca)   # [B,H]
                last_r = last_nonpad_hidden(enc_r["hidden_states"], ra)   # [B,H]
                r_c = rm_head(last_c)  # [B]
                r_r = rm_head(last_r)  # [B]

            # Pairwise advantage and GRPO loss
            A = (r_c - r_r) - args.beta * (kl_c - kl_r)                    # [B]
            loss = -(A * (logp_c - logp_r)).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % args.log_interval == 0:
                with torch.no_grad():
                    pref_margin = (r_c - r_r).mean().item()
                    kl_margin = (kl_c - kl_r).mean().item()
                    lp_margin = (logp_c - logp_r).mean().item()
                writer.add_scalar("train/loss", float(loss.item()), step)
                writer.add_scalar("train/reward_margin", pref_margin, step)
                writer.add_scalar("train/kl_margin", kl_margin, step)
                writer.add_scalar("train/logprob_margin", lp_margin, step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                print(f"[GRPO] step {step}/{args.max_steps}  loss={loss.item():.4f}  "
                      f"Δr={pref_margin:.4f}  ΔKL={kl_margin:.4f}  Δlogp={lp_margin:.4f}")

            step += 1

    # Save policy checkpoint
    ckpt_path = os.path.join(args.out_dir, "model_000000.pt")
    torch.save({
        "model_state_dict": policy.state_dict(),
        "config": {"beta": args.beta, "training_steps": step}
    }, ckpt_path)
    print(f"✓ Saved GRPO policy to {ckpt_path}")
    writer.close()

if __name__ == "__main__":
    main()
