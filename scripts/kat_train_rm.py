#!/usr/bin/env python3
"""
Train a Reward Model head on pairwise preferences with optional density-aware sampling.

Contract (required):
- Tokenizer: encode(text)->List[int], encode_special("<|assistant_end|>")->int
- SFT model (frozen encoder for RM): must accept:
    model(input_ids=..., attention_mask=..., return_hidden_states=True)
  and return a dict with keys "logits": [B,T,V], "hidden_states": [B,T,H]

The RewardHead maps last non-pad hidden state [B,H] -> [B] (scalar reward).
"""

from __future__ import annotations
import argparse, json, os, sys, hashlib
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from nanochat.checkpoint_manager import load_model, get_base_dir
from nanochat.tokenizer import get_tokenizer

# -------------------------
# Data
# -------------------------

class PreferenceDataset(Dataset):
    """Reads JSONL with keys: 'prompt', 'chosen', 'rejected'."""

    def __init__(self, pairs_path: str):
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
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
# RM Head
# -------------------------

class RewardHead(nn.Module):
    """Linear reward head over last non-pad hidden state."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_hidden: float tensor [B, H]
        Returns:
            rewards: float tensor [B]
        """
        assert last_hidden.dim() == 2, f"last_hidden must be [B,H], got {last_hidden.shape}"
        return self.linear(last_hidden).squeeze(-1)

# -------------------------
# Collation
# -------------------------

def _encode_pair(tokenizer, prompt: str, resp: str, pad_id: int, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (input_ids [T], attention_mask [T]) for prompt + response, truncated to max_len.
    Truncates from the left of the prompt first to preserve response.
    """
    p_ids = tokenizer.encode(prompt)
    r_ids = tokenizer.encode(resp)

    # Always keep at least 1 token of response if possible
    max_len = int(max_len)
    if len(p_ids) + len(r_ids) > max_len:
        keep_r = min(len(r_ids), max_len // 2 if len(r_ids) > 0 else 0)
        keep_p = max_len - keep_r
        if keep_p < 0:
            keep_p = 0
        p_ids = p_ids[-keep_p:]
        r_ids = r_ids[: (max_len - len(p_ids))]

    ids = p_ids + r_ids
    attn = [1] * len(ids)
    # pad to max_len (per-batch we will pad further)
    if len(ids) < max_len:
        pad = [pad_id] * (max_len - len(ids))
        padm = [0] * (max_len - len(ids))
        ids = ids + pad
        attn = attn + padm
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.bool)

def make_collate_rm(tokenizer, max_len: int):
    pad_id = tokenizer.encode_special("<|assistant_end|>")
    def _collate(batch: List[Dict[str,str]]):
        # Encode prompt+chosen and prompt+rejected separately
        ids_c, attn_c = [], []
        ids_r, attn_r = [], []
        for row in batch:
            ic, ac = _encode_pair(tokenizer, row["prompt"], row["chosen"], pad_id, max_len)
            ir, ar = _encode_pair(tokenizer, row["prompt"], row["rejected"], pad_id, max_len)
            ids_c.append(ic); attn_c.append(ac)
            ids_r.append(ir); attn_r.append(ar)
        chosen_input_ids  = torch.stack(ids_c, dim=0)   # [B,T]
        chosen_attn       = torch.stack(attn_c, dim=0)  # [B,T]
        rejected_input_ids= torch.stack(ids_r, dim=0)   # [B,T]
        rejected_attn     = torch.stack(attn_r, dim=0)  # [B,T]
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attn": chosen_attn,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attn": rejected_attn,
        }
    return _collate

# -------------------------
# Utilities
# -------------------------

def last_nonpad_hidden(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """
    hidden: [B,T,H], attn: [B,T] bool -> returns [B,H] at the last position where attn==1 for each sequence
    """
    assert hidden.dim()==3 and attn.dim()==2, f"hidden {hidden.shape}, attn {attn.shape}"
    B,T,H = hidden.shape
    lengths = attn.long().sum(dim=1)  # [B]
    idx = torch.clamp(lengths - 1, min=0)  # [B]
    batch_idx = torch.arange(B, device=hidden.device)
    return hidden[batch_idx, idx, :]  # [B,H]

# -------------------------
# Training
# -------------------------

def get_prompt_id(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16]

def main():
    parser = argparse.ArgumentParser()
    default_pairs = os.path.join(get_base_dir(), "data", "pairs_all.jsonl")
    parser.add_argument("--pairs_path", default=default_pairs)
    parser.add_argument("--sft_source", default="sft")
    parser.add_argument("--rm_source", default="rm")  # decides CKPT dir
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--embeddings_dir", default=os.path.join(get_base_dir(), "data", "embeddings_offline"))
    parser.add_argument("--density_aware", action="store_true")

    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    args = parser.parse_args()

    if args.out_dir is None:
        mapping = {
            "rm": os.path.join(get_base_dir(), "rm_checkpoints", "uniform", "d20"),
            "rm_density": os.path.join(get_base_dir(), "rm_checkpoints", "density", "d20"),
        }
        if args.rm_source not in mapping:
            raise ValueError(f"Unknown rm_source: {args.rm_source}")
        args.out_dir = mapping[args.rm_source]
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen encoder
    model, tokenizer, meta = load_model(source=args.sft_source, device=device, phase="eval")
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    # Hidden size
    hidden_size = getattr(getattr(model, "config", None), "n_embd", None)
    if hidden_size is None:
        raise RuntimeError("model.config.n_embd not found; required for RewardHead size")

    # Data
    dataset = PreferenceDataset(args.pairs_path)
    collate = make_collate_rm(tokenizer, args.max_seq_len)

    # Optional density weights
    sampler = None
    if args.density_aware:
        weights_path = os.path.join(args.embeddings_dir, "density_weights.npy")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Density weights not found: {weights_path}")
        weights = np.load(weights_path)
        # Build prompt_id -> index
        prompts_path = os.path.join(get_base_dir(), "data", "prompts_all.jsonl")
        prompt_id_to_idx: Dict[str,int] = {}
        with open(prompts_path, "r") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                prompt_id_to_idx[obj["id"]] = i
        pair_w = np.ones(len(dataset), dtype=np.float32)
        # Map weights per pair
        with open(args.pairs_path, "r") as f:
            for j, line in enumerate(f):
                obj = json.loads(line)
                pid = get_prompt_id(obj["prompt"])
                if pid in prompt_id_to_idx:
                    k = prompt_id_to_idx[pid]
                    if 0 <= k < len(weights): pair_w[j] = float(weights[k])
        sampler = WeightedRandomSampler(torch.from_numpy(pair_w), len(dataset), replacement=True)

    loader = DataLoader(dataset,
                        batch_size=args.device_batch_size,
                        sampler=sampler,
                        shuffle=(sampler is None),
                        num_workers=args.num_workers,
                        collate_fn=collate,
                        drop_last=False)

    rm_head = RewardHead(hidden_size).to(device)
    opt = torch.optim.Adam(rm_head.parameters(), lr=args.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))

    step = 0
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps: break
            # Forward encoder (frozen) to extract hidden states
            with torch.no_grad():
                out_c = model(input_ids=batch["chosen_input_ids"].to(device),
                              attention_mask=batch["chosen_attn"].to(device),
                              return_hidden_states=True)
                out_r = model(input_ids=batch["rejected_input_ids"].to(device),
                              attention_mask=batch["rejected_attn"].to(device),
                              return_hidden_states=True)

            # Strong contract checks
            for name, out in (("chosen", out_c), ("rejected", out_r)):
                assert isinstance(out, dict) and "hidden_states" in out, f"{name}: model must return dict with 'hidden_states'"
                H = out["hidden_states"]
                assert H.dim()==3, f"{name}: hidden_states shape must be [B,T,H], got {H.shape}"

            last_c = last_nonpad_hidden(out_c["hidden_states"], batch["chosen_attn"].to(device))  # [B,H]
            last_r = last_nonpad_hidden(out_r["hidden_states"], batch["rejected_attn"].to(device))  # [B,H]

            # Reward scores
            rc = rm_head(last_c)  # [B]
            rr = rm_head(last_r)  # [B]

            # Bradley–Terry loss
            loss = -F.logsigmoid(rc - rr).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            if step % args.log_interval == 0:
                writer.add_scalar("train/loss", float(loss.item()), step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                print(f"[RM] step {step}/{args.max_steps}  loss={loss.item():.4f}")

            step += 1

    # Save head checkpoint (format: rm_head_state_dict + config)
    ckpt = {
        "rm_head_state_dict": rm_head.state_dict(),
        "config": {"hidden_size": hidden_size, "density_aware": args.density_aware}
    }
    out_path = os.path.join(args.out_dir, "model_000000.pt")
    torch.save(ckpt, out_path)
    print(f"✓ Saved RM head to {out_path}")
    writer.close()

if __name__ == "__main__":
    main()
