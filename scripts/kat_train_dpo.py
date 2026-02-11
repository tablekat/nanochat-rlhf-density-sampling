# scripts/train_dpo.py
"""
python scripts/train_dpo.py \
  --pairs data/pairs_invden.jsonl \
  --ckpt_in out/ckpt.pt \
  --ckpt_out out/ckpt.dpo_invden.pt \
  --tokenizer tokenizer.model \
  --bsz 16 --lr 5e-6 --epochs 1 --beta 0.1
"""
import os, json, math, argparse, random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from nanochat.model import Transformer
from nanochat.tokenizer import get_tokenizer, RustBPETokenizer

from scripts.kat_utils import ensure_prefix_dict, prefix_from_example, render_prefix_for_completion

class Pairset(Dataset):
    def __init__(self, path, tok, max_len=2048):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.tok = tok
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def _pack(self, prefix_obj, answer):
        """Pack conversation prefix + answer into tokens.
        
        NEW: prefix_obj is now a full conversation object {"messages": [...]}
        """
        try:
            prefix_dict = ensure_prefix_dict(prefix_obj)
            prefix_ids = render_prefix_for_completion(self.tok, prefix_dict)
        except ValueError:
            prefix_ids = render_prefix_for_completion(self.tok, None)

        answer_ids = self.tok.encode(answer)
        assistant_end = self.tok.encode_special("<|assistant_end|>")
        return (prefix_ids + answer_ids + assistant_end)[: self.max_len]

    def __getitem__(self, i):
        r = self.rows[i]
        prefix = prefix_from_example(r)
        x_pos = self._pack(prefix, r["chosen"])
        x_neg = self._pack(prefix, r["rejected"])
        return torch.tensor(x_pos), torch.tensor(x_neg)

def collate(batch, pad_id):
    lens_pos = [len(b[0]) for b in batch]
    lens_neg = [len(b[1]) for b in batch]
    T = max(max(lens_pos), max(lens_neg))
    B = len(batch)
    x_pos = torch.full((B,T), pad_id, dtype=torch.long)
    x_neg = torch.full((B,T), pad_id, dtype=torch.long)
    attn_pos = torch.zeros((B,T), dtype=torch.bool)
    attn_neg = torch.zeros((B,T), dtype=torch.bool)
    for i,(p,n) in enumerate(batch):
        x_pos[i,:len(p)] = p; attn_pos[i,:len(p)] = True
        x_neg[i,:len(n)] = n; attn_neg[i,:len(n)] = True
    return x_pos, attn_pos, x_neg, attn_neg

def sequence_logprob(model, x, attn_mask, tok):
    # standard causal NLL over non-pad tokens, excluding first token
    logits = model(x)  # [B,T,V]
    logp = torch.log_softmax(logits, dim=-1)
    # shift for next-token
    tgt = x[:,1:].contiguous()
    lp = logp[:,:-1,:].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = attn_mask[:,1:].float()
    # average logprob per token
    return (lp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ckpt_in", required=True)
    ap.add_argument("--ckpt_out", required=True)
    ap.add_argument("--tokenizer", default="tokenizer.model")
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.tokenizer and os.path.isdir(args.tokenizer):
        tok = RustBPETokenizer.from_directory(args.tokenizer)
    else:
        if args.tokenizer and not os.path.isdir(args.tokenizer):
            print(f"Warning: tokenizer path '{args.tokenizer}' not found or not a directory; using default tokenizer")
        tok = get_tokenizer()

    ds = Pairset(args.pairs, tok)
    dl = DataLoader(ds, batch_size=args.bsz, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_id))

    ckpt = torch.load(args.ckpt_in, map_location="cpu")
    model_cfg = ckpt["model_cfg"]
    model = Transformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.1)

    beta = args.beta
    step = 0
    for ep in range(args.epochs):
        for x_pos, m_pos, x_neg, m_neg in tqdm(dl):
            x_pos, m_pos = x_pos.to(device), m_pos.to(device)
            x_neg, m_neg = x_neg.to(device), m_neg.to(device)
            # DPO objective (no ref model variant)
            lp_pos = sequence_logprob(model, x_pos, m_pos, tok)  # [B]
            lp_neg = sequence_logprob(model, x_neg, m_neg, tok)  # [B]
            # maximize: log sigma(beta*(lp_pos - lp_neg))
            pref_margin = beta * (lp_pos - lp_neg)
            loss = -torch.nn.functional.logsigmoid(pref_margin).mean()

            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            step += 1

    # save
    ckpt["model"] = model.state_dict()
    torch.save(ckpt, args.ckpt_out)
    print("wrote", args.ckpt_out)

if __name__ == "__main__":
    main()
