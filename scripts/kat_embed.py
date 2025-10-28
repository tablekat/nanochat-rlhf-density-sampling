# scripts/kat_embed.py
"""
Compute embeddings for prefixes or prompts.

Can handle either:
- prefixes_all.jsonl with full prefix conversation objects
- Legacy prompts_all.jsonl with simple prompt strings

Usage:
  python scripts/kat_embed.py \\
    --ckpt_source sft \\
    --data data/prefixes_all.jsonl \\
    --out data/prefixes_emb.pt \\
    --batch_size 32
"""
import os, json, argparse, torch, ujson as jsonf, sys
from torch.nn.functional import normalize
from tqdm import tqdm

from nanochat.gpt import GPT
from nanochat.checkpoint_manager import load_model, get_base_dir
from nanochat.tokenizer import get_tokenizer

def batch(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf: yield buf

def extract_text_from_row(row):
    """Extract text for embedding from either prefix object or legacy prompt string."""
    # New format: prefix is a full conversation object
    if 'prefix' in row and isinstance(row['prefix'], dict):
        prefix = row['prefix']
        messages = prefix.get('messages', [])
        # Concatenate all messages for embedding context
        text_parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if content:
                text_parts.append(f"{role}: {content}")
        if text_parts:
            return "\n".join(text_parts)
    
    # Legacy format: direct prompt field
    if 'prompt' in row:
        return row['prompt']
    
    return ""

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_source", default="sft", help="Model checkpoint source (base|mid|sft|grpo|rm)")
    ap.add_argument("--data", required=True, help="jsonl with fields: id, prefix (or prompt)")
    ap.add_argument("--out", required=True, help="output .pt with {'ids':..., 'emb': tensor[N,d]}")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model and tokenizer using checkpoint_manager
    print(f"Loading model from {args.ckpt_source} checkpoints...")
    try:
        model, tokenizer, meta = load_model(args.ckpt_source, device, phase="eval")
        model.eval()
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # load data
    print(f"Loading data from {args.data}...")
    rows = [json.loads(l) for l in open(args.data, "r", encoding="utf-8")]
    ids = [r["id"] for r in rows]
    print(f"✓ Loaded {len(rows)} rows")

    embs = []
    for chunk in tqdm(batch(rows, args.batch_size)):
        # Extract text from either prefix or prompt format
        texts = [extract_text_from_row(r) for r in chunk]
        
        # tokenize using nanochat tokenizer
        toks = [tokenizer.encode(t) for t in texts]
        max_len = max(len(t) for t in toks)
        
        pad_token_id = tokenizer.encode_special("<|assistant_end|>")
        x = torch.full((len(toks), max_len), pad_token_id, dtype=torch.long, device=device)
        for i, t in enumerate(toks):
            x[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # forward pass to get hidden states
        output = model(x, return_hidden_states=True)
        logits = output['logits']  # [B,T,V]
        hidden = output['hidden_states']  # [B,T,D]

        # mask out padding
        pad_mask = (x != pad_token_id).unsqueeze(-1)  # [B,T,1]
        h = hidden * pad_mask
        h = h.sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)
        # L2 normalize for cosine
        h = normalize(h, dim=-1)
        embs.append(h.cpu())

    embs = torch.cat(embs, dim=0)
    torch.save({"ids": ids, "emb": embs}, args.out)
    print(f"wrote {args.out}, shape={tuple(embs.shape)}")

if __name__ == "__main__":
    main()
