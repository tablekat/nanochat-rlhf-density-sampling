# scripts/embed.py
"""
python scripts/kat_embed.py \
  --ckpt out/ckpt.pt \
  --tokenizer tokenizer.model \
  --data data/my_conversations.jsonl \
  --out data/my_conversations_emb.pt \
  --batch_size 32
"""
import os, json, argparse, torch, ujson as jsonf, sys
from torch.nn.functional import normalize
from tqdm import tqdm

from nanochat.gpt import GPT
from nanochat.checkpoint_manager import load_model, get_base_dir
# ^ Model/tokenizer names match the repo; minimal codebase exposes them. :contentReference[oaicite:3]{index=3}

def batch(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf: yield buf

def to_chat_text(prompt, answer=None):
    # Harmony-ish chat tags. :contentReference[oaicite:4]{index=4}
    s = "<|bos|>\n<|user_start|>" + prompt + "<|user_end|>\n"
    if answer is not None:
        s += "<|assistant_start|>" + answer + "<|assistant_end|>\n"
    return s

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_source", default="sft", help="Model checkpoint source (base|mid|sft|grpo|rm)")
    ap.add_argument("--data", required=True, help="jsonl with fields: id, prompt, (optional) assistant")
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
        texts = [to_chat_text(r["prompt"], r.get("assistant")) for r in chunk]
        # tokenize using nanochat tokenizer
        toks = [tokenizer.encode(t) for t in texts]
        max_len = max(len(t) for t in toks)
        
        pad_token_id = tokenizer.encode_special("<|assistant_end|>")
        x = torch.full((len(toks), max_len), pad_token_id, dtype=torch.long, device=device)
        for i, t in enumerate(toks):
            x[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # forward pass to get hidden states
        # Use the new return_hidden_states parameter
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
