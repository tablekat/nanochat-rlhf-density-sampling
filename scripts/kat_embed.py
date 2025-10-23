# scripts/embed.py
"""
python scripts/kat_embed.py \
  --ckpt out/ckpt.pt \
  --tokenizer tokenizer.model \
  --data data/my_conversations.jsonl \
  --out data/my_conversations_emb.pt \
  --batch_size 32
"""
import os, json, argparse, torch, ujson as jsonf
from torch.nn.functional import normalize
from tqdm import tqdm

from nanochat.model import Transformer  # nanochat model
from nanochat.tokenizer import Tokenizer  # nanochat BPE
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
    ap.add_argument("--ckpt", required=True, help="path to nanochat checkpoint (e.g., out/ckpt.pt)")
    ap.add_argument("--tokenizer", default="tokenizer.model", help="path to trained tokenizer")
    ap.add_argument("--data", required=True, help="jsonl with fields: id, prompt, (optional) assistant")
    ap.add_argument("--out", required=True, help="output .pt with {'ids':..., 'emb': tensor[N,d]}")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer & model
    tok = Tokenizer(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_cfg = ckpt["model_cfg"]
    model = Transformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # load data
    rows = [jsonf.loads(l) for l in open(args.data, "r", encoding="utf-8")]
    ids = [r["id"] for r in rows]

    embs = []
    for chunk in tqdm(batch(rows, args.batch_size)):
        texts = [to_chat_text(r["prompt"], r.get("assistant")) for r in chunk]
        # tokenize
        toks = tok.encode_batch(texts, add_bos=False, add_eos=True)  # repo’s tokenizer wrapper
        max_len = max(len(t) for t in toks)
        x = torch.full((len(toks), max_len), tok.pad_id, dtype=torch.long, device=device)
        for i,t in enumerate(toks):
            x[i,:len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # forward pass to get hidden states
        # model returns logits by default; we’ll expose hidden states via an API or small edit:
        # Add return_hidden=True path or access model.last_hidden if available in your build.
        # For clarity, assume model(x, return_hidden=True) -> (logits, hidden)
        logits, hidden = model(x, return_hidden=True)  # [B,T,V], [B,T,D]

        # mask out padding
        pad_mask = (x != tok.pad_id).unsqueeze(-1)  # [B,T,1]
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
