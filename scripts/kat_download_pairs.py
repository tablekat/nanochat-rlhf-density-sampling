#!/usr/bin/env python3
"""
Download open pairwise preference datasets and write a single JSONL:
  .cache/data/pairs_all.jsonl
Each line: {"id","prompt","chosen","rejected"}

Datasets included (defaults):
  - Anthropic/hh-rlhf
  - HuggingFaceH4/ultrafeedback_binarized
  - HuggingFaceH4/stack-exchange-preferences

Usage:
  python -m scripts.kat_download_pairs            # default (all)
  python -m scripts.kat_download_pairs --no-hh    # skip Anthropic HH
  python -m scripts.kat_download_pairs --only hh  # only HH
"""

import argparse, os, uuid, re, json
from datasets import load_dataset

OUT_DIR = os.path.join(".cache", "data")
OUT_PATH = os.path.join(OUT_DIR, "pairs_all.jsonl")

def norm_space(s: str) -> str:
    # normalize whitespace for better dedup/join later
    return re.sub(r"\s+", " ", s.strip())

def write_jsonl(rows, fout):
    for r in rows:
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")

def from_hh():
    ds = load_dataset("Anthropic/hh-rlhf")
    def extract_first_pair(conv):
        prompt = answer = None
        for m in conv:
            role = (m.get("role") or "").lower()
            text = m.get("text") or ""
            if role == "human" and prompt is None:
                prompt = text
            elif role == "assistant" and answer is None:
                answer = text
        return prompt, answer

    for split in ("train", "test"):
        for r in ds[split]:
            p1,a1 = extract_first_pair(r["chosen"])
            p2,a2 = extract_first_pair(r["rejected"])
            if p1 and a1 and p2 and a2 and norm_space(p1) == norm_space(p2):
                yield {
                    "id": str(uuid.uuid4()),
                    "prompt": norm_space(p1),
                    "chosen": norm_space(a1),
                    "rejected": norm_space(a2),
                    "src": "hh-rlhf",
                }

def from_ultrafeedback_binarized():
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    for r in ds["train"]:
        p = r.get("prompt")
        c = r.get("chosen")
        rej = r.get("rejected")
        if p and c and rej:
            yield {
                "id": str(uuid.uuid4()),
                "prompt": norm_space(p),
                "chosen": norm_space(c),
                "rejected": norm_space(rej),
                "src": "ultrafeedback-binarized",
            }

def strip_html(s: str) -> str:
    # light HTML -> text for stack-exchange questions
    s = re.sub(r"<\s*br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return norm_space(s)

def from_stack_exchange_prefs():
    ds = load_dataset("HuggingFaceH4/stack-exchange-preferences")
    for r in ds["train"]:
        q = (r.get("question") or {}).get("body")
        a_win = r.get("winner") or r.get("chosen") or r.get("answer_0")
        a_lose = r.get("loser") or r.get("rejected") or r.get("answer_1")
        if q and a_win and a_lose:
            yield {
                "id": str(uuid.uuid4()),
                "prompt": strip_html(q),
                "chosen": norm_space(a_win),
                "rejected": norm_space(a_lose),
                "src": "stack-exchange-preferences",
            }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["hh","uf","se"], help="download only this dataset")
    ap.add_argument("--no-hh", action="store_true", help="skip Anthropic HH-RLHF")
    ap.add_argument("--no-uf", action="store_true", help="skip UltraFeedback-binarized")
    ap.add_argument("--no-se", action="store_true", help="skip StackExchange-preferences")
    ap.add_argument("--out", default=OUT_PATH, help="output JSONL path")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    cnt = 0
    with open(args.out, "w", encoding="utf-8") as f:
        if args.only in (None, "hh") and not args.no_hh:
            for row in from_hh():
                cnt += 1; f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.only in (None, "uf") and not args.no_uf:
            for row in from_ultrafeedback_binarized():
                cnt += 1; f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.only in (None, "se") and not args.no_se:
            for row in from_stack_exchange_prefs():
                cnt += 1; f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out} with {cnt} pairs")

if __name__ == "__main__":
    main()
