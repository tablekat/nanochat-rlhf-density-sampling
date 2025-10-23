#!/usr/bin/env python3
"""
Create a deduplicated prompt list from pairs_all.jsonl

Input:
  .cache/data/pairs_all.jsonl  (from scripts.kat_download_pairs)

Outputs:
  .cache/data/prompts_all.jsonl   [{"id","prompt"} ...]
  .cache/data/prompt_id_map.tsv   id <TAB> prompt
  .cache/data/stats.txt           counts

We use md5(prompt)[:16] as deterministic ID so we can join later.
"""

import os, json, hashlib, re
from collections import Counter

IN_PATH  = os.path.join(".cache", "data", "pairs_all.jsonl")
OUT_DIR  = os.path.join(".cache", "data")
PROMPTS  = os.path.join(OUT_DIR, "prompts_all.jsonl")
IDMAP    = os.path.join(OUT_DIR, "prompt_id_map.tsv")
STATS    = os.path.join(OUT_DIR, "stats.txt")

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def pid(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seen = set()
    by_src = Counter()
    total_pairs = 0
    uniq_prompts = 0

    with open(PROMPTS, "w", encoding="utf-8") as fp, \
         open(IDMAP, "w", encoding="utf-8") as fm, \
         open(IN_PATH, "r", encoding="utf-8") as fin:

        for line in fin:
            total_pairs += 1
            r = json.loads(line)
            p = norm_space(r["prompt"])
            s = r.get("src") or "unknown"
            by_src[s] += 1
            h = pid(p)
            if h in seen: 
                continue
            seen.add(h); uniq_prompts += 1
            fp.write(json.dumps({"id": h, "prompt": p}, ensure_ascii=False) + "\n")
            fm.write(f"{h}\t{p}\n")

    with open(STATS, "w", encoding="utf-8") as fs:
        fs.write(f"total_pairs\t{total_pairs}\n")
        fs.write(f"unique_prompts\t{uniq_prompts}\n")
        for k,v in by_src.items():
            fs.write(f"src_{k}\t{v}\n")

    print(f"Wrote {PROMPTS}, {IDMAP}, and {STATS}")

if __name__ == "__main__":
    main()
