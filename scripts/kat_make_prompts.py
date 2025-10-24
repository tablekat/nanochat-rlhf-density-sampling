#!/usr/bin/env python3
"""
Create a deduplicated prompt list from pairs_all.jsonl

Input:
  $NANOCHAT_BASE_DIR/data/pairs_all.jsonl  (from scripts.kat_download_pairs)

Outputs:
  $NANOCHAT_BASE_DIR/data/prompts_all.jsonl   [{"id","prompt"} ...]
  $NANOCHAT_BASE_DIR/data/prompt_id_map.tsv   id <TAB> prompt
  $NANOCHAT_BASE_DIR/data/stats.txt           counts

We use md5(prompt)[:16] as deterministic ID so we can join later.
"""

import os, json, hashlib, re, sys
from collections import Counter
from nanochat.common import get_base_dir

base_dir = get_base_dir()
OUT_DIR  = os.path.join(base_dir, "data")
IN_PATH  = os.path.join(OUT_DIR, "pairs_all.jsonl")
PROMPTS  = os.path.join(OUT_DIR, "prompts_all.jsonl")
IDMAP    = os.path.join(OUT_DIR, "prompt_id_map.tsv")
STATS    = os.path.join(OUT_DIR, "stats.txt")

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def pid(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16]

def main():
    # Validate input file exists
    if not os.path.exists(IN_PATH):
        print(f"❌ Error: Input file not found: {IN_PATH}")
        print(f"Run kat_download_pairs first:")
        print(f"  python -m scripts.kat_download_pairs")
        sys.exit(1)
    
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
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping invalid JSON on line {total_pairs}: {e}")
                continue
            
            # Validate required field
            if 'prompt' not in r:
                print(f"⚠️  Warning: Skipping entry without 'prompt' field on line {total_pairs}")
                continue
                
            p = norm_space(r["prompt"])
            s = r.get("src") or "unknown"
            by_src[s] += 1
            h = pid(p)
            if h in seen: 
                continue
            seen.add(h)
            uniq_prompts += 1
            fp.write(json.dumps({"id": h, "prompt": p}, ensure_ascii=False) + "\n")
            fm.write(f"{h}\t{p}\n")

    # Validate output
    if total_pairs == 0:
        print(f"❌ Error: No pairs found in {IN_PATH}")
        sys.exit(1)
    
    if uniq_prompts == 0:
        print(f"❌ Error: No unique prompts extracted from {total_pairs} pairs")
        sys.exit(1)
    
    print(f"✓ Processed {total_pairs} total pairs")
    print(f"✓ Extracted {uniq_prompts} unique prompts")
    print(f"✓ Source breakdown: {dict(by_src)}")
    print(f"✓ Written to:")
    print(f"    {PROMPTS}")
    print(f"    {IDMAP}")

    # Write stats
    with open(STATS, "w", encoding="utf-8") as f:
        f.write(f"total_pairs: {total_pairs}\n")
        f.write(f"unique_prompts: {uniq_prompts}\n")
        f.write(f"dedup_ratio: {uniq_prompts / total_pairs:.2%}\n")
        f.write(f"source_breakdown: {dict(by_src)}\n")

if __name__ == "__main__":
    main()
