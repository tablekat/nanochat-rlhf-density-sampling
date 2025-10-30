#!/usr/bin/env python3
"""
Extract unique prefixes from pairs_all.jsonl and deduplicate them.

Input:
  $NANOCHAT_BASE_DIR/data/pairs_all.jsonl  (from scripts.kat_download_pairs)

Outputs:
  $NANOCHAT_BASE_DIR/data/prefixes_all.jsonl   [{"id","prefix"} ...]
  $NANOCHAT_BASE_DIR/data/prefix_id_map.tsv    id <TAB> first_user_message
  $NANOCHAT_BASE_DIR/data/stats.txt            counts

Deduplicates by first user message (md5 hash) but stores FULL prefix objects.
This allows full conversation context to flow through embedding and training.

Usage:
  python -m scripts.kat_make_prefixes
"""

import os, json, sys
from collections import Counter
from nanochat.common import get_base_dir

from scripts.kat_utils import (
    ensure_prefix_dict,
    first_user_message,
    norm_space,
    prefix_id_from_prefix,
)

base_dir = get_base_dir()
OUT_DIR  = os.path.join(base_dir, "data")
IN_PATH  = os.path.join(OUT_DIR, "pairs_all.jsonl")
PREFIXES = os.path.join(OUT_DIR, "prefixes_all.jsonl")
IDMAP    = os.path.join(OUT_DIR, "prefix_id_map.tsv")
STATS    = os.path.join(OUT_DIR, "stats.txt")

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
    uniq_prefixes = 0

    with open(PREFIXES, "w", encoding="utf-8") as fp, \
         open(IDMAP, "w", encoding="utf-8") as fm, \
         open(IN_PATH, "r", encoding="utf-8") as fin:

        for line in fin:
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping invalid JSON on line {total_pairs}: {e}")
                continue
            
            # Validate and extract prefix object
            try:
                if 'prefix' not in r:
                    print(f"⚠️  Warning: Skipping entry without valid 'prefix' field on line {total_pairs}")
                    continue
                
                prefix_obj = ensure_prefix_dict(r['prefix'])
                
                # Extract first user message as dedup key
                first_user_msg = first_user_message(prefix_obj)
                if not first_user_msg:
                    print(f"⚠️  Warning: No user message found in prefix on line {total_pairs}")
                    continue
                
                first_user_msg_norm = norm_space(first_user_msg)
                
            except Exception as e:
                print(f"⚠️  Warning: Error processing prefix on line {total_pairs}: {e}")
                continue
            
            s = r.get("src") or "unknown"
            by_src[s] += 1
            h = prefix_id_from_prefix(prefix_obj)
            if not h:
                print(f"⚠️  Warning: Could not compute prefix id on line {total_pairs}")
                total_pairs += 1
                continue
            
            # Skip if we've already seen this prefix (by first user message)
            if h in seen: 
                total_pairs += 1
                continue
            
            seen.add(h)
            uniq_prefixes += 1
            
            # Write FULL prefix object, not just extracted text
            fp.write(json.dumps({"id": h, "prefix": prefix_obj}, ensure_ascii=False) + "\n")
            fm.write(f"{h}\t{first_user_msg_norm}\n")
            total_pairs += 1

    # Validate output
    if total_pairs == 0:
        print(f"❌ Error: No pairs found in {IN_PATH}")
        sys.exit(1)
    
    if uniq_prefixes == 0:
        print(f"❌ Error: No unique prefixes extracted from {total_pairs} pairs")
        sys.exit(1)
    
    print(f"✓ Processed {total_pairs} total pairs")
    print(f"✓ Extracted {uniq_prefixes} unique prefixes")
    print(f"✓ Source breakdown: {dict(by_src)}")
    print(f"✓ Written to:")
    print(f"    {PREFIXES}")
    print(f"    {IDMAP}")

    # Write stats
    with open(STATS, "w", encoding="utf-8") as f:
        f.write(f"total_pairs: {total_pairs}\n")
        f.write(f"unique_prefixes: {uniq_prefixes}\n")
        f.write(f"dedup_ratio: {uniq_prefixes / total_pairs:.2%}\n")
        f.write(f"source_breakdown: {dict(by_src)}\n")

if __name__ == "__main__":
    main()
