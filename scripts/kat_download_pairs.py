#!/usr/bin/env python3
"""
Download open pairwise preference datasets and write a single JSONL:
  $NANOCHAT_BASE_DIR/data/pairs_all.jsonl

Each line format:
  {
    "id": uuid,
    "prefix": {"messages": [{"role": "user", "content": "..."}, ...]},  # conversation ending with user message
    "chosen": "assistant response string",
    "rejected": "assistant response string",
    "src": "hh-rlhf|ultrafeedback-binarized|stack-exchange-preferences"
  }

The "prefix" field is a full conversation object that can be directly used with:
  - tokenizer.render_conversation() to get tokens for the full conversation
  - tokenizer.render_for_completion() to get tokens up to assistant_start token
  - reward model training to score chosen/rejected completions

Datasets included (defaults):
  - Anthropic/hh-rlhf
  - HuggingFaceH4/ultrafeedback_binarized
  - HuggingFaceH4/stack-exchange-preferences

Usage:
  python -m scripts.kat_download_pairs            # default (all)
  python -m scripts.kat_download_pairs --no-hh    # skip Anthropic HH
  python -m scripts.kat_download_pairs --only hh  # only HH
"""

import argparse, os, uuid, re, json, sys
from datasets import load_dataset
from nanochat.common import get_base_dir

from scripts.kat_utils import norm_space, prefix_id_from_prefix

base_dir = get_base_dir()
OUT_DIR = os.path.join(base_dir, "data")
OUT_PATH = os.path.join(OUT_DIR, "pairs_all.jsonl")

def write_jsonl(rows, fout):
    for r in rows:
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")

def from_hh():
    ds = load_dataset("Anthropic/hh-rlhf")
    # ds is jsonl of {"chosen", "rejected"} keys where each is a string like the following:
    #```
    # Human: What are some cuss words in english?
    # Assistant: Here's an incomplete list.
    # Ass ... (etc)
    # Human: What's your favorite one?
    # Assistant: Ass.
    #```
    
    def parse_conversation(text: str) -> list:
        """Parse conversation string with Human:/Assistant: markers into exchanges."""
        exchanges = []
        lines = text.split("\n")
        current_role = None
        current_text = []
        
        for line in lines:
            line = line.rstrip()
            
            # Check for role markers
            if line.startswith("Human:"):
                # Start of human message
                if current_role == "A" and current_text:
                    # Store previous assistant message
                    exchanges.append(("A", " ".join(current_text).strip()))
                    current_text = []
                current_role = "H"
                # Extract text after the marker
                current_text.append(line[6:].strip())
            elif line.startswith("Assistant:"):
                # Start of assistant message
                if current_role == "H" and current_text:
                    # Store previous human message
                    exchanges.append(("H", " ".join(current_text).strip()))
                    current_text = []
                current_role = "A"
                # Extract text after the marker
                current_text.append(line[10:].strip())
            elif current_role and line.strip():
                # Continuation of current message
                current_text.append(line.strip())
        
        # Store final message
        if current_role and current_text:
            exchanges.append((current_role, " ".join(current_text).strip()))
        
        return exchanges
    
    def extract_pairs_and_prefix(chosen_text: str, rejected_text: str):
        """
        Extract conversations and return:
        - prefix: full conversation ending with final user message (as conversation object)
        - chosen_response: assistant's chosen response (string)
        - rejected_response: assistant's rejected response (string)
        """
        chosen_exchanges = parse_conversation(chosen_text)
        rejected_exchanges = parse_conversation(rejected_text)
        
        if not chosen_exchanges or not rejected_exchanges:
            return None, None, None
        
        # Extract responses from chosen/rejected conversations
        chosen_response = None
        rejected_response = None
        
        # Find last assistant response in each
        for role, content in reversed(chosen_exchanges):
            if role == "A" and chosen_response is None:
                chosen_response = content
                break
        
        for role, content in reversed(rejected_exchanges):
            if role == "A" and rejected_response is None:
                rejected_response = content
                break
        
        if not chosen_response or not rejected_response:
            return None, None, None
        
        # Build prefix from exchanges, alternating user/assistant
        # We want to keep history up to the last user message before the final response
        # Extract alternating user/assistant messages
        messages = []
        for role, content in chosen_exchanges:
            if role == "H":
                messages.append({"role": "user", "content": norm_space(content)})
            elif role == "A":
                messages.append({"role": "assistant", "content": norm_space(content)})
        
        # Remove the final assistant message if present (we want prefix to end with user)
        if messages and messages[-1]["role"] == "assistant":
            messages.pop()
        
        # Ensure we have at least a user message
        if not messages or messages[-1]["role"] != "user":
            return None, None, None
        
        prefix = {"messages": messages}
        return prefix, chosen_response, rejected_response

    for split in ("train", "test"):
        for r in ds[split]:
            chosen_text = r.get("chosen") or ""
            rejected_text = r.get("rejected") or ""
            
            prefix, chosen_response, rejected_response = extract_pairs_and_prefix(chosen_text, rejected_text)
            
            if prefix and chosen_response and rejected_response:
                row = {
                    "id": str(uuid.uuid4()),
                    "prefix": prefix,
                    "chosen": norm_space(chosen_response),
                    "rejected": norm_space(rejected_response),
                    "src": "hh-rlhf",
                }
                prefix_id = prefix_id_from_prefix(prefix)
                if prefix_id:
                    row["prefix_id"] = prefix_id
                yield row

def from_ultrafeedback_binarized():
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    for r in ds["train"]:
        p = r.get("prompt")
        c = r.get("chosen")
        rej = r.get("rejected")
        if p and c and rej:
            prefix = {
                "messages": [
                    {"role": "user", "content": norm_space(p)}
                ]
            }
            row = {
                "id": str(uuid.uuid4()),
                "prefix": prefix,
                "chosen": norm_space(c),
                "rejected": norm_space(rej),
                "src": "ultrafeedback-binarized",
            }
            prefix_id = prefix_id_from_prefix(prefix)
            if prefix_id:
                row["prefix_id"] = prefix_id
            yield row

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
            prefix = {
                "messages": [
                    {"role": "user", "content": strip_html(q)}
                ]
            }
            row = {
                "id": str(uuid.uuid4()),
                "prefix": prefix,
                "chosen": norm_space(a_win),
                "rejected": norm_space(a_lose),
                "src": "stack-exchange-preferences",
            }
            prefix_id = prefix_id_from_prefix(prefix)
            if prefix_id:
                row["prefix_id"] = prefix_id
            yield row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["hh","uf","se"], help="download only this dataset")
    ap.add_argument("--no-hh", action="store_true", help="skip Anthropic HH-RLHF")
    ap.add_argument("--no-uf", action="store_true", help="skip UltraFeedback-binarized")
    ap.add_argument("--no-se", action="store_true", help="skip StackExchange-preferences")
    ap.add_argument("--out", default=OUT_PATH, help="output JSONL path")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Downloading preference pair datasets...")
    print(f"Output: {args.out}")
    
    cnt = 0
    with open(args.out, "w", encoding="utf-8") as f:
        if args.only in (None, "hh") and not args.no_hh:
            print("  - Downloading Anthropic/hh-rlhf...")
            for row in from_hh():
                cnt += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if cnt % 1000 == 0:
                    print(f"    {cnt} pairs downloaded...")
        
        if args.only in (None, "uf") and not args.no_uf:
            print("  - Downloading HuggingFaceH4/ultrafeedback_binarized...")
            for row in from_ultrafeedback_binarized():
                cnt += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if cnt % 1000 == 0:
                    print(f"    {cnt} pairs downloaded...")
        
        if args.only in (None, "se") and not args.no_se:
            print("  - Downloading HuggingFaceH4/stack-exchange-preferences...")
            for row in from_stack_exchange_prefs():
                cnt += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if cnt % 1000 == 0:
                    print(f"    {cnt} pairs downloaded...")
    
    # Validate output
    if cnt == 0:
        print(f"\n❌ Error: No pairs downloaded!")
        print(f"Check internet connection and HuggingFace dataset availability")
        sys.exit(1)
    
    if cnt < 1000:
        print(f"\n⚠️  Warning: Only {cnt} pairs downloaded (expected >10,000)")
    
    print(f"\n✓ Successfully wrote {args.out} with {cnt} pairs")

if __name__ == "__main__":
    main()
