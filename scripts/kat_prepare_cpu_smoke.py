#!/usr/bin/env python3
"""
Prepare a tiny local CPU smoke-test environment for KAT scripts.

Creates under $NANOCHAT_BASE_DIR:
- tokenizer/tokenizer.pkl (tiktoken-backed RustBPETokenizer)
- chatsft_checkpoints/d2/model_000000.pt (+meta json)
- data/pairs_all.jsonl (small synthetic preference set)
"""

import json
import os
import random
import pickle
from pathlib import Path

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
import tiktoken


def ensure_tokenizer(base_dir: Path) -> None:
    tok_dir = base_dir / "tokenizer"
    tok_file = tok_dir / "tokenizer.pkl"
    if tok_file.exists():
        print(f"Tokenizer already exists: {tok_file}")
        return
    tok_dir.mkdir(parents=True, exist_ok=True)
    base_enc = tiktoken.get_encoding("gpt2")
    mergeable_ranks = dict(base_enc._mergeable_ranks)  # type: ignore[attr-defined]
    next_id = max(mergeable_ranks.values()) + 1
    special_tokens = dict(base_enc._special_tokens)  # type: ignore[attr-defined]
    for token in SPECIAL_TOKENS:
        if token not in special_tokens:
            special_tokens[token] = next_id
            next_id += 1
    enc = tiktoken.Encoding(
        name="nanochat-smoke",
        pat_str=base_enc._pat_str,  # type: ignore[attr-defined]
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    with tok_file.open("wb") as f:
        pickle.dump(enc, f)
    print(f"Created tokenizer: {tok_file}")


def ensure_sft_checkpoint(base_dir: Path) -> None:
    ckpt_dir = base_dir / "chatsft_checkpoints" / "d2"
    model_file = ckpt_dir / "model_000000.pt"
    meta_file = ckpt_dir / "meta_000000.json"
    if model_file.exists() and meta_file.exists():
        print(f"SFT checkpoint already exists: {model_file}")
        return

    tok = RustBPETokenizer.from_directory(str(base_dir / "tokenizer"))
    cfg = GPTConfig(
        sequence_len=256,
        vocab_size=tok.get_vocab_size(),
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
    )
    model = GPT(cfg)
    model.init_weights()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        str(ckpt_dir),
        0,
        model.state_dict(),
        None,
        {
            "step": 0,
            "model_config": cfg.__dict__,
            "note": "tiny cpu smoke checkpoint",
        },
    )
    print(f"Created tiny SFT checkpoint: {model_file}")


def ensure_pairs(base_dir: Path, n_rows: int = 128) -> None:
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pairs_file = data_dir / "pairs_all.jsonl"
    if pairs_file.exists():
        print(f"Pairs file already exists: {pairs_file}")
        return

    random.seed(7)
    prompts = [
        "Explain overfitting in simple terms.",
        "How does gradient descent work?",
        "What is reinforcement learning?",
        "Why do models hallucinate?",
        "Give three tips for studying math.",
        "What is batch normalization?",
        "Explain bias vs variance tradeoff.",
        "How do transformers use attention?",
    ]

    with pairs_file.open("w", encoding="utf-8") as f:
        for i in range(n_rows):
            p = random.choice(prompts)
            chosen = (
                f"A concise and helpful answer for '{p}' with concrete steps "
                f"and an example #{i}."
            )
            rejected = f"I do not know. Maybe search online. #{i}"
            if i % 3 == 0:
                row = {
                    "id": f"pair_{i}",
                    "prompt": p,
                    "chosen": chosen,
                    "rejected": rejected,
                    "src": "synthetic",
                }
            else:
                row = {
                    "id": f"pair_{i}",
                    "prefix": {"messages": [{"role": "user", "content": p}]},
                    "chosen": chosen,
                    "rejected": rejected,
                    "src": "synthetic",
                }
            f.write(json.dumps(row) + "\n")
    print(f"Created synthetic pairs: {pairs_file} ({n_rows} rows)")


def main() -> None:
    base_dir = Path(get_base_dir())
    print(f"Using NANOCHAT_BASE_DIR: {base_dir}")
    ensure_tokenizer(base_dir)
    ensure_sft_checkpoint(base_dir)
    ensure_pairs(base_dir)
    print("CPU smoke prep complete.")


if __name__ == "__main__":
    main()
