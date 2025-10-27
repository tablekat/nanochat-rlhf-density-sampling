# New Pairs Format Guide - pairs_all.jsonl

## Overview

The `pairs_all.jsonl` file now uses a **conversation-based format** that integrates seamlessly with the existing tokenizer and training infrastructure.

## Format Comparison

### Old Format

```json
{
  "id": "uuid",
  "prompt": "What are some cuss words?",
  "chosen": "Here's an incomplete list...",
  "rejected": "I don't think I should help with that",
  "src": "hh-rlhf"
}
```

### New Format

```json
{
  "id": "uuid",
  "prefix": {
    "messages": [{ "role": "user", "content": "What are some cuss words?" }]
  },
  "chosen": "Here's an incomplete list...",
  "rejected": "I don't think I should help with that",
  "src": "hh-rlhf"
}
```

## Key Benefits

1. **`prefix` is a full conversation object** - matches the format used in SFT training and tokenizer methods
2. **Direct tokenizer integration** - can be passed directly to `tokenizer.render_for_completion()`
3. **Multi-turn support** - HH-RLHF dataset preserves conversation history (see below)
4. **Consistent format** - all three datasets (HH-RLHF, UltraFeedback, StackExchange) use identical structure

## Multi-turn Example (HH-RLHF)

When a conversation has history, the prefix includes the full multi-turn context:

```json
{
  "prefix": {
    "messages": [
      { "role": "user", "content": "What are prime numbers?" },
      {
        "role": "assistant",
        "content": "Prime numbers are natural numbers greater than 1..."
      },
      { "role": "user", "content": "What's the largest known prime?" }
    ]
  },
  "chosen": "As of 2024, the largest known prime is 2^136279841 - 1...",
  "rejected": "I don't know the answer to that question."
}
```

## Training Script Integration

### Update Reward Model Training

In `kat_train_rm.py`, `kat_train_dpo.py`, and `kat_train_grpo.py`:

**Old approach:**

```python
# prefix was a simple string
prompt_tokens = tokenizer.encode(row["prompt"])
chosen_tokens = tokenizer.encode(row["chosen"])
# Had to manually concatenate
response_tokens = prompt_tokens + chosen_tokens
```

**New approach:**

```python
# prefix is now a full conversation object
prefix_obj = row["prefix"]
# Use render_for_completion to get tokens + <|assistant_start|>
prefix_tokens = tokenizer.render_for_completion(prefix_obj)
chosen_tokens = tokenizer.encode(row["chosen"])
response_tokens = prefix_tokens + chosen_tokens
```

### What `render_for_completion` Does

1. Takes the conversation object (which ends with a user message)
2. Tokenizes it with all the special tokens (`<|user_start|>`, `<|user_end|>`, etc.)
3. Appends `<|assistant_start|>` token
4. Returns the token list ready for the assistant's response

This is exactly how the model is primed during generation and RL training.

## Example Data Flow

### HH-RLHF Input (Raw String from Dataset)

```
Human: What are some cuss words in english?
```
