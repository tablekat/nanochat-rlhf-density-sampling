"""Common utilities for KAT density-aware experiments.

This module centralizes prefix normalization, id generation, and
conversation rendering helpers so every stage of the pipeline agrees on
how to treat conversation data.
"""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Dict, List, Optional


__all__ = [
    "norm_space",
    "ensure_prefix_dict",
    "first_user_message",
    "prefix_id_from_text",
    "prefix_id_from_prefix",
    "prefix_from_example",
    "render_prefix_for_completion",
]


_WHITESPACE_RE = re.compile(r"\s+")


def norm_space(text: str) -> str:
    """Collapse whitespace so identical prompts hash to the same id."""

    if text is None:
        return ""
    return _WHITESPACE_RE.sub(" ", text.strip())


def ensure_prefix_dict(prefix: Any) -> Dict[str, Any]:
    """Coerce legacy prompt formats into the canonical prefix dict."""

    if isinstance(prefix, dict) and "messages" in prefix:
        # Guarantee we never mutate the caller's structure downstream.
        return copy.deepcopy(prefix)

    if isinstance(prefix, str):
        prefix = norm_space(prefix)
        return {"messages": [{"role": "user", "content": prefix}]}

    if prefix is None:
        return {"messages": [{"role": "user", "content": ""}]}

    raise ValueError(f"Unsupported prefix format: {type(prefix)}")


def first_user_message(prefix: Dict[str, Any]) -> str:
    """Return the first user message content from a prefix dict."""

    for message in prefix.get("messages", []):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def prefix_id_from_text(text: str) -> Optional[str]:
    """Compute the canonical md5-based prefix id from text."""

    text = norm_space(text)
    if not text:
        return None
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return digest[:16]


def prefix_id_from_prefix(prefix: Any) -> Optional[str]:
    """Compute the canonical prefix id from any prefix representation."""

    prefix_dict = ensure_prefix_dict(prefix)
    return prefix_id_from_text(first_user_message(prefix_dict))


def prefix_from_example(example: Any) -> Any:
    """Extract prefix payload from an example dict (supports legacy 'prompt')."""

    if isinstance(example, dict):
        if example.get("prefix") is not None:
            return example.get("prefix")
        if example.get("prompt") is not None:
            return example.get("prompt")
    return None


def render_prefix_for_completion(tokenizer, prefix: Any) -> List[int]:
    """Render a conversation prefix into completion-ready token ids.

    This accepts prefixes that end in a user turn (our preferred format)
    while remaining compatible with legacy prefixes that already include
    an assistant reply. The returned token ids always terminate with a
    single ``<|assistant_start|>`` token so the model is primed for the
    next assistant response.
    """

    prefix_dict = ensure_prefix_dict(prefix)
    messages = prefix_dict.setdefault("messages", [])

    if not messages:
        messages.append({"role": "user", "content": ""})

    last_role = messages[-1].get("role")

    if last_role == "assistant":
        # Leverage the existing helper which already drops the final
        # assistant turn and appends <|assistant_start|>.
        return tokenizer.render_for_completion(prefix_dict)

    # Otherwise render the conversation as-is and manually append the
    # assistant start token so the model begins generating the reply.
    ids, _mask = tokenizer.render_conversation(prefix_dict)
    ids.append(tokenizer.encode_special("<|assistant_start|>"))
    return ids


