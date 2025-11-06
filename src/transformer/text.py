"""Text tokenizer transformer.

Loads a HF tokenizer once at startup and reuses it. Returns OpenAI-style
chat payloads when the predictor is a vLLM server, else simple token ids.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        name = os.environ.get("TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
        logger.info("loading tokenizer %s", name)
        _TOKENIZER = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
    return _TOKENIZER


def tokenize(prompt: str, max_len: int = 4096) -> dict[str, Any]:
    tk = get_tokenizer()
    enc = tk(prompt, truncation=True, max_length=max_len, return_tensors=None)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc.get("attention_mask"),
    }


def to_vllm_chat_payload(
    prompt: str,
    *,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
