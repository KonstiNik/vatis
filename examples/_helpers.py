"""Shared utilities for vatis example scripts."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer


def tokenize_into_lm_batch(
    text: str,
    tokenizer: AutoTokenizer,
    *,
    batch_size: int,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """Tokenize ``text`` and reshape into a ``(batch_size, seq_len)`` LM batch.

    The text is tokenized with no special tokens, then truncated or
    cyclically repeated to produce exactly ``batch_size * seq_len``
    tokens, which are reshaped row-major into the batch. Labels are the
    HF causal-LM convention: ``labels[:, :-1] = input_ids[:, 1:]``,
    ``labels[:, -1] = -100`` so the last position of every sequence is
    ignored (no next-token target available there).
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]  # (T,)
    needed = batch_size * seq_len

    # If the text is shorter than needed, repeat it.
    while ids.shape[0] < needed:
        ids = torch.cat([ids, ids], dim=0)

    ids = ids[:needed].reshape(batch_size, seq_len).contiguous()
    attention_mask = torch.ones_like(ids)
    labels = torch.full_like(ids, fill_value=-100)
    labels[:, :-1] = ids[:, 1:]
    return {
        "input_ids": ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def step_from_revision(rev: str) -> int:
    """Parse the trailing integer step out of a Pythia-style revision tag."""
    return int(rev.removeprefix("step"))
