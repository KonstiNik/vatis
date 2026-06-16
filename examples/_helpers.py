"""Shared utilities for vatis example scripts."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

# A "chat sample" is a list of OpenAI-style message dicts, e.g.
# ``[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]``.
ChatSample = list[dict[str, str]]


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest shared prefix of two token-id lists."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _render_full_and_prompt_len(
    messages: ChatSample, tokenizer: AutoTokenizer
) -> tuple[list[int], int]:
    """Render the conversation and locate the prompt/completion boundary.

    transformers 5.x returns a ``BatchEncoding`` from
    ``apply_chat_template(tokenize=True)``, not a flat ``list[int]``. To stay
    robust across versions we render to *text* first (``tokenize=False``) and
    tokenize the string with ``add_special_tokens=False`` (the template already
    emits the special / role tokens). This reproduces the tokenized template
    exactly while giving plain id lists to diff.

    Returns ``(full_ids, prompt_len)`` where ``prompt_len`` is the longest
    common prefix of the prompt-only (``messages[:-1]``, ``add_generation_prompt
    =True``) and full renderings — the token index at which the assistant's
    completion content begins. The common-prefix rule is robust to templates
    that re-emit the generation header slightly differently in the two passes.
    """
    full_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return full_ids, _common_prefix_len(prompt_ids, full_ids)


def sft_prompt_completion_lengths(
    messages: ChatSample,
    tokenizer: AutoTokenizer,
) -> tuple[int, int]:
    """Return ``(prompt_len, total_len)`` in tokens for an SFT chat sample.

    ``prompt_len`` is the number of leading tokens that belong to the prompt
    (everything up to and including the final assistant header, i.e. the point
    where the assistant's *content* begins); ``total_len`` is the length of the
    full rendered conversation. Used by ``prefetch.py`` to keep only samples
    that will still have real completion tokens after truncation to ``seq_len``.

    The boundary is computed as the longest common prefix between the
    prompt-only and full renderings (see :func:`_render_full_and_prompt_len`).
    """
    full_ids, prompt_len = _render_full_and_prompt_len(messages, tokenizer)
    return prompt_len, len(full_ids)


def tokenize_chat_sft_sample(
    messages: ChatSample,
    tokenizer: AutoTokenizer,
    *,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """Render one chat sample into a fixed-length, completion-only LM example.

    This is the *realistic* SFT objective: cross-entropy is computed only on
    the assistant's response tokens; the prompt (system + user turns, plus the
    assistant header) is masked out with ``-100``. Contrast with
    :func:`tokenize_into_lm_batch`, which trains on every token of a flat text.

    The model's own chat template defines the prompt/completion boundary. We
    render the conversation twice — full and prompt-only — and take the common
    prefix as the boundary (see :func:`sft_prompt_completion_lengths`).

    Labels follow the **pre-shifted** HF-causal-LM convention that vatis expects
    (``causal_lm_loss`` compares ``logits[i]`` against ``labels[i]`` with no
    further shift): ``labels[i] = ids[i + 1]`` when ``ids[i + 1]`` is a
    completion token, else ``-100``; ``labels[-1] = -100``.

    The example is then right-truncated (keeping the prompt prefix and as much
    of the completion as fits) or right-padded to exactly ``seq_len``. Padding
    uses ``pad_token_id`` (falling back to ``eos_token_id``) with
    ``attention_mask = 0`` and ``labels = -100`` so pads count toward neither
    the loss nor the valid-token denominator.

    Returns a dict of ``(seq_len,)`` tensors: ``input_ids``, ``attention_mask``,
    ``labels``. Stack several with :func:`stack_lm_batch` to form a batch.
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")

    full_ids, prompt_len = _render_full_and_prompt_len(messages, tokenizer)

    ids = torch.tensor(full_ids, dtype=torch.long)
    total = ids.shape[0]

    # Pre-shifted next-token targets: labels[i] = ids[i + 1].
    labels = torch.full((total,), fill_value=-100, dtype=torch.long)
    labels[:-1] = ids[1:]
    # Mask every target that lands inside the prompt. Position i predicts
    # ids[i + 1]; that target is a completion token iff i + 1 >= prompt_len,
    # i.e. i >= prompt_len - 1. So positions [0, prompt_len - 2] are masked.
    if prompt_len >= 1:
        labels[: prompt_len - 1] = -100

    attention_mask = torch.ones(total, dtype=torch.long)

    # Truncate (keep the prefix) or pad on the right to exactly seq_len.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    if total >= seq_len:
        ids = ids[:seq_len].contiguous()
        labels = labels[:seq_len].contiguous()
        attention_mask = attention_mask[:seq_len].contiguous()
    else:
        pad = seq_len - total
        ids = torch.cat([ids, torch.full((pad,), pad_id, dtype=torch.long)])
        labels = torch.cat([labels, torch.full((pad,), -100, dtype=torch.long)])
        attention_mask = torch.cat([attention_mask, torch.zeros(pad, dtype=torch.long)])

    return {"input_ids": ids, "attention_mask": attention_mask, "labels": labels}


def stack_lm_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of equal-length single-example dicts into a ``(B, S)`` batch."""
    if not samples:
        raise ValueError("cannot stack an empty list of samples")
    return {key: torch.stack([s[key] for s in samples], dim=0) for key in samples[0]}


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
