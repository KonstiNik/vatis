"""Micro-batch splitting and shape utilities.

A "batch" in vatis can be one of:
    - a torch.Tensor of shape ``(B, ...)`` (input ids only; targets inferred);
    - a dict of tensors with at least an ``input_ids``-like key, optionally
      ``attention_mask`` and ``labels``;
    - a tuple/list of tensors interpreted as ``(input, target)``.

We need a single helper that can:
    1. report the batch dimension size,
    2. iterate over micro-batches of a given size.

This module knows nothing about the model — it only deals with the
``batch_size = leading-dim`` invariant.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import torch

# A batch is one of these — we treat them all uniformly via batch_size and
# slicing helpers.
Batch = torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor]


def batch_size(batch: Batch) -> int:
    """Return the leading-dim size of any supported batch object."""
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0])
    if isinstance(batch, Mapping):
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return int(v.shape[0])
        raise ValueError("dict batch contains no tensors")
    if isinstance(batch, Sequence):
        for v in batch:
            if isinstance(v, torch.Tensor):
                return int(v.shape[0])
        raise ValueError("sequence batch contains no tensors")
    raise TypeError(f"unsupported batch type: {type(batch).__name__}")


def slice_batch(batch: Batch, start: int, stop: int) -> Batch:
    """Slice a batch along its leading dim into the half-open range [start, stop)."""
    if isinstance(batch, torch.Tensor):
        return batch[start:stop]
    if isinstance(batch, Mapping):
        return {k: (v[start:stop] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    if isinstance(batch, Sequence):
        return type(
            batch
        )(  # type: ignore[call-arg]
            v[start:stop] if isinstance(v, torch.Tensor) else v for v in batch
        )
    raise TypeError(f"unsupported batch type: {type(batch).__name__}")


def iter_micro_batches(batch: Batch, micro_batch_size: int) -> Iterator[tuple[int, int, Batch]]:
    """Yield ``(start, stop, micro_batch)`` over the leading batch dim.

    The last micro-batch may be smaller than ``micro_batch_size`` if the
    batch is not evenly divisible.

    Args:
        batch: any supported batch object.
        micro_batch_size: how many examples per micro-batch. Must be >= 1.

    Yields:
        Tuples ``(start, stop, micro_batch)`` where ``stop - start <=
        micro_batch_size``.
    """
    if micro_batch_size < 1:
        raise ValueError(f"micro_batch_size must be >= 1, got {micro_batch_size}")
    n = batch_size(batch)
    for start in range(0, n, micro_batch_size):
        stop = min(start + micro_batch_size, n)
        yield start, stop, slice_batch(batch, start, stop)


def move_batch(batch: Batch, device: torch.device | str) -> Batch:
    """Move all tensor leaves of a batch to ``device``. Non-tensor leaves
    pass through unchanged."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device)
    if isinstance(batch, Mapping):
        return {
            k: (v.to(device=device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
        }
    if isinstance(batch, Sequence):
        return type(batch)(  # type: ignore[call-arg]
            v.to(device=device) if isinstance(v, torch.Tensor) else v for v in batch
        )
    raise TypeError(f"unsupported batch type: {type(batch).__name__}")


def extract_targets(batch: Batch) -> torch.Tensor | None:
    """Best-effort extraction of the target tensor from a batch.

    For an HF causal-LM dict, the labels are typically derived from
    ``input_ids`` (next-token prediction), so the analyzer constructs them
    explicitly via :func:`vatis.data.collate.shifted_lm_targets`. This helper
    is for the simpler tuple/dict cases used in unit tests.
    """
    if isinstance(batch, torch.Tensor):
        return None
    if isinstance(batch, Mapping):
        if "labels" in batch and isinstance(batch["labels"], torch.Tensor):
            return batch["labels"]
        if "targets" in batch and isinstance(batch["targets"], torch.Tensor):
            return batch["targets"]
        return None
    if isinstance(batch, Sequence) and len(batch) >= 2:
        cand = batch[1]
        if isinstance(cand, torch.Tensor):
            return cand
    return None


def shifted_lm_targets(input_ids: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Build next-token-prediction targets from ``input_ids``.

    The shape is preserved: position ``s`` in the targets contains the token
    that should appear at position ``s+1``. The last position has no
    "next token" so it is set to ``ignore_index``. Pad tokens (anything with
    ``input_ids == pad`` should already have been handled by the caller via
    ``attention_mask``.)
    """
    if input_ids.ndim != 2:
        raise ValueError(
            f"shifted_lm_targets expects (B, S) input_ids, got shape {tuple(input_ids.shape)}"
        )
    targets = torch.full_like(input_ids, fill_value=ignore_index)
    targets[:, :-1] = input_ids[:, 1:]
    return targets
