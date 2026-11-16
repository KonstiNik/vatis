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


def shift_hf_labels(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Convert HuggingFace / TRL-style labels to vatis's pre-shifted form.

    The two conventions differ by one position:

    - **HuggingFace / TRL**: ``labels`` is aligned with ``input_ids``
      (``labels[:, i] == input_ids[:, i]``, ``ignore_index`` at ignored
      positions) and the *model* shifts when it computes the loss.
    - **vatis**: ``labels[:, i]`` is the target of ``logits[:, i]``, i.e. the
      token at position ``i + 1``, and nothing is shifted afterwards
      (:func:`vatis.models.hf.causal_lm_loss` and the closed-form ``chi_loss``
      both compare ``logits[:, i]`` with ``labels[:, i]``).

    This applies the shift ``transformers`` applies inside its causal-LM loss
    (``ForCausalLMLoss``: pad one ``ignore_index`` on the right, drop the first
    label), so the converted labels score a batch exactly as the HuggingFace
    model would: ``out[:, i] = labels[:, i + 1]`` and ``out[:, -1] =
    ignore_index``. Masks already written into the labels (padding, prompt
    tokens of a completion-only objective) move with them.

    Returns a new tensor of the same shape; ``labels`` is not modified.
    """
    if labels.ndim < 1:
        raise ValueError("shift_hf_labels expects labels with a sequence dimension, got a scalar")
    out = torch.full_like(labels, fill_value=ignore_index)
    out[..., :-1] = labels[..., 1:]
    return out


def unshifted_label_rows(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Flag the rows whose labels are provably in the *unshifted* HF/TRL form.

    A row is flagged iff, over its non-ignored labels,

    1. every label equals the input token at the **same** position
       (``labels[i] == input_ids[i]``, the HF/TRL alignment), and
    2. at least one label differs from the input token at the **next** position
       (``labels[i] != input_ids[i + 1]``, so the row cannot be pre-shifted).

    A correctly pre-shifted row is never flagged, whatever its tokens and
    however much of it is masked, because condition 2 is false for it by
    definition. The last position is left out of condition 2: its target, if
    any, is not in ``input_ids``. A row for which both readings hold (e.g. a run
    of one repeated token) is ambiguous and is not flagged.

    Returns a ``(B,)`` boolean tensor.
    """
    if input_ids.shape != labels.shape or labels.ndim != 2:
        raise ValueError(
            "unshifted_label_rows expects (B, S) input_ids and labels of equal shape, got "
            f"{tuple(input_ids.shape)} and {tuple(labels.shape)}"
        )
    counted = labels != ignore_index
    aligned = ((labels == input_ids) | ~counted).all(dim=1)
    not_shifted = (counted[:, :-1] & (labels[:, :-1] != input_ids[:, 1:])).any(dim=1)
    return aligned & not_shifted
