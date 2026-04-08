"""HuggingFace causal-LM loader.

This module is a thin wrapper that:
    - calls ``transformers.AutoModelForCausalLM.from_pretrained(name, revision=...)``
    - returns a small ``HFModelBundle`` containing the model, the loss
      function, the forward function, and a valid-mask function — exactly
      the four pieces the analyzer needs to drive an LNA computation.

Why a bundle? Because vatis treats the model as opaque. The analyzer needs
to know how to (a) call the model on a batch and get logits, (b) compute the
scalar loss, (c) figure out which output positions are valid. For HF causal
LMs, all four of these have a canonical form, so we ship it.

For non-HF models, users supply the same four ingredients themselves and call
``Analyzer`` directly with their own ``ModelBundle``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vatis.core.normalization import DEFAULT_IGNORE_INDEX

# Type aliases (mirrored in core/chi_net/base for the estimator side).
ForwardFn = Callable[[nn.Module, Any], torch.Tensor]
LossFn = Callable[[torch.Tensor, Any], torch.Tensor]
ValidMaskFn = Callable[[Any, torch.Tensor], torch.Tensor | None]


@dataclass
class ModelBundle:
    """Everything the analyzer needs to evaluate one model on one batch.

    Attributes:
        model: ``nn.Module`` (must be in eval mode by the time the analyzer
            is called; the analyzer will set it back if needed).
        params: list of parameters to back-propagate into. By default this is
            ``[p for p in model.parameters() if p.requires_grad]``, but for
            checkpoint analysis we typically want all parameters regardless
            of ``requires_grad``.
        forward_fn: ``forward_fn(model, batch) -> logits`` of shape
            ``(B, S, V)`` for an LM or ``(B, V)`` for a classifier.
        loss_fn: ``loss_fn(logits, batch) -> scalar``. Should be mean-reduced
            over valid tokens (HF convention).
        valid_mask_fn: ``valid_mask_fn(batch, logits) -> bool tensor`` of
            shape ``logits.shape[:-1]``, identifying valid output positions.
            ``None`` is allowed for "all positions valid".
        identifier: human-readable model name (e.g. ``"pythia-160m@step1000"``)
            for log/sink rows.
    """

    model: nn.Module
    params: list[nn.Parameter]
    forward_fn: ForwardFn
    loss_fn: LossFn
    valid_mask_fn: ValidMaskFn
    identifier: str = ""


def causal_lm_forward(model: nn.Module, batch: Any) -> torch.Tensor:
    """Standard HF causal-LM forward: returns ``outputs.logits``.

    The batch must be a dict with ``"input_ids"`` (and optionally
    ``"attention_mask"``). Other keys are passed through to the model.
    """
    if not isinstance(batch, dict):
        raise TypeError(f"causal_lm_forward expects a dict batch, got {type(batch).__name__}")
    inputs = {k: v for k, v in batch.items() if k not in ("labels",)}
    out = model(**inputs)
    if hasattr(out, "logits"):
        logits = out.logits
        assert isinstance(logits, torch.Tensor)
        return logits
    if isinstance(out, torch.Tensor):
        return out
    raise TypeError(
        f"unexpected forward output type for causal-LM: {type(out).__name__}; "
        "expected ModelOutput with .logits or a Tensor."
    )


def causal_lm_loss(
    logits: torch.Tensor,
    batch: Any,
    *,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> torch.Tensor:
    """Mean-reduced cross entropy on (shifted) labels.

    The batch must contain either:
        - ``"labels"`` (already shifted, with ``-100`` at ignored positions),
          which is the canonical HF form;
        - or just ``"input_ids"`` and ``"attention_mask"``, in which case
          we shift internally.

    Returns:
        A 0-dim scalar tensor with a graph back to the model.
    """
    if not isinstance(batch, dict):
        raise TypeError(f"causal_lm_loss expects a dict batch, got {type(batch).__name__}")
    if "labels" in batch and isinstance(batch["labels"], torch.Tensor):
        labels = batch["labels"]
    else:
        if "input_ids" not in batch:
            raise ValueError("causal_lm_loss needs either 'labels' or 'input_ids' in the batch")
        from vatis.data.collate import shifted_lm_targets

        input_ids = batch["input_ids"]
        labels = shifted_lm_targets(input_ids, ignore_index=ignore_index)
        if "attention_mask" in batch and isinstance(batch["attention_mask"], torch.Tensor):
            mask = batch["attention_mask"]
            shifted_mask = torch.zeros_like(mask)
            shifted_mask[:, :-1] = mask[:, 1:]
            labels = labels.masked_fill(shifted_mask == 0, ignore_index)

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )


def causal_lm_valid_mask(
    batch: Any,
    logits: torch.Tensor,
    *,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> torch.Tensor | None:
    """Return a ``(B, S)`` boolean mask of valid label positions."""
    if not isinstance(batch, dict):
        return None
    if "labels" in batch and isinstance(batch["labels"], torch.Tensor):
        return batch["labels"] != ignore_index
    # Fall back to attention_mask if no explicit labels.
    if "attention_mask" in batch and isinstance(batch["attention_mask"], torch.Tensor):
        mask = batch["attention_mask"]
        # Shift the mask along the sequence dim — only positions whose target
        # is valid count.
        shifted = torch.zeros_like(mask)
        shifted[:, :-1] = mask[:, 1:]
        return shifted.to(dtype=torch.bool)
    # No mask info → assume all positions are valid except the last (no
    # next-token target).
    out = torch.ones(logits.shape[:-1], dtype=torch.bool, device=logits.device)
    out[:, -1] = False
    return out


def load_hf_model(
    name: str,
    *,
    revision: str | None = None,
    dtype: str = "bf16",
    device: torch.device | str = "cpu",
    trust_remote_code: bool = False,
) -> ModelBundle:
    """Load an HF causal LM by name + revision and wrap it in a ModelBundle.

    Args:
        name: HuggingFace hub identifier, e.g. ``"EleutherAI/pythia-160m"``.
        revision: optional revision (a tag or commit hash, e.g.
            ``"step1000"`` for Pythia checkpoints).
        dtype: forward dtype, one of ``"fp32"``, ``"bf16"``, ``"fp16"``.
            Gradients are always accumulated in fp32 by the analyzer.
        device: target device for the model.
        trust_remote_code: passed through to ``from_pretrained``.

    Returns:
        A ModelBundle ready to feed to ``Analyzer``.
    """
    from transformers import AutoModelForCausalLM

    torch_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[dtype]

    model: nn.Module = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device=device)
    model.eval()
    # Make sure all parameters are leaves with requires_grad=True so backwards
    # land where we expect.
    for p in model.parameters():
        p.requires_grad_(True)

    identifier = name if revision is None else f"{name}@{revision}"
    return ModelBundle(
        model=model,
        params=[p for p in model.parameters() if p.requires_grad],
        forward_fn=causal_lm_forward,
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        identifier=identifier,
    )
