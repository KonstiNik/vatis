"""Hutchinson probe vector generation, with optional output-position masking.

Probes are always drawn in fp32 (so the variance is well-controlled), then
optionally cast to the forward dtype before being multiplied into the model
output for the trace projection.
"""

from __future__ import annotations

from typing import Literal

import torch

ProbeDistribution = Literal["rademacher", "gaussian"]


def make_probe(
    shape: tuple[int, ...],
    *,
    distribution: ProbeDistribution = "rademacher",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw a single Hutchinson probe vector.

    Args:
        shape: shape of the probe; should match the model output for the
            current micro-batch (e.g. ``(M, S, V)`` for an HF causal LM).
        distribution: ``"rademacher"`` (default, lower variance) or
            ``"gaussian"``.
        device: target device; defaults to CPU.
        dtype: target dtype; defaults to fp32. Most callers cast this to the
            forward dtype only at the multiplication site.
        generator: optional torch.Generator for reproducibility. If provided,
            the device of the generator must match ``device``.

    Returns:
        A tensor of shape ``shape`` whose entries have unit variance.

    Notes:
        Padding masking is NOT applied here. Call :func:`mask_probe` afterwards
        if you need padded positions zeroed.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    if distribution == "rademacher":
        # ``torch.randint`` does not honor a CPU generator on a CUDA device, so
        # we always draw on the requested device. Use uint8 then map to ±1.
        # We pick {0, 1} via randint and shift to {-1, +1} to keep numerical
        # accuracy at fp32.
        raw = torch.randint(0, 2, shape, device=device, dtype=torch.int64, generator=generator)
        probe = raw.to(dtype=dtype) * 2.0 - 1.0
        return probe
    if distribution == "gaussian":
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
    raise ValueError(
        f"unknown probe distribution: {distribution!r}. expected 'rademacher' or 'gaussian'."
    )


def mask_probe(probe: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    """Zero out probe entries at invalid positions.

    Args:
        probe: probe tensor, e.g. shape ``(M, S, V)``.
        valid_mask: boolean tensor whose shape broadcasts to ``probe`` (e.g.
            ``(M, S)`` for an LM, with ``True`` at valid token positions). If
            ``None``, the probe is returned unchanged.

    Returns:
        A new tensor with invalid positions zeroed. If ``valid_mask`` is
        ``None``, returns the input unchanged.
    """
    if valid_mask is None:
        return probe
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.to(dtype=torch.bool)
    # Reshape valid_mask so it broadcasts over trailing dims of probe.
    # E.g. probe is (M, S, V), valid_mask is (M, S) -> we add a trailing dim.
    while valid_mask.ndim < probe.ndim:
        valid_mask = valid_mask.unsqueeze(-1)
    return probe * valid_mask.to(dtype=probe.dtype)
