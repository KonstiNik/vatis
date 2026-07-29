"""Hutchinson probe vector generation, with optional output-position masking.

Probes are always drawn in fp32 (so the variance is well-controlled), then
optionally cast to the forward dtype before being multiplied into the model
output for the trace projection.

**Probes are always drawn on the CPU and copied to the target device.** A
``torch.Generator`` on CPU and one on CUDA produce *different* sequences from the
same seed (different algorithms), so drawing on the device would make
``hutchinson_seed`` mean different probes on different hardware: a CPU reference
run and a GPU production run of the same configuration could then only agree to
within the Hutchinson noise floor (measured ~4e-2 relative at ``n_h = 400`` on a
toy, growing as ``1/sqrt(n_h)``), which is far above the ~5e-08 reduction-order
difference everything else shows. Drawing on the CPU makes a seed mean one probe
set everywhere.

The cost is one host-to-device copy of the probe per draw. The draw itself is
unchanged (same ``int64`` ``randint``, same arithmetic), so **CPU runs are
bit-identical to before**; only device runs move, and they move onto the CPU
values.
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
        device: device the probe is **returned** on; defaults to CPU. The draw
            itself always happens on the CPU (see the module docstring).
        dtype: target dtype; defaults to fp32. Most callers cast this to the
            forward dtype only at the multiplication site.
        generator: optional ``torch.Generator`` for reproducibility. It must be
            a **CPU** generator, because that is where the draw happens; a CUDA
            generator is refused rather than silently ignored, since accepting
            it would reintroduce device-dependent probes.

    Returns:
        A tensor of shape ``shape`` on ``device`` whose entries have unit
        variance. The values do not depend on ``device``.

    Notes:
        Padding masking is NOT applied here. Call :func:`mask_probe` afterwards
        if you need padded positions zeroed.
    """
    target = torch.device(device) if device is not None else torch.device("cpu")
    if generator is not None and generator.device.type != "cpu":
        raise ValueError(
            f"make_probe needs a CPU generator, got one on {generator.device}. Probes are "
            "drawn on the CPU and copied to the target device so that a seed identifies the "
            "same probe set on every device; a CUDA generator would silently reintroduce "
            "device-dependent probes. Build it with torch.Generator() and select the "
            "destination with device=..."
        )
    if distribution == "rademacher":
        # {0, 1} via randint, shifted to {-1, +1}: exact in fp32 and cheaper than
        # a sign() of a uniform. Drawn on CPU with the same dtype and arithmetic
        # as before, so CPU probe values are unchanged by the move.
        raw = torch.randint(0, 2, shape, device="cpu", dtype=torch.int64, generator=generator)
        probe = raw.to(dtype=dtype) * 2.0 - 1.0
    elif distribution == "gaussian":
        probe = torch.randn(shape, device="cpu", dtype=dtype, generator=generator)
    else:
        raise ValueError(
            f"unknown probe distribution: {distribution!r}. expected 'rademacher' or 'gaussian'."
        )
    return probe.to(device=target)


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
