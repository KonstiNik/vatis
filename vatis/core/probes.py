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

**The Rademacher draw is bit-packed**, because the naive version is far too slow
to ship. Generating one ``fp32`` element per probe entry on the CPU costs a *full
CPU RNG call per entry*; measured on the WP0 example's probe shape
``(8, 512, 4096)`` = 16.8M entries, RTX 3090 Ti:

===========================================================  ==============
draw strategy                                                 ms per probe
===========================================================  ==============
on-device ``randint`` (device-dependent values, the old way)          0.69
CPU ``int64 randint`` -> fp32 -> H2D (naive CPU draw)               125.25
CPU ``uint8 randint`` -> H2D -> expand on device                     44.16
**CPU packed bytes -> H2D -> bit-unpack on device**                  11.85
===========================================================  ==============

So the packing matters by 10.6x: a Rademacher entry needs exactly **one random
bit**, and a uniform ``uint8`` supplies eight. We draw ``ceil(N/8)`` bytes on the
CPU, ship 1/8 as many bytes, and unpack on the device where the work is cheap.
Each bit of a uniform byte is an independent fair coin, so this is exactly the
same distribution -- no per-bit-position bias (asserted in
``tests/unit/test_probes.py``).

The residual overhead over an on-device draw is ~17x per probe, which on the WP0
example works out to ~1.1 s per checkpoint against ~4.1 s of backwards (~26%).
The naive CPU draw was ~12 s per checkpoint (3.5x end to end) and would have made
the 8B production target infeasible, which is why it is not what ships.

``gaussian`` probes cannot be bit-packed and take the plain CPU draw. They are not
the default; if a large-scale run needs them, that is the place to look.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

ProbeDistribution = Literal["rademacher", "gaussian"]

# One random *bit* is all a Rademacher entry needs; a uniform byte supplies 8.
_BITS_PER_BYTE = 8


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
        return _rademacher_packed(shape, dtype=dtype, target=target, generator=generator)
    if distribution == "gaussian":
        # Cannot be bit-packed; the plain CPU draw is the cost of a Gaussian probe.
        return torch.randn(shape, device="cpu", dtype=dtype, generator=generator).to(device=target)
    raise ValueError(
        f"unknown probe distribution: {distribution!r}. expected 'rademacher' or 'gaussian'."
    )


def _rademacher_packed(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    target: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """A +-1 probe from CPU-drawn packed bytes, unpacked on ``target``.

    Draws ``ceil(N/8)`` uniform bytes on the CPU (one RNG call per *eight*
    entries instead of one per entry, the 10.6x in the module docstring), ships
    them, and expands bit ``k`` of byte ``i`` into entry ``8i + k``. Every bit of
    a uniform byte is an independent fair coin, so the result is an exact
    Rademacher vector; the values depend only on the seed, never on ``target``.
    """
    n_elem = math.prod(shape) if shape else 1
    n_bytes = (n_elem + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE
    packed = torch.randint(0, 256, (n_bytes,), device="cpu", dtype=torch.uint8, generator=generator)
    shifts = torch.arange(_BITS_PER_BYTE, device=target, dtype=torch.uint8)
    bits = (packed.to(device=target).unsqueeze(1) >> shifts) & 1
    return (bits.reshape(-1)[:n_elem].to(dtype=dtype) * 2.0 - 1.0).reshape(shape)


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
