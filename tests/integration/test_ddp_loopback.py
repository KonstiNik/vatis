"""DDP loopback: 2 ranks on CPU via torchrun, against a single-process baseline.

The interesting assertion is the **pinned-probe** one. With the shipped RNG each
rank seeds its generator identically and then draws probes *for its own shard*,
so the two runs apply different random numbers to different samples — ranks do
**not** partition one global probe (an earlier version of this docstring said
they did). The DDP total is therefore a different, still unbiased, estimate of
the same trace and the two runs can only be compared statistically. The original
assertion allowed 50% relative error on ``chi_net``, which is loose enough to
pass through a real sharding bug (the native-RNG gap it tolerates is ~6%).

So the exactness claim is made with ``--pin-probes``: ``make_probe`` becomes a
shape-keyed constant (see ``_ddp_worker.ShapeKeyedProbe``), which turns
``chi_net`` into a deterministic functional of the shard. Then DDP **must**
reproduce single-process for every observable, including the preconditioned
triple, and it does to ~1e-7 — the floor set by the fp32 flat gradient the
all-reduce operates on. Unbiasedness of the estimator is covered by
``tests/unit/test_chi_net_probe_exactness.py``; this file is about sharding and
the all-reduce.

Also covered: ``B % world_size != 0`` (batch 5 over 2 ranks), which exercises the
token-weighted rank re-weighting rather than an even split, plus padding so the
per-rank valid-token counts differ.

The multi-GPU path remains unvalidated: this is a CPU gloo loopback
(``CUDA_VISIBLE_DEVICES=""``), because NCCL wants one device per rank.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import vatis.core.chi_net.hutchinson as hutch_mod
import vatis.core.chi_net.per_seq_cv as cv_mod
from tests.fixtures.tiny_transformer import (
    TinyTransformer,
    causal_lm_loss,
    causal_lm_valid_mask,
    make_tiny_lm_batch,
)
from tests.integration._ddp_worker import ShapeKeyedProbe
from vatis import analyze
from vatis.core.preconditioner import Preconditioner
from vatis.models.hf import ModelBundle

pytestmark = pytest.mark.integration

# The DDP path re-weights and all-reduces an **fp32** flat gradient, so
# "exact in exact arithmetic" lands at ~1e-8 relative here.
DDP_REL = 1e-6


def _run_ddp(
    out_path: Path,
    *,
    seed: int,
    batch_size: int,
    n_h: int,
    method: str | None = None,
    pin_probes: bool = False,
    preconditioner: bool = False,
) -> dict[str, float]:
    """Spawn a 2-rank torchrun pinned to CPU, return rank 0's observables."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU even if a GPU is around
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--standalone",
        str(Path(__file__).resolve().parent / "_ddp_worker.py"),
        "--out",
        str(out_path),
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
        "--n-hutchinson",
        str(n_h),
    ]
    if method is not None:
        cmd += ["--chi-net-method", method]
    if pin_probes:
        cmd.append("--pin-probes")
    if preconditioner:
        cmd.append("--preconditioner")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            "torchrun failed:\n--- stdout ---\n"
            + result.stdout
            + "\n--- stderr ---\n"
            + result.stderr
        )
    rows = json.loads(out_path.read_text())
    return {r["observable"]: r["value"] for r in rows}


def _baseline(
    *,
    seed: int,
    batch_size: int,
    n_h: int,
    method: str | None = None,
    pin_probes: bool = False,
    preconditioner: bool = False,
) -> dict[str, float]:
    """The same computation in one process, with the same probe treatment."""
    real_hutch, real_cv = hutch_mod.make_probe, cv_mod.make_probe
    if pin_probes:
        pinned = ShapeKeyedProbe(4242)
        hutch_mod.make_probe = pinned  # type: ignore[assignment]
        cv_mod.make_probe = pinned  # type: ignore[assignment]
    try:
        torch.manual_seed(seed)
        model = TinyTransformer().eval()
        for p in model.parameters():
            p.requires_grad_(True)
        bundle = ModelBundle(
            model=model,
            params=list(model.parameters()),
            forward_fn=lambda m, b: m(b["input_ids"]),
            loss_fn=causal_lm_loss,
            valid_mask_fn=causal_lm_valid_mask,
            identifier="toy@step0",
        )
        precond = None
        if preconditioner:
            n_params = sum(p.numel() for p in model.parameters())
            gen = torch.Generator().manual_seed(5)
            precond = Preconditioner.from_diagonal(
                torch.exp(torch.randn(n_params, generator=gen) * 2.0), descriptor="explicit"
            )
        batch = make_tiny_lm_batch(
            batch_size=batch_size, seed=seed, pad_fraction=0.25, device="cpu"
        )
        results = analyze(
            model=bundle,
            revisions=["step0"],
            eval_batches={"val": batch},
            preconditioner=precond,
            chi_net_method=method,
            n_hutchinson=n_h,
            micro_batch_size=1,
            sink=None,
            device="cpu",
        )
        return {r.observable: r.value for r in results[0].rows}
    finally:
        hutch_mod.make_probe = real_hutch
        cv_mod.make_probe = real_cv


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
@pytest.mark.parametrize("batch_size", [4, 5])
def test_ddp_matches_single_process_exactly_with_pinned_probes(
    tmp_path: Path, method: str, batch_size: int
) -> None:
    """Every observable, including the P triple. ``batch_size=5`` is the uneven split."""
    out = tmp_path / f"ddp_{method}_{batch_size}.json"
    ddp = _run_ddp(
        out,
        seed=0,
        batch_size=batch_size,
        n_h=8,
        method=method,
        pin_probes=True,
        preconditioner=True,
    )
    base = _baseline(
        seed=0,
        batch_size=batch_size,
        n_h=8,
        method=method,
        pin_probes=True,
        preconditioner=True,
    )
    assert set(ddp) == set(base)
    for observable, expected in base.items():
        got = ddp[observable]
        rel = abs(got - expected) / max(abs(expected), 1e-300)
        assert rel < DDP_REL, (method, batch_size, observable, got, expected, rel)


def test_ddp_loopback_matches_single_process(tmp_path: Path) -> None:
    """The shipped RNG path: deterministic observables exact, ``chi_net`` sampled.

    The ``chi_net`` bound here is a **sampling** tolerance, not an accuracy claim:
    with the shipped RNG the DDP and single-process estimates use different
    probes, and the measured gap is a few percent. The exactness claim lives in
    ``test_ddp_matches_single_process_exactly_with_pinned_probes``.
    """
    out = tmp_path / "ddp.json"
    ddp = _run_ddp(out, seed=0, batch_size=4, n_h=8)
    base = _baseline(seed=0, batch_size=4, n_h=8)

    # Probe-independent observables: sharding + the re-weighted all-reduce is
    # exact, so these match up to the fp32 accumulation floor.
    for observable in ("chi_loss", "chi_loss_normalized", "delta_loss"):
        rel = abs(ddp[observable] - base[observable]) / max(abs(base[observable]), 1e-300)
        assert rel < DDP_REL, (observable, ddp[observable], base[observable], rel)

    # The absolute bounds this test carried before it was rewritten, verbatim.
    # The relative check above is tighter for ``delta_loss`` but ~20x / ~10x
    # looser than these for ``chi_loss`` / ``chi_loss_normalized`` at their
    # magnitudes here, so on its own it would have weakened the test. Both hold.
    assert abs(ddp["chi_loss"] - base["chi_loss"]) < 1e-9
    assert abs(ddp["chi_loss_normalized"] - base["chi_loss_normalized"]) < 1e-7
    assert abs(ddp["delta_loss"] - base["delta_loss"]) < 1e-7

    rel_chi_net = abs(ddp["chi_net"] - base["chi_net"]) / max(1e-12, abs(base["chi_net"]))
    assert rel_chi_net < 0.5, f"chi_net mismatch: {ddp['chi_net']} vs {base['chi_net']}"
    assert all(value == value for value in ddp.values()), "no NaNs"
