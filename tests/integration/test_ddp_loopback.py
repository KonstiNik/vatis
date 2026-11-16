"""DDP loopback: 2 ranks on CPU via torchrun, against a single-process baseline.

**Seeding.** Each rank draws the probes *for its own shard* from its own stream:
rank 0 from the per-call seed (the ``hutchinson_seed`` column — the stream a
single process uses), rank ``r > 0`` from a seed derived from it and ``r``
(``vatis.analyzer._rank_probe_seed``). Ranks do **not** partition one global
probe, so with the shipped RNG the DDP total is a different, still unbiased,
estimate of the same trace than the single-process one, and those two can only
be compared statistically. The original assertion allowed 50% relative error on
``chi_net``, which is loose enough to pass through a real sharding bug (the
native-RNG gap it tolerates is ~6%).

Until this was fixed every rank seeded its generator *identically*. The same
random vectors then landed on the ``k``-th sample of every shard, the shards'
estimation errors were strongly positively correlated, and the variance of the
summed ``chi_net`` grew by up to the world size.
``test_ddp_ranks_do_not_replay_one_probe_stream`` pins that on the real
``torchrun`` path, and ``test_ddp_shipped_rng_equals_the_per_rank_streams`` pins
the seeding rule exactly; the statistics are in
``tests/unit/test_ddp_probe_streams.py``.

The exactness claim against **single-process** is made with ``--pin-probes``:
``make_probe`` becomes a shape-keyed constant (see
``_ddp_worker.ShapeKeyedProbe``), which turns ``chi_net`` into a deterministic
functional of the shard. Then DDP **must** reproduce single-process for every
observable, including the preconditioned triple, and it does to ~1e-7 — the
floor set by the fp32 flat gradient the all-reduce operates on. Unbiasedness of
the estimator is covered by ``tests/unit/test_chi_net_probe_exactness.py``; this
file is about sharding, the all-reduce and the per-rank probe streams.

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

import vatis.analyzer as analyzer_mod
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
from vatis.core.chi_net import build_estimator
from vatis.core.observables import chi_pos
from vatis.core.preconditioner import Preconditioner
from vatis.distributed.sharding import shard_batch
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
    duplicate_half: bool = False,
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
    if duplicate_half:
        cmd.append("--duplicate-half")
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
    probes, and the measured gap is a few percent. The exactness claims live in
    ``test_ddp_matches_single_process_exactly_with_pinned_probes`` (against
    single-process) and ``test_ddp_shipped_rng_equals_the_per_rank_streams``
    (the shipped RNG, against the documented per-rank streams).
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


# --------------------------------------------------------------- per-rank probe streams


def _toy_bundle(seed: int) -> tuple[ModelBundle, Preconditioner]:
    """The worker's model and explicit P, rebuilt in this process."""
    torch.manual_seed(seed)
    model = TinyTransformer().eval()
    bundle = ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=lambda m, b: m(b["input_ids"]),
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        identifier="toy@step0",
    )
    n_params = sum(p.numel() for p in model.parameters())
    gen = torch.Generator().manual_seed(5)
    precond = Preconditioner.from_diagonal(
        torch.exp(torch.randn(n_params, generator=gen) * 2.0), descriptor="explicit"
    )
    return bundle, precond


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_ddp_ranks_do_not_replay_one_probe_stream(tmp_path: Path, method: str) -> None:
    """Two ranks holding the same samples must not return the same number twice.

    The batch is ``[a, b, a, b]``, so both shards are ``[a, b]``. Rank 0 keeps the
    per-call stream, hence its partial is exactly the single-process ``chi_net`` of
    ``[a, b]`` at the same seed. If rank 1 replayed that stream, the all-reduced
    total would be exactly twice that number — which is what this returned before
    the ranks got distinct streams (it is the world-size variance inflation in its
    purest form: the second rank contributes no new probe). Deterministic.
    """
    ddp = _run_ddp(
        tmp_path / "dup.json",
        seed=0,
        batch_size=4,
        n_h=8,
        method=method,
        preconditioner=True,
        duplicate_half=True,
    )
    half = _baseline(seed=0, batch_size=2, n_h=8, method=method, preconditioner=True)
    for observable in ("chi_net", "chi_net_P"):
        rank1_partial = ddp[observable] - half[observable]
        rel = abs(rank1_partial - half[observable]) / abs(half[observable])
        assert rel > 1e-4, (
            f"{observable}: 2-rank total {ddp[observable]!r} is twice the one-shard value "
            f"{half[observable]!r} -- rank 1 replayed rank 0's probe stream"
        )
        # Sanity on the other side: rank 1's partial is an estimate of the same trace.
        assert rel < 0.5, (observable, rank1_partial, half[observable])


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
@pytest.mark.parametrize("batch_size", [4, 5])
def test_ddp_shipped_rng_equals_the_per_rank_streams(
    tmp_path: Path, method: str, batch_size: int
) -> None:
    """The shipped RNG, exactly: DDP ``chi_net`` is the sum over ranks of the
    estimator on that rank's shard, each on its documented stream.

    This is the reproducibility contract — ``seed``, rank and world size determine
    every probe — checked against the real ``torchrun`` run rather than within a
    sampling bound. ``batch_size=5`` is the uneven split (shards of 3 and 2).
    """
    world_size, n_h = 2, 8
    ddp = _run_ddp(
        tmp_path / f"ddp_rng_{method}_{batch_size}.json",
        seed=0,
        batch_size=batch_size,
        n_h=n_h,
        method=method,
        preconditioner=True,
    )
    bundle, precond = _toy_bundle(0)
    batch = make_tiny_lm_batch(batch_size=batch_size, seed=0, pad_fraction=0.25, device="cpu")
    call_seed = analyzer_mod._per_call_seed(0, "toy@step0", "val")
    estimator = build_estimator(method, n_hutchinson=n_h, distribution="rademacher")
    expected = {"chi_net": 0.0, "chi_net_P": 0.0}
    for rank in range(world_size):
        shard, _, _ = shard_batch(batch, rank=rank, world_size=world_size)
        gen = torch.Generator().manual_seed(analyzer_mod._rank_probe_seed(call_seed, rank))
        result = estimator.compute(
            bundle.model,
            shard,
            bundle.forward_fn,
            loss_fn=bundle.loss_fn,
            valid_mask_fn=causal_lm_valid_mask,
            params=bundle.params,
            micro_batch_size=1,
            generator=gen,
            preconditioner=precond,
        )
        assert result.chi_net_p is not None
        expected["chi_net"] += float(result.chi_net.to(dtype=torch.float64))
        expected["chi_net_P"] += float(result.chi_net_p.to(dtype=torch.float64))
    expected["chi_pos"] = float(chi_pos(ddp["delta_loss"], ddp["chi_loss"], expected["chi_net"]))
    expected["chi_pos_P"] = float(
        chi_pos(ddp["delta_loss_P"], ddp["chi_loss"], expected["chi_net_P"])
    )
    for observable, want in expected.items():
        rel = abs(ddp[observable] - want) / abs(want)
        assert rel < DDP_REL, (method, batch_size, observable, ddp[observable], want, rel)
