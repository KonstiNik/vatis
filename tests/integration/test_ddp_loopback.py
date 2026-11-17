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
estimation errors were positively correlated, and the variance of the summed
``chi_net`` was up to the world size times what independent rank streams give.
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

The ``*_capped`` rows have their own pinned-probe test, because the ``P`` of the
test above has no ceiling and therefore no capped rows: with it, dropping the
all-reduce of ``chi_net_P_capped`` changed nothing any tier asserts.
``test_ddp_shipped_rng_capped_row_equals_the_per_rank_streams`` adds the same row
under the shipped RNG, against the per-rank streams.

The ``loss_change_first_order*`` and realized-change rows have their own test
(``test_ddp_update_response_matches_single_process``): they involve no probes,
so DDP has to reproduce a single process as it is.

The provenance columns and ``value_probe_stderr`` have their own section at the end:
every rank's rows name the estimator, the world size, the micro-batch size and
the probe distribution of the run, the rows of an AdamW
``P`` carry its ``eps`` / ``beta2`` / capped fraction, and the standard error of
the all-reduced ``chi_net`` is the root of the **sum over ranks** of the
variances the estimator reports on each rank's stream -- at
``micro_batch_size = 2`` on the uneven split, so the sum over micro-batches
within a rank and the sum across ranks are both exercised on the real path.

The multi-GPU path remains unvalidated: this is a CPU gloo loopback
(``CUDA_VISIBLE_DEVICES=""``), because NCCL wants one device per rank.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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
from tests.integration._ddp_worker import (
    ADAMW_BETA2,
    ADAMW_EPS,
    GatedTiny,
    ShapeKeyedProbe,
    adamw_preconditioner,
    adamw_second_moment,
    ceiling_preconditioner,
    gate_batch,
    update_response_setup,
)
from vatis import analyze
from vatis.analyzer import (
    REALIZED_CHANGE_OBSERVABLES,
    UPDATE_RESPONSE_OBSERVABLES,
    UPDATE_STEP_OBSERVABLES,
)
from vatis.core.chi_net import build_estimator
from vatis.core.observables import chi_pos
from vatis.core.preconditioner import Preconditioner
from vatis.distributed.sharding import shard_batch
from vatis.models.hf import ModelBundle

pytestmark = pytest.mark.integration

# The DDP path re-weights and all-reduces an **fp32** flat gradient, so
# "exact in exact arithmetic" lands at ~1e-8 relative here.
DDP_REL = 1e-6


def _run_ddp_rows(
    out_path: Path,
    *,
    seed: int,
    batch_size: int,
    n_h: int,
    method: str | None = None,
    pin_probes: bool = False,
    preconditioner: bool = False,
    ceiling: bool = False,
    duplicate_half: bool = False,
    update_response: bool = False,
    adamw: bool = False,
    micro_batch_size: int | None = None,
    distribution: str | None = None,
    gated: bool = False,
    timeout: float = 300,
) -> list[list[dict[str, Any]]]:
    """Spawn a 2-rank torchrun pinned to CPU, return every rank's rows (as dicts)."""
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
    if ceiling:
        cmd.append("--ceiling")
    if duplicate_half:
        cmd.append("--duplicate-half")
    if update_response:
        cmd.append("--update-response")
    if adamw:
        cmd.append("--adamw")
    if micro_batch_size is not None:
        cmd += ["--micro-batch-size", str(micro_batch_size)]
    if distribution is not None:
        cmd += ["--hutchinson-distribution", distribution]
    if gated:
        cmd.append("--gated")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            "torchrun failed:\n--- stdout ---\n"
            + result.stdout
            + "\n--- stderr ---\n"
            + result.stderr
        )
    return [
        json.loads(out_path.read_text()),
        json.loads(Path(f"{out_path}.rank1").read_text()),
    ]


def _run_ddp(out_path: Path, **kwargs: Any) -> dict[str, float]:
    """Spawn a 2-rank torchrun pinned to CPU, return rank 0's observables."""
    rank0_rows = _run_ddp_rows(out_path, **kwargs)[0]
    return {r["observable"]: r["value"] for r in rank0_rows}


def _baseline(
    *,
    seed: int,
    batch_size: int,
    n_h: int,
    method: str | None = None,
    pin_probes: bool = False,
    preconditioner: bool = False,
    ceiling: bool = False,
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
        if preconditioner and ceiling:
            precond = ceiling_preconditioner(sum(p.numel() for p in model.parameters()))
        elif preconditioner:
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


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
@pytest.mark.parametrize("batch_size", [4, 5])
def test_ddp_matches_single_process_for_the_capped_rows(
    tmp_path: Path, method: str, batch_size: int
) -> None:
    """A ``P`` with a ceiling and entries at it: the two ``*_capped`` rows as well.

    ``chi_net_P_capped`` is a per-rank partial like ``chi_net_P`` and needs its
    own all-reduce; ``delta_loss_P_capped`` is a reduction of the all-reduced
    gradient. Both must equal the single-process values, like every other row.
    """
    out = tmp_path / f"ddp_capped_{method}_{batch_size}.json"
    kwargs = {
        "seed": 0,
        "batch_size": batch_size,
        "n_h": 8,
        "method": method,
        "pin_probes": True,
        "preconditioner": True,
        "ceiling": True,
    }
    ddp = _run_ddp(out, **kwargs)
    base = _baseline(**kwargs)
    assert set(ddp) == set(base)
    capped_rows = {"delta_loss_P_capped": "delta_loss_P", "chi_net_P_capped": "chi_net_P"}
    assert set(capped_rows) <= set(base)
    # vacuity guard: the capped part is a real share of each observable, neither
    # nothing nor all of it, so a wrong capped row cannot hide behind a zero.
    for capped, full in capped_rows.items():
        assert 0.05 * base[full] < base[capped] < 0.95 * base[full], (capped, base[capped])
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

    # The absolute bounds this test carried before it was rewritten, for the two
    # observables where they are the tighter check (~20x / ~10x tighter than the
    # relative one at these magnitudes).
    #
    # ``delta_loss`` is deliberately checked relatively only (owner decision,
    # 2026-11-16). Its DDP-vs-single-process difference is fp32 rounding from a
    # different summation order, so it scales with the value: the former
    # absolute ``< 1e-7`` was 1 part in ~8,000 on the toy model this test started
    # with and asks for more than fp32 resolves once ``delta_loss`` is of order
    # one. The relative 1e-6 above holds at any magnitude and is still orders of
    # magnitude below what a sharding or all-reduce defect produces.
    assert abs(ddp["chi_loss"] - base["chi_loss"]) < 1e-9
    assert abs(ddp["chi_loss_normalized"] - base["chi_loss_normalized"]) < 1e-7

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


def _sum_over_rank_streams(
    bundle: ModelBundle,
    precond: Preconditioner,
    batch: dict[str, torch.Tensor],
    *,
    method: str,
    n_h: int,
    call_seed: int,
    world_size: int,
) -> dict[str, float]:
    """What the all-reduce must return for the probe-dependent traces.

    The estimator on each rank's shard, with a generator seeded
    ``_rank_probe_seed(call_seed, rank)``, summed over the ranks: ``chi_net``,
    ``chi_net_P`` and, for a ``P`` with a ceiling, ``chi_net_P_capped``.
    """
    estimator = build_estimator(method, n_hutchinson=n_h, distribution="rademacher")
    total: dict[str, float] = {}
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
        parts = {"chi_net": result.chi_net, "chi_net_P": result.chi_net_p}
        if result.chi_net_p_capped is not None:
            parts["chi_net_P_capped"] = result.chi_net_p_capped
        for name, value in parts.items():
            total[name] = total.get(name, 0.0) + float(value.to(dtype=torch.float64))
    return total


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

    This is the reproducibility contract — ``seed``, checkpoint id, batch name,
    rank and world size determine every probe — checked against the real
    ``torchrun`` run rather than within a sampling bound. ``batch_size=5`` is the
    uneven split (shards of 3 and 2).

    It also pins what the ``hutchinson_seed`` column of a multi-rank run holds:
    the per-call seed, on every row of every rank — not the rank-derived seed a
    rank ``r > 0`` actually seeds its generator with.
    """
    world_size, n_h = 2, 8
    rank_rows = _run_ddp_rows(
        tmp_path / f"ddp_rng_{method}_{batch_size}.json",
        seed=0,
        batch_size=batch_size,
        n_h=n_h,
        method=method,
        preconditioner=True,
    )
    ddp = {r["observable"]: r["value"] for r in rank_rows[0]}
    bundle, precond = _toy_bundle(0)
    batch = make_tiny_lm_batch(batch_size=batch_size, seed=0, pad_fraction=0.25, device="cpu")
    call_seed = analyzer_mod._per_call_seed(0, "toy@step0", "val")
    assert len(rank_rows) == world_size
    for rank, rows in enumerate(rank_rows):
        assert rows, rank
        assert {r["hutchinson_seed"] for r in rows} == {call_seed}, (rank, call_seed)
    expected = _sum_over_rank_streams(
        bundle, precond, batch, method=method, n_h=n_h, call_seed=call_seed, world_size=world_size
    )
    assert set(expected) == {"chi_net", "chi_net_P"}  # this P has no ceiling
    expected["chi_pos"] = float(chi_pos(ddp["delta_loss"], ddp["chi_loss"], expected["chi_net"]))
    expected["chi_pos_P"] = float(
        chi_pos(ddp["delta_loss_P"], ddp["chi_loss"], expected["chi_net_P"])
    )
    for observable, want in expected.items():
        rel = abs(ddp[observable] - want) / abs(want)
        assert rel < DDP_REL, (method, batch_size, observable, ddp[observable], want, rel)


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_ddp_shipped_rng_capped_row_equals_the_per_rank_streams(
    tmp_path: Path, method: str
) -> None:
    """The same contract for a ``P`` with a ceiling: ``chi_net_P_capped`` too.

    ``chi_net_P_capped`` is a per-rank partial drawn from the same probes as
    ``chi_net`` and ``chi_net_P``, so with the shipped RNG it must be the sum of
    the per-rank capped parts on the per-rank streams. The pinned-probe test
    above cannot see which stream a rank draws from (it replaces the probes);
    this one does, on the uneven split (batch 5: shards of 3 and 2).
    """
    world_size, n_h, batch_size = 2, 8, 5
    rank_rows = _run_ddp_rows(
        tmp_path / f"ddp_rng_capped_{method}.json",
        seed=0,
        batch_size=batch_size,
        n_h=n_h,
        method=method,
        preconditioner=True,
        ceiling=True,
    )
    ddp = {r["observable"]: r["value"] for r in rank_rows[0]}
    bundle, _ = _toy_bundle(0)
    precond = ceiling_preconditioner(sum(p.numel() for p in bundle.params))
    batch = make_tiny_lm_batch(batch_size=batch_size, seed=0, pad_fraction=0.25, device="cpu")
    call_seed = analyzer_mod._per_call_seed(0, "toy@step0", "val")
    for rank, rows in enumerate(rank_rows):
        assert rows, rank
        assert {r["hutchinson_seed"] for r in rows} == {call_seed}, (rank, call_seed)
    expected = _sum_over_rank_streams(
        bundle, precond, batch, method=method, n_h=n_h, call_seed=call_seed, world_size=world_size
    )
    assert set(expected) == {"chi_net", "chi_net_P", "chi_net_P_capped"}
    # vacuity guard, as in the pinned-probe test: the capped part is a real share.
    assert (
        0.05 * expected["chi_net_P"] < expected["chi_net_P_capped"] < 0.95 * expected["chi_net_P"]
    )
    for observable, want in expected.items():
        rel = abs(ddp[observable] - want) / abs(want)
        assert rel < DDP_REL, (method, observable, ddp[observable], want, rel)


@pytest.mark.parametrize("batch_size", [4, 5])
def test_ddp_update_response_matches_single_process(tmp_path: Path, batch_size: int) -> None:
    """The update-response and realized-change rows under DDP, against one process.

    No probes enter these rows, so no pinning is needed: the three parts are dot
    products of the all-reduced gradients with an update every rank rebuilds
    identically, and the realized change is two token-weighted mean losses that
    have to be all-reduced over the shards. ``batch_size=5`` is the uneven split,
    and the batches are padded, so the ranks hold different token counts — a
    plain average of per-rank losses, or a dot product with a rank's partial
    gradient, would not reproduce the single-process number.
    """
    seed = 0
    ranks = _run_ddp_rows(
        tmp_path / f"ddp_response_{batch_size}.json",
        seed=seed,
        batch_size=batch_size,
        n_h=2,
        method="hutchinson",
        update_response=True,
    )

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
    batch = make_tiny_lm_batch(batch_size=batch_size, seed=seed, pad_fraction=0.25, device="cpu")
    other = make_tiny_lm_batch(
        batch_size=batch_size, seed=seed + 1, pad_fraction=0.25, device="cpu"
    )
    update = update_response_setup(bundle, batch)
    results = analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"val": batch, "other": other},
        update_response=lambda ckpt_id: update,
        chi_net_method="hutchinson",
        n_hutchinson=2,
        micro_batch_size=1,
        sink=None,
        device="cpu",
    )
    names = UPDATE_RESPONSE_OBSERVABLES + REALIZED_CHANGE_OBSERVABLES
    base = {
        (r.batch_a, r.batch_b, r.observable): r.value
        for r in results[0].rows
        if r.observable in names
    }
    assert set(base) == {("val", b, name) for b in ("val", "other") for name in names}
    assert all(value != 0.0 for value in base.values())

    for rank, rows in enumerate(ranks):
        got = {
            (r["batch_a"], r["batch_b"], r["observable"]): r["value"]
            for r in rows
            if r["observable"] in names
        }
        assert set(got) == set(base), rank
        for key, expected in base.items():
            # The residual is a difference of two numbers of the size of the
            # first-order term, so its tolerance is set by that size.
            scale = abs(base[(key[0], key[1], "loss_change_first_order_weights")])
            tolerance = DDP_REL * (scale if key[2] == "loss_change_residual" else abs(expected))
            assert abs(got[key] - expected) <= tolerance, (rank, key, got[key], expected)
        counts = {
            (r["batch_b"], r["n_valid_a"], r["n_valid_b"]) for r in rows if r["observable"] in names
        }
        assert len(counts) == 2, counts

    # The step's learning rate and clip factor: one row each on the update batch's
    # own pair, the same on every rank. The rule clips, and the factor is below
    # one, so it was formed from the all-reduced gradient — a rank's partial
    # gradient has another norm.
    base_step = {
        (r.batch_a, r.batch_b, r.observable): r.value
        for r in results[0].rows
        if r.observable in UPDATE_STEP_OBSERVABLES
    }
    assert set(base_step) == {("val", "val", "update_lr"), ("val", "val", "update_clip_factor")}
    assert base_step[("val", "val", "update_lr")] == 1e-2
    assert 0.0 < base_step[("val", "val", "update_clip_factor")] < 1.0
    for rank, rows in enumerate(ranks):
        got_step = {
            (r["batch_a"], r["batch_b"], r["observable"]): r["value"]
            for r in rows
            if r["observable"] in UPDATE_STEP_OBSERVABLES
        }
        assert set(got_step) == set(base_step), rank
        for key, expected in base_step.items():
            assert abs(got_step[key] - expected) <= DDP_REL * abs(expected), (rank, key)


# ------------------------------------------------ provenance columns and value_probe_stderr

_P_ROWS = ("delta_loss_P", "chi_net_P", "chi_pos_P", "delta_loss_P_capped", "chi_net_P_capped")


def _rank_stream_variances(
    bundle: ModelBundle,
    precond: Preconditioner,
    batch: dict[str, torch.Tensor],
    *,
    method: str,
    n_h: int,
    micro_batch_size: int,
    call_seed: int,
    world_size: int,
) -> list[dict[str, float]]:
    """Per rank, the variances the estimator reports on that rank's shard and stream.

    The same construction as :func:`_sum_over_rank_streams`, for the three
    variances instead of the three estimates, and not yet summed, so a test can
    see that every rank contributes.
    """
    estimator = build_estimator(method, n_hutchinson=n_h, distribution="rademacher")
    per_rank = []
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
            micro_batch_size=micro_batch_size,
            generator=gen,
            preconditioner=precond,
        )
        assert result.chi_net_var is not None
        assert result.chi_net_p_var is not None
        assert result.chi_net_p_capped_var is not None
        per_rank.append(
            {
                "chi_net": float(result.chi_net_var),
                "chi_net_P": float(result.chi_net_p_var),
                "chi_net_P_capped": float(result.chi_net_p_capped_var),
            }
        )
    return per_rank


@pytest.mark.parametrize(
    ("method", "distribution", "micro"),
    [("hutchinson", "rademacher", 1), ("per_sequence_cv", "gaussian", 2)],
)
def test_ddp_rows_record_the_estimator_the_ranks_and_the_adamw_p(
    tmp_path: Path, method: str, distribution: str, micro: int
) -> None:
    """What a two-rank run writes into the provenance columns, on every rank.

    ``n_ranks`` is the world size (2, where a single process writes 1),
    ``chi_net_method`` the estimator, ``micro_batch_size`` and
    ``hutchinson_distribution`` the micro-batch size and probe distribution the
    run was made with -- once with the defaults, once with Gaussian probes at
    micro-batch 2 -- on every row. The rows computed with the
    AdamW ``P`` carry its ``eps`` and ``beta2`` and the fraction of its
    coordinates whose second moment is zero -- counted here from the second
    moment itself -- and the raw rows carry none of the three.
    """
    rank_rows = _run_ddp_rows(
        tmp_path / f"ddp_provenance_{method}_{distribution}.json",
        seed=0,
        batch_size=5,
        n_h=4,
        method=method,
        preconditioner=True,
        adamw=True,
        micro_batch_size=micro,
        distribution=distribution,
    )
    bundle, _ = _toy_bundle(0)
    exp_avg_sq = adamw_second_moment(sum(p.numel() for p in bundle.params))
    zero_fraction = float((exp_avg_sq == 0).sum()) / exp_avg_sq.numel()
    assert 0.05 < zero_fraction < 0.2
    assert len(rank_rows) == 2
    for rank, rows in enumerate(rank_rows):
        assert {r["observable"] for r in rows} >= set(_P_ROWS), rank
        for r in rows:
            assert r["n_ranks"] == 2, (rank, r)
            assert r["chi_net_method"] == method, (rank, r)
            assert r["micro_batch_size"] == micro, (rank, r)
            assert r["hutchinson_distribution"] == distribution, (rank, r)
            if r["observable"] in _P_ROWS:
                assert r["preconditioner"] == "adamw", (rank, r)
                assert r["preconditioner_eps"] == ADAMW_EPS, (rank, r)
                assert r["preconditioner_beta2"] == ADAMW_BETA2, (rank, r)
                assert r["preconditioner_capped_fraction"] == zero_fraction, (rank, r)
            else:
                assert r["preconditioner"] == "none", (rank, r)
                assert r["preconditioner_eps"] is None, (rank, r)
                assert r["preconditioner_beta2"] is None, (rank, r)
                assert r["preconditioner_capped_fraction"] is None, (rank, r)


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_ddp_stderr_is_the_root_of_the_summed_rank_variances(tmp_path: Path, method: str) -> None:
    """``value_probe_stderr`` of a two-rank run, exactly, on the real ``torchrun`` path.

    Each rank's estimator reports the variance of its own partial ``chi_net``
    (a sum over its micro-batches); the all-reduce must sum those across the
    ranks, and the row's standard error is the square root. Checked against the
    estimator run here on each rank's shard and documented stream -- the shipped
    RNG, no statistics. Batch 5 over 2 ranks at ``micro_batch_size = 2``: rank 0
    has micro-batches of 2 and 1 sequences, rank 1 one of 2.

    Then the rows derived from ``chi_net``: ``chi_net_normalized`` and
    ``chi_pos`` (and ``chi_pos_P``) carry the same relative error, and the rows
    without a probe carry none. Every rank holds the same all-reduced numbers.
    """
    world_size, n_h, batch_size, micro = 2, 4, 5, 2
    rank_rows = _run_ddp_rows(
        tmp_path / f"ddp_stderr_{method}.json",
        seed=0,
        batch_size=batch_size,
        n_h=n_h,
        method=method,
        preconditioner=True,
        adamw=True,
        micro_batch_size=micro,
    )
    ddp = {r["observable"]: r for r in rank_rows[0]}
    bundle, _ = _toy_bundle(0)
    precond = adamw_preconditioner(sum(p.numel() for p in bundle.params))
    batch = make_tiny_lm_batch(batch_size=batch_size, seed=0, pad_fraction=0.25, device="cpu")
    call_seed = analyzer_mod._per_call_seed(0, "toy@step0", "val")
    per_rank = _rank_stream_variances(
        bundle,
        precond,
        batch,
        method=method,
        n_h=n_h,
        micro_batch_size=micro,
        call_seed=call_seed,
        world_size=world_size,
    )
    for observable in ("chi_net", "chi_net_P", "chi_net_P_capped"):
        parts = [rank[observable] for rank in per_rank]
        # vacuity guard: a rank that carries a share q of the summed variance
        # changes the standard error by a factor sqrt(1 - q) when it is left
        # out. With q > 1% for both ranks that is more than 0.5%, against a
        # comparison at DDP_REL = 1e-6 -- a missing all-reduce cannot pass.
        assert all(part > 0.01 * sum(parts) for part in parts), (observable, parts)
        want = sum(parts) ** 0.5
        got = ddp[observable]["value_probe_stderr"]
        assert abs(got - want) / want < DDP_REL, (method, observable, got, want)
        assert 0.0 < got < ddp[observable]["value"], (observable, got)

    def relative(observable: str) -> float:
        return ddp[observable]["value_probe_stderr"] / abs(ddp[observable]["value"])

    assert relative("chi_net_normalized") == pytest.approx(relative("chi_net"), rel=1e-12)
    assert relative("chi_pos") == pytest.approx(relative("chi_net"), rel=1e-12)
    assert relative("chi_pos_P") == pytest.approx(relative("chi_net_P"), rel=1e-12)
    for observable in ("chi_loss", "chi_loss_normalized", "delta_loss", "delta_loss_P"):
        assert ddp[observable]["value_probe_stderr"] is None, observable
    assert ddp["delta_loss_P_capped"]["value_probe_stderr"] is None

    other = {r["observable"]: r for r in rank_rows[1]}
    assert set(other) == set(ddp)
    for observable, row in ddp.items():
        assert other[observable]["value_probe_stderr"] == row["value_probe_stderr"], observable


def test_ddp_ranks_agree_on_which_parameters_received_a_gradient(tmp_path: Path) -> None:
    """A parameter that is in the graph on one rank's shard only.

    ``GatedTiny.gate`` gets a gradient from the one marked sequence of the
    update batch, which lies in rank 0's shard; on rank 1 autograd returns
    ``None`` for it. The saved state moves the gate (momentum, weight decay), so
    a rank that believed it had received no gradient would refuse to rebuild the
    step. The record of who received a gradient is therefore all-reduced: both
    ranks must proceed, and write the single-process numbers.

    Without that all-reduce rank 1 raises while rank 0 goes on to the next
    collective call; the run then fails or hangs, and the timeout turns a hang
    into a failure.
    """
    seed, batch_size = 0, 4
    ranks = _run_ddp_rows(
        tmp_path / "ddp_gated.json",
        seed=seed,
        batch_size=batch_size,
        n_h=2,
        method="hutchinson",
        update_response=True,
        gated=True,
        timeout=180,
    )

    def single_process(gated_rows: tuple[int, ...]) -> dict[tuple[str, str, str], float]:
        torch.manual_seed(seed)
        model = GatedTiny().eval()
        bundle = ModelBundle(
            model=model,
            params=list(model.parameters()),
            forward_fn=lambda m, b: m(b["input_ids"]),
            loss_fn=causal_lm_loss,
            valid_mask_fn=causal_lm_valid_mask,
            identifier="toy@step0",
        )
        batch = gate_batch(
            make_tiny_lm_batch(batch_size=batch_size, seed=seed, pad_fraction=0.25, device="cpu"),
            gated_rows,
        )
        other = gate_batch(
            make_tiny_lm_batch(
                batch_size=batch_size, seed=seed + 1, pad_fraction=0.25, device="cpu"
            ),
            (),
        )
        update = update_response_setup(bundle, batch)
        results = analyze(
            model=bundle,
            revisions=["step0"],
            eval_batches={"val": batch, "other": other},
            update_response=lambda ckpt_id: update,
            chi_net_method="hutchinson",
            n_hutchinson=2,
            micro_batch_size=1,
            sink=None,
            device="cpu",
        )
        return {
            (r.batch_a, r.batch_b, r.observable): r.value
            for r in results[0].rows
            if r.observable in UPDATE_RESPONSE_OBSERVABLES
        }

    # The premise: with no marked sequence at all the rule IS refused, because
    # the saved state moves the gate. So a rank deciding on its own shard would
    # refuse on rank 1.
    with pytest.raises(ValueError, match="receives no gradient on the update batch"):
        single_process(())
    # ... and the marked sequence is in rank 0's shard only.
    assert shard_batch({"x": torch.arange(batch_size)}, rank=0, world_size=2)[0]["x"].tolist() == [
        0,
        1,
    ]

    base = single_process((0,))
    assert len(base) == 2 * len(UPDATE_RESPONSE_OBSERVABLES)
    for rank, rows in enumerate(ranks):
        got = {
            (r["batch_a"], r["batch_b"], r["observable"]): r["value"]
            for r in rows
            if r["observable"] in UPDATE_RESPONSE_OBSERVABLES
        }
        assert set(got) == set(base), rank
        for key, expected in base.items():
            # The momentum part is the gradient dotted with this rule's seeded
            # random first moment: a sum of terms of random sign, far smaller
            # than the terms. The fp32 all-reduce of the gradient shows in it at
            # 1.05e-6 of its own size here (3.4e-9 absolute), so the bound is
            # taken against the largest part of the pair.
            scale = max(abs(base[(key[0], key[1], name)]) for name in UPDATE_RESPONSE_OBSERVABLES)
            assert abs(got[key] - expected) <= DDP_REL * scale, (rank, key, got[key], expected)


def test_ddp_rows_of_the_step_taken_follow_the_column_rule(tmp_path: Path) -> None:
    """The provenance columns on the ``loss_change*`` and step rows, on the real path.

    Two ranks, an AdamW ``P`` and an update rule in the same run, Gaussian probes
    at micro-batch 2 on the uneven split. On every rank the rows of the step
    taken carry the run settings like every other row (``n_ranks = 2``), none of
    the three ``preconditioner_*`` columns and no ``value_probe_stderr``; the
    rows of the AdamW ``P`` carry its ``eps``, and selecting them by observable
    name, by the descriptor ``"adamw"`` or by a non-null ``preconditioner_eps``
    gives the same rows — none of them a row tagged ``adamw_update``.
    """
    rank_rows = _run_ddp_rows(
        tmp_path / "ddp_step_taken_columns.json",
        seed=0,
        batch_size=5,
        n_h=4,
        method="hutchinson",
        preconditioner=True,
        adamw=True,
        micro_batch_size=2,
        distribution="gaussian",
        update_response=True,
    )
    mine_names = UPDATE_RESPONSE_OBSERVABLES + UPDATE_STEP_OBSERVABLES + REALIZED_CHANGE_OBSERVABLES
    assert len(rank_rows) == 2
    for rank, rows in enumerate(rank_rows):
        mine = [r for r in rows if r["observable"] in mine_names]
        assert {r["observable"] for r in mine} == set(mine_names) - {
            "update_lr_min",
            "update_lr_max",
        }, rank
        for r in mine:
            assert r["n_ranks"] == 2 and r["chi_net_method"] == "hutchinson", (rank, r)
            assert r["micro_batch_size"] == 2 and r["n_hutchinson"] == 4, (rank, r)
            assert r["hutchinson_distribution"] == "gaussian", (rank, r)
            assert r["hutchinson_seed"] is None and r["wallclock_s"] is None, (rank, r)
            assert r["preconditioner_eps"] is None, (rank, r)
            assert r["preconditioner_beta2"] is None, (rank, r)
            assert r["preconditioner_capped_fraction"] is None, (rank, r)
            assert r["value_probe_stderr"] is None, (rank, r)
            tagged = r["observable"] in UPDATE_RESPONSE_OBSERVABLES + UPDATE_STEP_OBSERVABLES
            assert r["preconditioner"] == ("adamw_update" if tagged else "none"), (rank, r)
        by_name = [i for i, r in enumerate(rows) if r["observable"] in _P_ROWS]
        by_descriptor = [i for i, r in enumerate(rows) if r["preconditioner"] == "adamw"]
        by_eps = [i for i, r in enumerate(rows) if r["preconditioner_eps"] is not None]
        assert by_name == by_descriptor == by_eps and by_name, rank
        assert all(rows[i]["preconditioner_eps"] == ADAMW_EPS for i in by_name), rank
        assert not any(rows[i]["observable"] in mine_names for i in by_name), rank
