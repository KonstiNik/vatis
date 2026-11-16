"""The ``*_capped`` rows: how much of a P observable sits at the ``1/eps`` ceiling.

AdamW's ``P = 1/(sqrt(v_hat) + eps)`` equals ``1/eps`` wherever the second moment
is zero, i.e. on every coordinate that has not received a non-zero gradient yet.
``P`` is built from the *pre-step* state, so a token the run sees for the first
time at step ``t`` has ``v = 0`` on its embedding row while its gradient on that
step's own batch is not zero: it enters ``delta_loss_P`` and ``chi_net_P`` with a
weight orders of magnitude above every coordinate with history. The fixture below
is that situation, and on it almost the whole of both observables comes from a
dozen coordinates.

Nothing is clipped. Instead the analyzer writes, next to each P observable, the
same quantity with ``P`` zeroed everywhere but at the ceiling
(``delta_loss_P_capped``, ``chi_net_P_capped``), so every published number carries
the information needed to judge it. These tests pin those rows against values
recomputed with plain torch from the optimizer's own state.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.fixtures.synthetic_bundle import (
    BundleTinyLM,
    bundle_metadata,
    tiny_lm_factory,
    write_checkpoint,
    write_run_files,
)
from vatis import Analyzer, load_local_bundle
from vatis.analyzer import P_CAPPED_OBSERVABLES, P_OBSERVABLES
from vatis.core.chi_net.hutchinson import HutchinsonEstimator
from vatis.core.chi_net.per_seq_cv import PerSequenceControlVariateEstimator
from vatis.core.preconditioner import (
    Preconditioner,
    epsilon_cap_mask,
    weighted_dot_fp64,
    weighted_sum_sq_fp64,
    weighted_sum_sq_grads,
)
from vatis.data.collate import shifted_lm_targets
from vatis.models.hf import causal_lm_loss, causal_lm_valid_mask
from vatis.models.local_bundle import plain_lm_forward

VOCAB, D_MODEL, SEQ_LEN, BATCH = 16, 6, 5, 3
BETAS, EPS, LR = (0.9, 0.999), 1e-8, 1e-3
STEP = 30
SEEN_VOCAB = 12  # tokens 12..15 never occur before step STEP


def _lm_batch(ids: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"input_ids": ids, "labels": shifted_lm_targets(ids)}


def _loss(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    logits = model(ids)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), shifted_lm_targets(ids).reshape(-1), ignore_index=-100
    )


@pytest.fixture
def first_seen_bundle(tmp_path: Path) -> dict[str, Any]:
    """30 real AdamW steps on tokens 0..11, then a batch that contains new tokens."""
    model = BundleTinyLM(vocab_size=VOCAB, d_model=D_MODEL, seq_len=SEQ_LEN, seed=0)
    run_dir = tmp_path / "run"
    order = write_run_files(
        run_dir,
        model,
        metadata=bundle_metadata(
            model, betas=BETAS, eps=EPS, batch_size=BATCH, seq_len=SEQ_LEN, lr=LR
        ),
    )
    named = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [named[n] for n in order], lr=LR, betas=BETAS, eps=EPS, weight_decay=0.0
    )
    rng = np.random.default_rng(0)
    for _ in range(STEP):
        ids = torch.from_numpy(rng.integers(0, SEEN_VOCAB, size=(BATCH, SEQ_LEN))).long()
        optimizer.zero_grad(set_to_none=True)
        _loss(model, ids).backward()
        optimizer.step()
    batch_a = rng.integers(0, VOCAB, size=(BATCH, SEQ_LEN), dtype=np.int32)
    batch_a[0, 0] = 13
    state = optimizer.state_dict()
    write_checkpoint(run_dir, STEP, model=model, optim_state=state, batch_a=batch_a)

    # The reference P, straight from the optimizer's state in fp64.
    v = torch.cat([state["state"][i]["exp_avg_sq"].reshape(-1) for i in range(len(order))])
    p_ref = 1.0 / ((v.double() / (1.0 - BETAS[1] ** STEP)).sqrt() + EPS)
    val = torch.from_numpy(
        np.random.default_rng(999).integers(0, VOCAB, size=(BATCH, SEQ_LEN))
    ).long()
    return {"run_dir": run_dir, "order": order, "p_ref": p_ref, "untouched": v == 0, "val": val}


def _flat_grad(source: Any, order: list[str], ids: torch.Tensor) -> torch.Tensor:
    """Gradient of the loss at the weights the loader rebuilt, by plain autograd."""
    model = source.checkpoints[0].model_bundle.model
    named = dict(model.named_parameters())
    grads = torch.autograd.grad(_loss(model, ids), [named[n] for n in order])
    return torch.cat([g.reshape(-1).double() for g in grads])


def _run(source: Any, val: torch.Tensor, precond: Any, **kwargs: Any) -> dict[Any, float]:
    analyzer = Analyzer(
        eval_batches={"train": source.batch_a_spec(), "val": _lm_batch(val)},
        preconditioner=precond,
        cross_pairs=[("train", "val")],
        chi_net_method="hutchinson",
        n_hutchinson=8,
        micro_batch_size=BATCH,
        dtype="fp32",
        sink=None,
        seed=0,
        **kwargs,
    )
    rows = [r for result in analyzer.run(source.model_specs()) for r in result.rows]
    return {(r.batch_a, r.batch_b, r.observable): r.value for r in rows} | {
        ("descriptor", r.observable): r.preconditioner for r in rows
    }


def test_capped_delta_loss_is_the_sum_over_coordinates_without_history(
    first_seen_bundle: dict[str, Any],
) -> None:
    b = first_seen_bundle
    source = load_local_bundle(b["run_dir"], tiny_lm_factory, dtype="fp32")
    values = _run(source, b["val"], source.adamw_preconditioner())

    ids_a = source.checkpoints[0].batch_a["input_ids"]
    g_a = _flat_grad(source, b["order"], ids_a)
    g_val = _flat_grad(source, b["order"], b["val"])
    p, untouched = b["p_ref"], b["untouched"]
    assert int(untouched.sum()) == (VOCAB - SEEN_VOCAB) * D_MODEL  # the unseen embedding rows

    # self pair on the update batch: the capped row is exactly the untouched part
    total = float((p * g_a * g_a).sum())
    capped = float((p * g_a * g_a)[untouched].sum())
    assert capped > 0.0  # the fixture really has a first-seen token in batch_a
    assert values[("train", "train", "delta_loss_P")] == pytest.approx(total, rel=1e-5)
    assert values[("train", "train", "delta_loss_P_capped")] == pytest.approx(capped, rel=1e-5)
    # ... and here it is nearly everything: a dozen coordinates carry the observable.
    assert capped / total > 0.9

    # cross pair: the capped part of the P-weighted dot
    cross_capped = float((p * g_a * g_val)[untouched].sum())
    assert values[("train", "val", "delta_loss_P_capped")] == pytest.approx(
        cross_capped, rel=1e-5, abs=1e-12
    )
    # the cross chi_net_P is a geometric mean; so is its capped row
    assert values[("train", "val", "chi_net_P_capped")] == pytest.approx(
        math.sqrt(
            values[("train", "train", "chi_net_P_capped")]
            * values[("val", "val", "chi_net_P_capped")]
        ),
        rel=1e-12,
    )
    for name in P_CAPPED_OBSERVABLES:
        assert values[("descriptor", name)] == "adamw"


def test_capped_chi_net_is_what_p_restricted_to_the_ceiling_measures(
    first_seen_bundle: dict[str, Any],
) -> None:
    """Three runs on the same probes: ``P``, ``P`` at the ceiling only, the rest."""
    b = first_seen_bundle
    source = load_local_bundle(b["run_dir"], tiny_lm_factory, dtype="fp32")
    full_p = source.adamw_preconditioner()(source.checkpoints[0].identifier)
    assert full_p is not None and full_p.ceiling == pytest.approx(1.0 / EPS)
    mask = epsilon_cap_mask(full_p.diag, EPS)
    assert int(mask.sum()) == full_p.count_capped() == int(b["untouched"].sum())

    full = _run(source, b["val"], full_p)
    only_capped = _run(source, b["val"], Preconditioner.from_diagonal(full_p.diag * mask))
    only_live = _run(source, b["val"], Preconditioner.from_diagonal(full_p.diag * ~mask))
    for pair in (("train", "train"), ("val", "val")):
        total = full[(*pair, "chi_net_P")]
        capped = full[(*pair, "chi_net_P_capped")]
        assert capped == pytest.approx(only_capped[(*pair, "chi_net_P")], rel=1e-6)
        assert total == pytest.approx(capped + only_live[(*pair, "chi_net_P")], rel=1e-5)
    assert (
        full[("train", "train", "chi_net_P_capped")] > 0.9 * full[("train", "train", "chi_net_P")]
    )


def _explicit_p_with_ceiling(n: int, *, at_ceiling: bool) -> tuple[torch.Tensor, float]:
    # The ceiling is only twice the largest other weight on purpose: with a
    # 1/eps-sized gap the capped part swamps everything, and a live term leaking
    # into it (e.g. the control variate's exact term) would vanish in the
    # tolerance.
    generator = torch.Generator().manual_seed(11)
    diag = 0.5 + 49.5 * torch.rand(n, generator=generator)
    ceiling = 100.0
    if at_ceiling:
        diag[torch.randperm(n, generator=generator)[: n // 20]] = ceiling
    return diag, ceiling


@pytest.mark.parametrize(
    "estimator",
    [
        HutchinsonEstimator(n_hutchinson=6),
        PerSequenceControlVariateEstimator(n_hutchinson=6, compute_alignment_matrix=False),
        PerSequenceControlVariateEstimator(n_hutchinson=6, compute_alignment_matrix=True),
    ],
    ids=["hutchinson", "per_sequence_cv", "per_sequence_cv_stored_grads"],
)
def test_every_estimator_path_splits_the_trace_exactly(estimator: Any) -> None:
    model = BundleTinyLM(vocab_size=VOCAB, d_model=D_MODEL, seq_len=SEQ_LEN, seed=3).eval()
    params = list(model.parameters())
    n = sum(p.numel() for p in params)
    ids = torch.randint(0, VOCAB, (4, SEQ_LEN), generator=torch.Generator().manual_seed(5))
    diag, ceiling = _explicit_p_with_ceiling(n, at_ceiling=True)
    mask = diag >= ceiling

    def trace(precond: Preconditioner) -> Any:
        return estimator.compute(
            model,
            _lm_batch(ids),
            plain_lm_forward,
            loss_fn=causal_lm_loss,
            valid_mask_fn=causal_lm_valid_mask,
            params=params,
            micro_batch_size=2,
            generator=torch.Generator().manual_seed(7),
            preconditioner=precond,
        )

    with_ceiling = trace(Preconditioner.from_diagonal(diag, ceiling=ceiling))
    without = trace(Preconditioner.from_diagonal(diag))
    only_capped = trace(Preconditioner.from_diagonal(diag * mask))
    only_live = trace(Preconditioner.from_diagonal(diag * ~mask))

    # declaring a ceiling changes neither the raw nor the P estimate
    assert without.chi_net_p_capped is None
    assert torch.equal(with_ceiling.chi_net, without.chi_net)
    assert torch.equal(with_ceiling.chi_net_p, without.chi_net_p)
    # the capped row is the P-restricted estimate, and the two parts add up
    capped = float(with_ceiling.chi_net_p_capped)
    assert 0.0 < capped < float(with_ceiling.chi_net_p)
    assert capped == pytest.approx(float(only_capped.chi_net_p), rel=1e-6)
    assert float(with_ceiling.chi_net_p) == pytest.approx(
        capped + float(only_live.chi_net_p), rel=1e-5
    )


def test_a_ceiling_nobody_sits_at_gives_zero_rows_and_leaves_the_p_rows_alone(
    first_seen_bundle: dict[str, Any],
) -> None:
    b = first_seen_bundle
    source = load_local_bundle(b["run_dir"], tiny_lm_factory, dtype="fp32")
    diag, ceiling = _explicit_p_with_ceiling(source.layout.total_numel, at_ceiling=False)
    with_ceiling = _run(source, b["val"], Preconditioner.from_diagonal(diag, ceiling=ceiling))
    without = _run(source, b["val"], Preconditioner.from_diagonal(diag))
    for pair in (("train", "train"), ("val", "val"), ("train", "val")):
        for name in P_CAPPED_OBSERVABLES:
            assert with_ceiling[(*pair, name)] == 0.0
            assert (*pair, name) not in without
        for name in P_OBSERVABLES:
            assert with_ceiling[(*pair, name)] == without[(*pair, name)]


@pytest.mark.parametrize("precond", ["identity", "sgd-like"])
def test_a_p_without_a_ceiling_has_no_capped_rows(
    first_seen_bundle: dict[str, Any], precond: str
) -> None:
    b = first_seen_bundle
    source = load_local_bundle(b["run_dir"], tiny_lm_factory, dtype="fp32")
    spec: Any = (
        "identity"
        if precond == "identity"
        else Preconditioner.identity(source.layout.total_numel, descriptor="sgd")
    )
    values = _run(source, b["val"], spec)
    assert ("train", "train", "chi_pos_P") in values
    assert not [key for key in values if key[-1] in P_CAPPED_OBSERVABLES]


def test_capped_rows_need_a_preconditioner() -> None:
    with pytest.raises(ValueError, match="no preconditioner was given"):
        Analyzer(eval_batches={"a": torch.zeros(1, 2)}, observables=["delta_loss_P_capped"])


def test_the_adamw_builder_records_its_ceiling() -> None:
    v = torch.tensor([0.0, 1e-4, 0.0, 4e-2])
    precond = Preconditioner.adamw(v, step=5, beta2=0.999, eps=1e-8)
    assert precond.ceiling == pytest.approx(1e8)
    assert precond.count_capped() == 2
    assert precond.to(dtype=torch.float64).ceiling == precond.ceiling
    assert Preconditioner.adamw(v + 1e-6, step=5, beta2=0.999, eps=0.0).ceiling is None
    assert Preconditioner.identity(4).ceiling is None
    assert Preconditioner.identity(4).count_capped() == 0
    with pytest.raises(ValueError, match="ceiling"):
        Preconditioner.from_diagonal(torch.ones(3), ceiling=0.0)


# ------------------------------------ what "at the ceiling" means, at its boundary


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_the_ceiling_criterion_is_pinned_at_its_boundary(dtype: torch.dtype) -> None:
    """``P_i >= (1 - 1e-6) * ceiling``: just inside counts, just outside does not.

    No other test fixes the width of that band — every other fixture has entries
    either exactly at the ceiling or orders of magnitude below it, so the
    relative slack could be widened a thousandfold without a failure.
    """
    ceiling = 1e8
    # fp32 values next to 1e8 are 8 apart, so both of these are exact in fp32 too.
    inside = ceiling - 88.0  # 0.88e-6 below the ceiling, relative
    outside = ceiling - 112.0  # 1.12e-6 below
    diag = torch.tensor([ceiling, inside, outside, 0.5 * ceiling, 1.0], dtype=dtype)
    assert diag.tolist() == [ceiling, inside, outside, 0.5 * ceiling, 1.0]
    at_ceiling = torch.tensor([True, True, False, False, False])

    precond = Preconditioner.from_diagonal(diag, ceiling=ceiling)
    floor = precond.capped_weight_floor
    assert floor == pytest.approx(ceiling * (1.0 - 1e-6), rel=1e-12)
    assert outside < floor < inside
    assert precond.count_capped() == 2
    assert torch.equal(epsilon_cap_mask(diag, 1.0 / ceiling), at_ceiling)

    # The reductions the ``*_capped`` rows are computed with draw the same line.
    g = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    expected = ceiling * 1.0 + inside * 4.0
    assert float(weighted_sum_sq_fp64(g, diag, min_weight=floor)) == pytest.approx(
        expected, rel=1e-12
    )
    assert float(weighted_dot_fp64(g, g, diag, min_weight=floor)) == pytest.approx(
        expected, rel=1e-12
    )
    params = [torch.nn.Parameter(torch.zeros(2, dtype=dtype)) for _ in range(2)]
    params.append(torch.nn.Parameter(torch.zeros(1, dtype=dtype)))
    by_parameter = weighted_sum_sq_grads(
        [g[0:2], g[2:4], g[4:5]],
        precond.parameter_views(params),
        device=torch.device("cpu"),
        min_weight=floor,
    )
    assert float(by_parameter) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_a_second_moment_of_the_order_of_eps_is_not_at_the_ceiling(dtype: torch.dtype) -> None:
    """Through the AdamW builder: only a second moment that is zero to rounding counts.

    ``P = 1/(sqrt(v_hat) + eps)`` lies ``d/(1 + d)`` below ``1/eps`` (relative) for
    ``d = sqrt(v_hat)/eps``. A coordinate with ``sqrt(v_hat) = eps`` therefore has
    half the ceiling weight — still dominated by ``eps`` — and is **not** counted,
    which makes the ``*_capped`` rows a lower bound on the eps-dominated share.
    """
    eps, beta2, step = 1e-8, 0.999, 5
    ratio = torch.tensor(
        [0.0, 0.5e-6, 2e-6, 1.0, 10.0, 1e5], dtype=torch.float64
    )  # sqrt(v_hat)/eps
    v = ((ratio * eps) ** 2 * (1.0 - beta2**step)).to(dtype)
    precond = Preconditioner.adamw(v, step=step, beta2=beta2, eps=eps)
    assert precond.ceiling == pytest.approx(1.0 / eps)

    at_ceiling = torch.tensor([True, True, False, False, False, False])
    assert torch.equal(epsilon_cap_mask(precond.diag, eps), at_ceiling)
    assert torch.equal(precond.diag >= precond.capped_weight_floor, at_ceiling)
    assert precond.count_capped() == 2
    # sqrt(v_hat) == eps: half the ceiling, and outside the band by a wide margin
    assert float(precond.diag[3]) == pytest.approx(0.5 / eps, rel=1e-6)
    assert float(precond.diag[4]) == pytest.approx(1.0 / (11.0 * eps), rel=1e-6)
