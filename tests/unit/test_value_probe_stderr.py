"""``value_probe_stderr``: the standard error of the sampled values, checked four ways.

**What is reported.** ``chi_net`` is a sum over micro-batches of a mean of
``n_hutchinson`` independent per-probe values (plus, for ``per_sequence_cv``, a
term no probe enters). Its variance under the probe draw is therefore the sum
over micro-batches of ``s^2 / n`` (``s^2`` the unbiased sample variance of that
micro-batch's per-probe values), summed over the ranks of a DDP run, and
``value_probe_stderr`` is its square root. ``chi_net_normalized`` is ``chi_net`` over a
constant, ``chi_pos = delta_loss / (chi_loss * chi_net)`` has ``chi_net`` as its
only sampled factor and inherits its relative error to first order, and a cross
pair combines two independently probed batches through a geometric mean. The
``_P`` and ``_P_capped`` rows are the same with their own per-probe values.

**How it is checked.**

1. *Probe for probe* (no statistics): the probes are regenerated with the
   estimator's own seed, the per-probe values are computed from an explicit
   Jacobian, and ``ChiNetResult.chi_net_var`` must be the sum over micro-batches
   of their ``s^2 / n``. This pins the definition, the ``n - 1``, the ``1 / n``
   and the sum over micro-batches, on a padded transformer batch.
2. *Against the exact variance*: the variance of one per-probe value
   ``v^T A v`` is ``2 sum_{i != j} A_ij^2`` for Rademacher probes and
   ``2 sum_{i, j} A_ij^2`` for Gaussian ones (``A`` the micro-batch's
   output-space Gram matrix, projected for ``per_sequence_cv``), so the true
   variance of ``chi_net`` is known without sampling. The reported variance,
   averaged over seeds, must equal it, for both distributions.
3. *Against the empirical spread*, through ``analyze()``: over many analyzer
   seeds, the mean of ``value_probe_stderr**2`` must equal the sample variance of
   ``value``, for every row type that carries a standard error — both
   estimators, two micro-batch sizes, an AdamW ``P`` with coordinates at its
   ceiling, the propagated ``chi_pos`` rows and the cross pair. The same for the
   per-rank partial sums of an emulated two-rank run.
4. *Row arithmetic*: the propagated rows are exactly the stated functions of the
   rows they are derived from, and the rows that are not sampled carry none.

**Tolerances** are derived where they are used, from the sampling error of the
comparison itself; none was adjusted to make a test pass. Everything is seeded,
so each outcome is deterministic.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

import vatis.analyzer as analyzer_mod
from tests.fixtures.preconditioned_toys import (
    Fp64MLP,
    all_valid_mask,
    ce_loss,
    rademacher_trace_std,
    tuple_forward,
)
from tests.fixtures.tiny_transformer import (
    TinyTransformer,
    causal_lm_loss,
    causal_lm_valid_mask,
    make_tiny_lm_batch,
)
from tests.unit.test_chi_net import _build_full_jacobian_and_u
from vatis import analyze
from vatis.core.chi_net.base import ChiNetResult, ProbeSpread
from vatis.core.chi_net.hutchinson import HutchinsonEstimator
from vatis.core.chi_net.per_seq_cv import PerSequenceControlVariateEstimator
from vatis.core.normalization import normalization_factor
from vatis.core.preconditioner import Preconditioner
from vatis.core.probes import ProbeDistribution, make_probe, mask_probe
from vatis.models.hf import ModelBundle
from vatis.sinks.base import ResultRow

FP64 = torch.float64

# Number of standard errors every statistical comparison below allows. A normal
# variable leaves +-4 sigma with probability 6.3e-5; the statistical tests of
# this file make 82 such comparisons (many of them on the same runs), so if every
# reported standard error is right the chance that some comparison fails is below
# 82 * 6.3e-5 = 0.5%. The seeds are fixed, so on one torch build and platform the
# outcome is the same on every run; another build can draw different probes or
# round differently, and is then a new draw from that 0.5%.
Z_SIGMA = 4.0

# Rows whose standard error comes straight from the per-probe spread ...
DIRECT = ("chi_net", "chi_net_normalized", "chi_net_P", "chi_net_P_capped")
# ... and rows that inherit it through a ratio: observable -> its sampled factor.
PROPAGATED = {"chi_pos": "chi_net", "chi_pos_P": "chi_net_P"}
# Rows that involve no probe.
NOT_SAMPLED = (
    "chi_loss",
    "chi_loss_normalized",
    "delta_loss",
    "delta_loss_P",
    "delta_loss_P_capped",
)


# ======================================================================= ProbeSpread


def test_probe_spread_is_the_unbiased_variance_of_the_mean() -> None:
    gen = torch.Generator().manual_seed(0)
    values = torch.rand(7, generator=gen, dtype=FP64) * 3.0 + 5.0
    spread = ProbeSpread(torch.device("cpu"))
    for i, v in enumerate(values):
        # Undefined below two values: the spread of one number is not zero.
        assert (spread.variance_of_mean() is None) == (i < 2)
        spread.add(v)
    got = spread.variance_of_mean()
    assert got is not None and got.dtype == FP64
    assert float(got) == pytest.approx(float(values.var(unbiased=True) / 7), rel=1e-12)
    total = torch.tensor(2.5, dtype=FP64)
    assert float(spread.add_variance_of_mean_to(total)) == pytest.approx(2.5 + float(got))
    assert ProbeSpread(torch.device("cpu")).add_variance_of_mean_to(total) is total


def test_probe_spread_keeps_a_spread_far_below_the_mean() -> None:
    """Values equal to 1e-9 relative: ``sum q^2 - (sum q)^2 / n`` would lose all
    but a digit of this in fp64; the sums relative to the first value do not."""
    base = 1.0e6
    offsets = torch.tensor([0.0, 1.0, -2.0, 3.0, 0.5], dtype=FP64) * 1e-3
    spread = ProbeSpread(torch.device("cpu"))
    for off in offsets:
        spread.add(torch.tensor(base, dtype=FP64) + off)
    got = spread.variance_of_mean()
    assert got is not None
    assert float(got) == pytest.approx(float(offsets.var(unbiased=True) / 5), rel=1e-6)

    constant = ProbeSpread(torch.device("cpu"))
    for _ in range(4):
        constant.add(torch.tensor(base, dtype=FP64))
    zero = constant.variance_of_mean()
    assert zero is not None and float(zero) == 0.0


# ============================================ 1. probe for probe, on a padded LM batch

_VOCAB, _SEQ = 16, 8
# Relative error of one per-probe value as the estimator computes it: an fp32
# backward through a model of this depth, squared and summed in fp64. The
# sibling file ``test_chi_net_probe_exactness.py`` asserts the *mean* of these
# values to 8e-7; a single value is allowed the same.
_PER_PROBE_REL = 8e-7


def _lm_case() -> dict[str, Any]:
    """A padded 3-sequence batch on a tiny transformer, with its explicit Jacobian.

    ``weights`` holds the three diagonal weights a per-probe value can carry: all
    ones (raw), a spread ``P`` with a ceiling, and that ``P`` restricted to the
    entries at the ceiling (the capped part).
    """
    torch.manual_seed(42)
    model = TinyTransformer(
        vocab_size=_VOCAB, seq_len=_SEQ, d_model=16, n_layers=1, n_heads=2, d_ff=32
    ).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    batch = make_tiny_lm_batch(
        batch_size=3, seq_len=_SEQ, vocab_size=_VOCAB, pad_fraction=0.25, seed=0
    )
    mask = batch["labels"] != -100
    jac, u, _n_valid = _build_full_jacobian_and_u(model, batch)
    # Row range of every sequence in the valid-rows-only Jacobian.
    groups, off = [], 0
    for n_pos in mask.sum(dim=1).tolist():
        groups.append((off, off + int(n_pos) * _VOCAB))
        off += int(n_pos) * _VOCAB
    assert len({hi - lo for lo, hi in groups}) > 1, "padding should make the sequences differ"

    gen = torch.Generator().manual_seed(3)
    ceiling = 20.0
    diag = torch.exp(torch.randn(n_params, generator=gen) * 1.5).clamp_(max=0.5 * ceiling)
    diag[torch.rand(n_params, generator=gen) < 0.15] = ceiling
    precond = Preconditioner.from_diagonal(diag, descriptor="explicit", ceiling=ceiling)
    floor = precond.capped_weight_floor
    assert floor is not None and 0 < precond.count_capped() < n_params
    p64 = diag.to(FP64)
    weights = {
        "raw": torch.ones(n_params, dtype=FP64),
        "p": p64,
        "capped": torch.where(p64 >= floor, p64, torch.zeros_like(p64)),
    }
    # The output-space projector of per_sequence_cv, block-diagonal per sequence.
    pi = torch.zeros(jac.shape[0], jac.shape[0], dtype=FP64)
    for lo, hi in groups:
        block = u[lo:hi]
        pi[lo:hi, lo:hi] = torch.outer(block, block) / float((block * block).sum())
    return {
        "model": model,
        "params": params,
        "batch": batch,
        "mask": mask,
        "jac": jac,
        "groups": groups,
        "precond": precond,
        "weights": weights,
        "residual_projector": torch.eye(jac.shape[0], dtype=FP64) - pi,
    }


def _probe_matrices(case: dict[str, Any], method: str) -> dict[str, torch.Tensor]:
    """``A`` with per-probe value ``v^T A v``, for each of the three weights.

    ``hutchinson``: ``J W J^T``. ``per_sequence_cv``: the same between the
    residual projectors -- its other term is exact and has no spread.
    """
    jac = case["jac"]
    out = {}
    for name, w in case["weights"].items():
        a = (jac * w.unsqueeze(0)) @ jac.T
        if method.startswith("per_sequence_cv"):
            proj = case["residual_projector"]
            a = proj @ a @ proj
        out[name] = a
    return out


def _micro_row_ranges(case: dict[str, Any], micro: int) -> list[tuple[int, int, int, int]]:
    """``(first sequence, one past the last, first row, one past the last row)``."""
    groups, n = case["groups"], len(case["groups"])
    return [
        (start, min(n, start + micro), groups[start][0], groups[min(n, start + micro) - 1][1])
        for start in range(0, n, micro)
    ]


def _estimate(
    case: dict[str, Any],
    method: str,
    *,
    n_h: int,
    micro: int,
    seed: int,
    distribution: ProbeDistribution = "rademacher",
) -> ChiNetResult:
    estimator: HutchinsonEstimator | PerSequenceControlVariateEstimator
    if method == "hutchinson":
        estimator = HutchinsonEstimator(n_hutchinson=n_h, distribution=distribution)
    else:
        estimator = PerSequenceControlVariateEstimator(
            n_hutchinson=n_h,
            distribution=distribution,
            compute_alignment_matrix=method.endswith(":store"),
        )
    return estimator.compute(
        case["model"],
        case["batch"],
        lambda m, b: m(b["input_ids"]),
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        params=case["params"],
        micro_batch_size=micro,
        generator=torch.Generator().manual_seed(seed),
        preconditioner=case["precond"],
    )


@pytest.mark.parametrize("micro", [1, 2, 3])
@pytest.mark.parametrize("n_h", [2, 5])
@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv", "per_sequence_cv:store"])
def test_reported_variance_is_the_summed_spread_of_the_probes_drawn(
    method: str, n_h: int, micro: int
) -> None:
    """``chi_net_var`` (and the P / capped ones) for one seed, with no expectation.

    The estimator draws, per micro-batch, ``n_h`` probes from its generator in
    order. Regenerating exactly those and evaluating ``v^T A v`` on the explicit
    Jacobian gives the per-probe values ``q_{m,i}`` it must have seen, hence

        chi_net_var = sum_m  [ sum_i (q_{m,i} - mean_m)^2 / (n_h - 1) ] / n_h.

    Tolerance: the estimator's own ``q`` differ from the fp64 ones by at most
    ``eps = 8e-7`` relative (fp32 backward), and ``d var / d q_i =
    2 (q_i - mean) / ((n - 1) n)``, so the variance may differ by
    ``2 eps sum_i |q_i - mean| |q_i| / ((n - 1) n)`` per micro-batch -- computed
    here, not chosen. ``micro = 2`` cuts the three sequences into 2 + 1.
    """
    case = _lm_case()
    seed = 20261117
    matrices = _probe_matrices(case, method)
    expected = dict.fromkeys(matrices, 0.0)
    slack = dict.fromkeys(matrices, 0.0)
    gen = torch.Generator().manual_seed(seed)
    for start, stop, lo, hi in _micro_row_ranges(case, micro):
        mask = case["mask"][start:stop]
        per_probe: dict[str, list[float]] = {name: [] for name in matrices}
        for _ in range(n_h):
            probe = make_probe(
                (stop - start, _SEQ, _VOCAB),
                distribution="rademacher",
                device="cpu",
                dtype=torch.float32,
                generator=gen,
            )
            v = mask_probe(probe, mask).to(FP64).reshape(-1, _VOCAB)[mask.reshape(-1)].reshape(-1)
            for name, a in matrices.items():
                per_probe[name].append(float(v @ a[lo:hi, lo:hi] @ v))
        for name, values in per_probe.items():
            q = torch.tensor(values, dtype=FP64)
            expected[name] += float(q.var(unbiased=True)) / n_h
            slack[name] += (
                2.0 * _PER_PROBE_REL * float(((q - q.mean()).abs() * q.abs()).sum())
            ) / ((n_h - 1) * n_h)

    result = _estimate(case, method, n_h=n_h, micro=micro, seed=seed)
    reported = {
        "raw": result.chi_net_var,
        "p": result.chi_net_p_var,
        "capped": result.chi_net_p_capped_var,
    }
    for name, want in expected.items():
        got = reported[name]
        assert got is not None and got.dtype == FP64, name
        assert want > 0.0, (name, "vacuous: no spread to compare")
        assert abs(float(got) - want) <= slack[name], (
            f"{method} n_h={n_h} micro={micro} {name}: reported variance {float(got)!r}, "
            f"spread of the regenerated probes {want!r} "
            f"(off by {abs(float(got) - want):.3e}, fp32 allowance {slack[name]:.3e})"
        )
        # The allowance is a round-off bound, far below anything a wrong formula
        # produces (n vs n - 1 is a factor 2 at n_h = 2 and 1.25 at n_h = 5).
        assert slack[name] < 1e-3 * want, (name, slack[name], want)


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv", "per_sequence_cv:store"])
def test_one_probe_per_micro_batch_reports_no_variance(method: str) -> None:
    """A single value has no spread: ``None``, not zero, and not a guess."""
    result = _estimate(_lm_case(), method, n_h=1, micro=2, seed=1)
    assert result.chi_net_p is not None and result.chi_net_p_capped is not None
    assert result.chi_net_var is None
    assert result.chi_net_p_var is None
    assert result.chi_net_p_capped_var is None


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_variances_are_absent_exactly_where_the_estimates_are(method: str) -> None:
    case = _lm_case()
    no_p = {**case, "precond": None}
    result = _estimate(no_p, method, n_h=3, micro=2, seed=1)
    assert result.chi_net_var is not None
    assert result.chi_net_p is None and result.chi_net_p_var is None
    assert result.chi_net_p_capped is None and result.chi_net_p_capped_var is None

    # A P without a ceiling has no capped part; one with a ceiling nobody sits at
    # has a capped part that is exactly zero for every probe -- variance zero.
    diag = case["weights"]["p"].to(torch.float32).clamp(max=5.0)
    without = {**case, "precond": Preconditioner.from_diagonal(diag)}
    result = _estimate(without, method, n_h=3, micro=2, seed=1)
    assert result.chi_net_p_var is not None and result.chi_net_p_capped_var is None
    unreached = {**case, "precond": Preconditioner.from_diagonal(diag, ceiling=1e3)}
    result = _estimate(unreached, method, n_h=3, micro=2, seed=1)
    assert result.chi_net_p_capped is not None and float(result.chi_net_p_capped) == 0.0
    assert result.chi_net_p_capped_var is not None
    assert float(result.chi_net_p_capped_var) == 0.0


# ================================================== 2. against the exact variance


def _exact_probe_variance(a: torch.Tensor, distribution: str) -> float:
    """``Var[v^T A v]`` for one probe ``v`` with iid entries and symmetric ``A``.

    Rademacher: ``2 sum_{i != j} A_ij^2`` -- ``v_i^2 = 1`` removes the diagonal
    (``rademacher_trace_std``, squared, at one probe). Gaussian: by Isserlis,
    ``E[v_i v_j v_k v_l] = d_ij d_kl + d_ik d_jl + d_il d_jk``, hence
    ``E[(v^T A v)^2] = (Tr A)^2 + 2 ||A||_F^2`` and the variance is
    ``2 ||A||_F^2 = 2 sum_{i, j} A_ij^2`` -- the diagonal counts.
    """
    if distribution == "rademacher":
        return rademacher_trace_std(a, 1) ** 2
    assert distribution == "gaussian"
    return 2.0 * float((a * a).sum())


@pytest.mark.slow
@pytest.mark.parametrize("distribution", ["rademacher", "gaussian"])
@pytest.mark.parametrize("micro", [1, 3])
@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_reported_variance_is_unbiased_for_the_exact_probe_variance(
    method: str, micro: int, distribution: ProbeDistribution
) -> None:
    """Averaged over seeds, the reported variance is the variance that is known exactly.

    One probe ``v`` gives a per-probe value ``v^T A v`` whose variance is known
    in closed form for both probe distributions (``_exact_probe_variance``).
    The probes of different micro-batches are independent, so with ``A_m`` the
    block of micro-batch ``m``

        Var[chi_net] = sum_m Var[v^T A_m v] / n_h

    -- no sampling on this side of the comparison. ``micro = 3`` puts all three
    sequences under one probe, so ``A_m`` includes the cross-sequence blocks; at
    ``micro = 1`` it does not, and the two variances differ.

    Tolerance, the same for both distributions: the mean of ``K`` independent
    reported variances has standard error ``sd / sqrt(K)``, ``sd`` their sample
    standard deviation; ``Z_SIGMA`` of those.

    Number of seeds: enough for that tolerance to be below a quarter of the
    exact value (the guard at the end). With four probes one micro-batch yields
    a three-degrees-of-freedom spread estimate, whose relative standard
    deviation is about 1 for either distribution; three micro-batches halve it.
    Hence 150 seeds at ``micro = 1`` and 300 at ``micro = 3``.
    """
    case = _lm_case()
    n_h = 4
    n_seeds = {1: 150, 3: 300}[micro]
    matrices = _probe_matrices(case, method)

    def exact_variance(a: torch.Tensor, cut: int, dist: str) -> float:
        return sum(
            _exact_probe_variance(a[lo:hi, lo:hi], dist) / n_h
            for _s, _e, lo, hi in _micro_row_ranges(case, cut)
        )

    exact = {name: exact_variance(a, micro, distribution) for name, a in matrices.items()}
    reported: dict[str, list[float]] = {name: [] for name in matrices}
    for seed in range(n_seeds):
        result = _estimate(
            case, method, n_h=n_h, micro=micro, seed=1000 + seed, distribution=distribution
        )
        assert result.chi_net_var is not None
        assert result.chi_net_p_var is not None
        assert result.chi_net_p_capped_var is not None
        reported["raw"].append(float(result.chi_net_var))
        reported["p"].append(float(result.chi_net_p_var))
        reported["capped"].append(float(result.chi_net_p_capped_var))
    for name, want in exact.items():
        values = torch.tensor(reported[name], dtype=FP64)
        mean = float(values.mean())
        stderr = float(values.std(unbiased=True)) / math.sqrt(n_seeds)
        assert abs(mean - want) <= Z_SIGMA * stderr, (
            f"{method} micro={micro} {distribution} {name}: mean reported variance "
            f"{mean:.6e} over {n_seeds} seeds, exact {want:.6e} "
            f"({(mean - want) / stderr:+.2f} standard errors, ratio {mean / want:.3f})"
        )
        # vacuity guard: the comparison resolves the two formulas a slip would
        # most plausibly produce -- dividing by n instead of n - 1 (3/4 of the
        # exact value at n_h = 4), or not dividing by n at all (4 times it).
        assert Z_SIGMA * stderr < 0.25 * want, (name, stderr, want)
        if name == "raw":
            # ... and the probe distribution: the measured variance is that of
            # *this* distribution, not of the other one, whose closed form
            # differs by the diagonal of A. Asserted on the raw row only, where
            # the diagonal's share is largest; for the P and capped rows at
            # ``micro = 3`` the two closed forms are 18-32% apart, about the
            # size of the tolerance, and nothing is claimed there.
            other = "gaussian" if distribution == "rademacher" else "rademacher"
            other_exact = exact_variance(matrices[name], micro, other)
            assert abs(mean - other_exact) > Z_SIGMA * stderr, (distribution, mean, other_exact)

    if method == "hutchinson" and distribution == "rademacher":
        # ... and the micro-batch cut is part of what is reported.
        one = exact_variance(matrices["raw"], 1, distribution)
        whole = exact_variance(matrices["raw"], 3, distribution)
        assert whole > 1.2 * one, (one, whole)


# =========================== 3. through analyze(), against the spread over many seeds

_EPS, _BETA2 = 0.5, 0.99


def _mlp_bundle() -> tuple[ModelBundle, Preconditioner, float]:
    """An fp64 MLP classifier and an AdamW ``P`` with coordinates at its ceiling.

    The second moment is zero on a seeded fifth of the coordinates, so
    ``P = 1/eps`` there; ``eps`` is chosen large (0.5) so that those coordinates
    carry a share of ``chi_net_P`` rather than all of it. Returns the bundle,
    the ``P`` and the fraction of zero second moments.
    """
    model = Fp64MLP(in_dim=6, hidden=8, n_out=6, seed=0)
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    gen = torch.Generator().manual_seed(5)
    exp_avg_sq = torch.rand(n_params, generator=gen, dtype=FP64) * 3.9 + 0.1
    exp_avg_sq[torch.rand(n_params, generator=gen) < 0.2] = 0.0
    precond = Preconditioner.adamw(exp_avg_sq, step=10, beta2=_BETA2, eps=_EPS)
    bundle = ModelBundle(
        model=model,
        params=params,
        forward_fn=tuple_forward,
        loss_fn=ce_loss,
        valid_mask_fn=all_valid_mask,
        identifier="mlp@step10",
    )
    return bundle, precond, float((exp_avg_sq == 0).to(FP64).mean())


def _mlp_batch(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, 6, generator=gen, dtype=FP64),
        torch.randint(0, 6, (n,), generator=gen),
    )


RowKey = tuple[str, str, str]


def _rows_by_key(rows: list[ResultRow]) -> dict[RowKey, ResultRow]:
    return {(r.batch_a, r.batch_b, r.observable): r for r in rows}


def _collect(
    n_seeds: int, **analyze_kwargs: Any
) -> tuple[dict[RowKey, torch.Tensor], dict[RowKey, torch.Tensor]]:
    """``value`` and ``value_probe_stderr`` of every row that has one, over ``n_seeds`` runs."""
    values: dict[RowKey, list[float]] = {}
    stderrs: dict[RowKey, list[float]] = {}
    for seed in range(n_seeds):
        results = analyze(revisions=["step10"], sink=None, seed=seed, **analyze_kwargs)
        for key, row in _rows_by_key(results[0].rows).items():
            if row.value_probe_stderr is None:
                continue
            values.setdefault(key, []).append(row.value)
            stderrs.setdefault(key, []).append(row.value_probe_stderr)
    assert all(len(v) == n_seeds for v in values.values())
    return (
        {k: torch.tensor(v, dtype=FP64) for k, v in values.items()},
        {k: torch.tensor(v, dtype=FP64) for k, v in stderrs.items()},
    )


def _reported_minus_empirical(x: torch.Tensor, s: torch.Tensor) -> tuple[float, float, float]:
    """``(V_emp - V_rep, its standard error, V_rep)`` for values ``x`` with errors ``s``.

    ``V_rep = mean_k s_k^2`` is the reported variance, ``V_emp = sum_k (x_k -
    mean)^2 / (K - 1)`` the unbiased sample variance of the values. Their
    difference is the mean of

        d_k = K / (K - 1) * (x_k - mean)^2 - s_k^2,

    ``K`` terms that are independent apart from the shared sample mean (an
    ``O(1/K)`` coupling), so its standard error is ``sd(d_k) / sqrt(K)``. Taking
    the difference per run, rather than adding the standard errors of ``V_emp``
    and ``V_rep``, accounts for their covariance: both come from the same runs.
    """
    k = x.numel()
    d = (x - x.mean()) ** 2 * (k / (k - 1)) - s * s
    return float(d.mean()), float(d.std(unbiased=True)) / math.sqrt(k), float((s * s).mean())


def _first_order_allowance(factors: list[torch.Tensor], n_seeds: int) -> float:
    """Relative bias the first-order propagation is allowed, from its own next order.

    Let ``x`` be a sampled factor with relative standard deviation ``r`` and
    skewness ``g``, and ``f = c * x**p`` (``p = -1`` for a self-pair ``chi_pos``,
    ``p = +-1/2`` per batch in a cross pair). Expanding in ``(x - mean) / mean``:

        Var[f]                    = p^2 r^2 f^2 * (1 + (p - 1) g r + O(r^2))
        E[reported stderr^2]      = p^2 r^2 f^2 * (1 + 2 (p - 1) g r + O(r^2))

    (the second because the reported value divides the unbiased variance
    estimate by the *estimated* ``x``, and that estimate co-varies with ``x``
    through its third moment). They differ by ``(p - 1) g r`` at leading order
    and by ``c2 r^2`` at the next, with ``|p - 1| <= 2`` and ``|c2| <= 2`` for the
    three exponents (``c2 = 2, 1/8, 9/8`` for ``p = -1, 1/2, -1/2``; for a
    Gaussian ``x`` and ``p = -1``: ``Var[1/x] = (r^2 / mean^2)(1 + 8 r^2)`` against
    ``E[1/x^4] = (1 + 10 r^2) / mean^4``). A cross pair has two independent
    factors: their terms add in proportion to their share of the variance, plus
    a cross term ``p^2 r_a^2 r_b^2 / (r_a^2 + r_b^2) <= r^2 / 8``, which keeps the
    ``r^2`` coefficient below 2. So the relative bias is at most

        2 r (|g| + r),

    with ``r`` and ``g`` the largest over the factors involved, measured on the
    same seeds. ``g`` is a sample skewness with standard error ``sqrt(6 / K)``;
    three of those are added so that the allowance is not understated.
    """
    r = max(float(f.std(unbiased=True) / f.mean().abs()) for f in factors)
    g = max(float((((f - f.mean()) / f.std(unbiased=True)) ** 3).mean().abs()) for f in factors)
    return 2.0 * r * (g + 3.0 * math.sqrt(6.0 / n_seeds) + r)


# (estimator, micro_batch_size, n_hutchinson, check the propagated rows)
#
# The rows whose standard error is the per-probe spread itself are checked in
# every configuration, including two probes per micro-batch, where the estimate
# of the spread is as coarse as it gets and a wrong divisor is a factor of two.
# The propagated rows are a first-order statement, so they are checked where the
# relative error of chi_net is a few percent (more probes), with the allowance
# of ``_first_order_allowance``.
_STDERR_CONFIGS = [
    ("hutchinson", 1, 2, False),
    ("hutchinson", 4, 16, True),
    ("per_sequence_cv", 4, 2, False),
    ("per_sequence_cv", 1, 8, True),
]


@pytest.mark.slow
@pytest.mark.parametrize(("method", "micro", "n_h", "propagated"), _STDERR_CONFIGS)
def test_reported_stderr_matches_the_spread_over_seeds(
    method: str, micro: int, n_h: int, propagated: bool
) -> None:
    """Through ``analyze()``: mean ``value_probe_stderr**2`` over seeds = variance of ``value``.

    Two batches of 5 and 7 samples (so ``micro = 4`` cuts them 4 + 1 and 4 + 3)
    and their cross pair, 256 analyzer seeds, an AdamW ``P`` with a fifth of its
    coordinates at the ceiling. For each row type the difference between the
    sample variance of the values and the mean reported variance must be within
    ``Z_SIGMA`` standard errors of that difference (``_reported_minus_empirical``);
    the propagated rows additionally get the relative bias their first-order
    formula is known to have (``_first_order_allowance``).

    What this test does **not** pin: that the two batches of a cross pair are
    probed independently, which the cross-pair formula assumes. With 256 seeds
    the comparison resolves the variance to a few tens of percent, and a review
    showed that both batches drawing from one probe stream (the batch name
    dropped from the per-call seed) under-reports the cross-pair variance
    without failing here (to 0.71 of the observed one in the review; 0.66-0.79
    when re-measured over 1500 seeds on a copy with that defect). The seed derivation is
    pinned by ``tests/unit/test_ddp_probe_streams.py``; its consequence for a
    cross pair is pinned without statistics by
    ``test_two_batches_of_a_cross_pair_draw_different_probes`` below.
    """
    n_seeds = 256
    bundle, precond, _ = _mlp_bundle()
    values, stderrs = _collect(
        n_seeds,
        model=bundle,
        eval_batches={"a": _mlp_batch(5, 1), "b": _mlp_batch(7, 2)},
        cross_pairs=[("a", "b")],
        preconditioner=precond,
        chi_net_method=method,
        n_hutchinson=n_h,
        micro_batch_size=micro,
        dtype="fp32",
    )
    pairs = [("a", "a"), ("b", "b"), ("a", "b")]
    assert set(values) == {(a, b, obs) for a, b in pairs for obs in (*DIRECT, *PROPAGATED)}, (
        "every sampled row type must carry a standard error, and no other row"
    )

    # vacuity guard: the capped part is a real share of chi_net_P on both batches.
    for name in ("a", "b"):
        share = values[(name, name, "chi_net_P_capped")] / values[(name, name, "chi_net_P")]
        assert 0.2 < float(share.mean()) < 0.9, (name, float(share.mean()))

    checked = 0
    for a, b in pairs:
        for obs in (*DIRECT, *PROPAGATED):
            first_order = a != b or obs in PROPAGATED
            if first_order and not propagated:
                continue
            x, s = values[(a, b, obs)], stderrs[(a, b, obs)]
            diff, diff_stderr, v_rep = _reported_minus_empirical(x, s)
            tolerance = Z_SIGMA * diff_stderr
            allowance = 0.0
            if first_order:
                factor = PROPAGATED.get(obs, obs).removesuffix("_normalized")
                allowance = _first_order_allowance(
                    [values[(name, name, factor)] for name in {a, b}], n_seeds
                )
                tolerance += allowance * v_rep
                # The allowance must stay a correction: the sampling error of the
                # comparison, not the allowance, is what sets the tolerance.
                assert allowance * v_rep < 0.5 * Z_SIGMA * diff_stderr, (obs, allowance)
            v_emp = diff + v_rep
            assert abs(diff) <= tolerance, (
                f"{method} micro={micro} n_h={n_h} ({a}, {b}) {obs}: reported variance "
                f"{v_rep:.6e}, sample variance of the value over {n_seeds} seeds {v_emp:.6e} "
                f"(ratio {v_rep / v_emp:.3f}); difference {diff:+.3e} = "
                f"{diff / diff_stderr:+.2f} standard errors, first-order allowance "
                f"{allowance:.3f} of the reported variance"
            )
            # vacuity guard: a reported variance of half the true one would be
            # off by v_rep, so the tolerance has to be below that to see it.
            assert tolerance < v_rep, (obs, tolerance, v_rep)
            checked += 1
    assert checked == (18 if propagated else 8)


def _rank_partial_rows(
    monkeypatch: pytest.MonkeyPatch, world_size: int, **analyze_kwargs: Any
) -> list[dict[RowKey, ResultRow]]:
    """Per rank, the rows that rank computes on its own shard.

    The emulation of ``tests/unit/test_ddp_probe_streams.py``: the rank and world
    size the analyzer sees are patched while ``is_distributed()`` stays false, so
    there is no all-reduce and each rank's ``chi_net`` row -- and its standard
    error -- is that rank's partial, the number a real rank hands to the
    all-reduce. The real ``torchrun`` path is in ``tests/integration``.
    """
    per_rank = []
    for rank in range(world_size):
        with monkeypatch.context() as patch:
            patch.setattr(analyzer_mod, "get_rank", lambda rank=rank: rank)
            patch.setattr(analyzer_mod, "get_world_size", lambda: world_size)
            results = analyze(revisions=["step10"], sink=None, **analyze_kwargs)
        per_rank.append(_rows_by_key(results[0].rows))
    return per_rank


@pytest.mark.slow
@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_rank_variances_add_like_the_rank_estimates(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Two ranks: the summed per-rank variance is the variance of the summed estimate.

    The all-reduce sums the ranks' partial ``chi_net`` and, next to it, their
    partial variances. That is right only because the ranks draw independent
    probe streams; if they replayed one stream the sum of the partials would
    vary more than the sum of their variances says. Batch of 5 over 2 ranks
    (shards of 3 and 2), ``micro_batch_size = 2`` (so rank 0 has two micro-batches
    and rank 1 one), two probes, 256 seeds; same comparison and tolerance as
    the single-process test, on the three rows a rank holds a partial sum of.
    """
    n_seeds, world_size = 256, 2
    bundle, precond, _ = _mlp_bundle()
    batch = _mlp_batch(5, 3)
    totals: dict[str, list[float]] = {
        obs: [] for obs in ("chi_net", "chi_net_P", "chi_net_P_capped")
    }
    variances: dict[str, list[float]] = {obs: [] for obs in totals}
    for seed in range(n_seeds):
        per_rank = _rank_partial_rows(
            monkeypatch,
            world_size,
            model=bundle,
            eval_batches={"a": batch},
            preconditioner=precond,
            chi_net_method=method,
            n_hutchinson=2,
            micro_batch_size=2,
            seed=seed,
            dtype="fp32",
        )
        for obs in totals:
            rows = [rank_rows[("a", "a", obs)] for rank_rows in per_rank]
            assert all(r.value_probe_stderr is not None and r.n_ranks == world_size for r in rows)
            totals[obs].append(sum(r.value for r in rows))
            variances[obs].append(sum(r.value_probe_stderr**2 for r in rows))  # type: ignore[operator]
    for obs in totals:
        x = torch.tensor(totals[obs], dtype=FP64)
        s = torch.tensor(variances[obs], dtype=FP64).sqrt()
        diff, diff_stderr, v_rep = _reported_minus_empirical(x, s)
        assert abs(diff) <= Z_SIGMA * diff_stderr, (
            f"{method} {obs}: summed rank variances {v_rep:.6e}, sample variance of the "
            f"summed rank estimates {diff + v_rep:.6e} ({diff / diff_stderr:+.2f} standard errors)"
        )
        assert Z_SIGMA * diff_stderr < v_rep, (obs, diff_stderr, v_rep)


# ========================================================= 4. row arithmetic and nulls


def _one_run(**overrides: Any) -> dict[RowKey, ResultRow]:
    bundle, precond, _ = _mlp_bundle()
    kwargs: dict[str, Any] = {
        "model": bundle,
        "revisions": ["step10"],
        "eval_batches": {"a": _mlp_batch(5, 1), "b": _mlp_batch(7, 2)},
        "cross_pairs": [("a", "b")],
        "preconditioner": precond,
        "chi_net_method": "hutchinson",
        "n_hutchinson": 4,
        "micro_batch_size": 2,
        "sink": None,
        "seed": 7,
        "dtype": "fp32",
    }
    kwargs.update(overrides)
    return _rows_by_key(analyze(**kwargs)[0].rows)


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_the_rows_without_a_probe_carry_no_stderr(method: str) -> None:
    rows = _one_run(chi_net_method=method)
    for (a, b, obs), row in rows.items():
        if obs in NOT_SAMPLED:
            assert row.value_probe_stderr is None, (a, b, obs, row.value_probe_stderr)
        else:
            assert row.value_probe_stderr is not None and row.value_probe_stderr > 0.0, (a, b, obs)
    assert {obs for _a, _b, obs in rows} == {*NOT_SAMPLED, *DIRECT, *PROPAGATED}


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_a_single_probe_gives_no_stderr_on_any_row(method: str) -> None:
    rows = _one_run(chi_net_method=method, n_hutchinson=1)
    assert rows and all(row.value_probe_stderr is None for row in rows.values())


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_propagated_rows_are_the_stated_functions_of_the_sampled_ones(method: str) -> None:
    """The first-order formulas, as arithmetic on the rows of one run.

    - ``chi_net_normalized``: ``chi_net`` over ``sqrt(n_valid_a * n_valid_b)``;
    - ``chi_pos`` (and ``chi_pos_P``): ``|chi_pos| * stderr(chi_net) / chi_net``,
      self pair and cross pair alike;
    - cross-pair ``chi_net`` (and the P / capped ones): the geometric mean of two
      independently probed batches, ``sqrt(a b) / 2 * sqrt((s_a / a)^2 + (s_b / b)^2)``.
    """
    rows = _one_run(chi_net_method=method)

    def stderr(a: str, b: str, obs: str) -> float:
        value = rows[(a, b, obs)].value_probe_stderr
        assert value is not None
        return value

    for a, b in (("a", "a"), ("b", "b"), ("a", "b")):
        net = rows[(a, b, "chi_net")]
        norm = normalization_factor(net.n_valid_a, net.n_valid_b)
        assert stderr(a, b, "chi_net_normalized") == pytest.approx(
            stderr(a, b, "chi_net") / norm, rel=1e-12
        )
        for pos, factor in PROPAGATED.items():
            want = abs(rows[(a, b, pos)].value) * stderr(a, b, factor) / rows[(a, b, factor)].value
            assert stderr(a, b, pos) == pytest.approx(want, rel=1e-12), (a, b, pos)
    for obs in ("chi_net", "chi_net_P", "chi_net_P_capped"):
        va, vb = rows[("a", "a", obs)].value, rows[("b", "b", obs)].value
        assert rows[("a", "b", obs)].value == pytest.approx(math.sqrt(va * vb), rel=1e-12)
        want = (
            math.sqrt(va * vb)
            * 0.5
            * math.sqrt((stderr("a", "a", obs) / va) ** 2 + (stderr("b", "b", obs) / vb) ** 2)
        )
        assert stderr("a", "b", obs) == pytest.approx(want, rel=1e-12), obs


def test_two_batches_of_a_cross_pair_draw_different_probes() -> None:
    """The cross-pair standard error adds two batches' errors as independent.

    That is only right if the two batches are probed with different streams.
    Deterministic check: the *same* data under two batch names. Everything that
    involves no probe is then identical for the two names, and if they shared a
    probe stream the sampled values would be identical as well -- bit for bit --
    while the reported cross-pair error would still be that of two independent
    draws, too small by a factor ``sqrt(2)``. With separate streams the sampled
    values differ.
    """
    batch = _mlp_batch(6, 1)
    rows = _one_run(
        eval_batches={"first": batch, "second": batch}, cross_pairs=[("first", "second")]
    )
    first = {obs: row for (a, b, obs), row in rows.items() if (a, b) == ("first", "first")}
    second = {obs: row for (a, b, obs), row in rows.items() if (a, b) == ("second", "second")}
    assert set(first) == set(second) == {*NOT_SAMPLED, *DIRECT, *PROPAGATED}
    for obs in NOT_SAMPLED:
        assert first[obs].value == second[obs].value, obs
    assert first["chi_net"].hutchinson_seed != second["chi_net"].hutchinson_seed
    for obs in (*DIRECT, *PROPAGATED):
        relative = abs(first[obs].value - second[obs].value) / abs(first[obs].value)
        assert relative > 1e-4, (
            f"{obs}: {first[obs].value!r} for both names of one batch -- the two batches "
            "of a cross pair are drawing the same probes, so their errors are not "
            "independent and the cross-pair value_probe_stderr is too small"
        )
        assert first[obs].value_probe_stderr != second[obs].value_probe_stderr, obs


def test_a_capped_part_that_is_exactly_zero_has_stderr_zero() -> None:
    """A ceiling nobody sits at: the capped rows are zero for every probe draw.

    Zero is then the exact spread, on the self pair and -- although a geometric
    mean cannot be differentiated at zero -- on the cross pair too, because the
    product is zero whatever the other batch's probes do.
    """
    bundle, _, _ = _mlp_bundle()
    n_params = sum(p.numel() for p in bundle.params)
    gen = torch.Generator().manual_seed(1)
    diag = torch.rand(n_params, generator=gen, dtype=FP64) + 0.5
    rows = _one_run(preconditioner=Preconditioner.from_diagonal(diag, ceiling=100.0))
    for pair in (("a", "a"), ("b", "b"), ("a", "b")):
        row = rows[(*pair, "chi_net_P_capped")]
        assert row.value == 0.0 and row.value_probe_stderr == 0.0, (pair, row)
        assert rows[(*pair, "chi_net_P")].value_probe_stderr > 0.0  # type: ignore[operator]


def test_geometric_mean_stderr_edge_cases() -> None:
    f = analyzer_mod._geometric_mean_stderr
    assert f(4.0, 0.4, 9.0, 0.9) == pytest.approx(6.0 * 0.5 * math.hypot(0.1, 0.1))
    # No standard error on either side -> none for the pair.
    assert f(4.0, None, 9.0, 0.9) is None
    assert f(4.0, 0.4, 9.0, None) is None
    # An exactly-zero factor makes the product exactly zero.
    assert f(0.0, 0.0, 9.0, 0.9) == 0.0
    assert f(4.0, 0.4, 0.0, 0.0) == 0.0
    # A zero factor that claims a spread is not something the estimators produce:
    # no number is made up for it.
    assert f(0.0, 0.1, 9.0, 0.9) is None


def test_ratio_stderr_edge_cases() -> None:
    f = analyzer_mod._ratio_stderr
    assert f(-0.5, 10.0, 1.0) == pytest.approx(0.05)
    assert f(0.5, 10.0, None) is None
    # chi_net == 0 is the degenerate chi_pos = 0 convention, not a ratio.
    assert f(0.0, 0.0, 0.0) is None
