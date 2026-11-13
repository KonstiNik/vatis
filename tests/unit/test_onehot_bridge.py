"""The code-vs-theory bridge: vatis against closed forms on the one-hot toy.

Everything the LNP decomposition names has a closed form on this fixture (see
`tests/fixtures/onehot_toy.py` for the derivation), so a disagreement here means
vatis and the theory are naming **different objects** — not that an estimator is
noisy. That distinction is what the rest of the suite cannot make: the exact-NTK
tests in ``test_chi_net.py`` and the brute-force tests in ``test_chi_pos_p.py``
compare vatis against a *materialised* ``Theta_P``, which shares vatis's
definition of the kernel by construction. Here the reference is an independent
algebraic identity.

Two reference preconditioners, both from ``00_context.md`` §"P conventions":

    P = (J^T J)^+                    ->  chi_pos^P = 1/M       (any residual)
    P = 1/(|g| + eps), eps = 0       ->  chi_pos^P = (sum pi s) / ((sum pi s^2)(sum 1/s))

The closed forms are themselves re-derived against ``eigh(Theta_P)`` first, so
they are not taken on faith either.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.fixtures.onehot_toy import (
    build_onehot_case,
    lnp_from_explicit_kernel,
    onehot_all_valid,
    onehot_forward,
)
from vatis.analyzer import _add_grads_into_flat, _sum_sq_fp64, _zero_param_vector
from vatis.core.chi_net.hutchinson import HutchinsonEstimator
from vatis.core.chi_net.per_seq_cv import PerSequenceControlVariateEstimator
from vatis.core.preconditioner import Preconditioner, weighted_sum_sq_fp64

FP64 = torch.float64
SEED = 20260729

# Uniform multiplicities -> Theta_P diagonal -> Rademacher variance is exactly 0,
# so plain `hutchinson` is exact too. Non-uniform -> block-diagonal Theta_P, only
# `per_sequence_cv` stays exact (m = 1 makes its projector the identity).
UNIFORM = {"multiplicities": [1] * 6, "residuals": [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]}
NONUNIFORM = {"multiplicities": [5, 3, 2, 1, 1], "residuals": [3.0, 1.5, 0.75, 0.2, 0.05]}

# The one-hot model is fp64, but the analyzer assembles delta_loss from an fp32
# flat gradient vector (`_zero_param_vector`), so ~1e-7 relative is the floor for
# any delta_loss-derived quantity no matter how exact the model is.
FP32_GRAD_REL = 1e-6


def _rademacher_trace_std(a: torch.Tensor, n_probes: int) -> float:
    """``sqrt(2 sum_{i != j} A_ij^2 / n)`` — zero for diagonal ``A``."""
    off = a - torch.diag(torch.diagonal(a))
    return (2.0 * float((off * off).sum()) / float(n_probes)) ** 0.5


def _run(
    case: dict,
    p_diag: torch.Tensor | None,
    *,
    method: str,
    n_h: int,
    seed: int,
    micro: int,
    store_alignment: bool = False,
) -> dict[str, float]:
    """Shipped estimator for chi_net, the analyzer's own arithmetic for delta_loss."""
    precond = (
        None if p_diag is None else Preconditioner.from_diagonal(p_diag, descriptor="explicit")
    )
    estimator: HutchinsonEstimator | PerSequenceControlVariateEstimator = (
        HutchinsonEstimator(n_hutchinson=n_h)
        if method == "hutchinson"
        else PerSequenceControlVariateEstimator(
            n_hutchinson=n_h, compute_alignment_matrix=store_alignment
        )
    )
    gen = torch.Generator().manual_seed(seed)
    result = estimator.compute(
        case["model"],
        case["batch"],
        onehot_forward,
        loss_fn=case["loss_fn"],
        valid_mask_fn=onehot_all_valid,
        params=case["params"],
        micro_batch_size=micro,
        generator=gen,
        preconditioner=precond,
    )

    flat = _zero_param_vector(case["params"], torch.device("cpu"))
    loss = case["loss_fn"](onehot_forward(case["model"], case["batch"]), case["batch"])
    grads = torch.autograd.grad(loss, case["params"], allow_unused=True)
    _add_grads_into_flat(flat, grads, case["params"])
    delta = float(_sum_sq_fp64(flat))

    out = {
        "chi_net": float(result.chi_net),
        "delta_loss": delta,
        "n_valid": float(result.n_valid_tokens),
        "chi_pos": delta / (case["chi_loss"] * float(result.chi_net)),
    }
    if result.chi_net_p is not None and precond is not None:
        delta_p = float(weighted_sum_sq_fp64(flat, precond.diag))
        out["chi_net_P"] = float(result.chi_net_p)
        out["delta_loss_P"] = delta_p
        out["chi_pos_P"] = delta_p / (case["chi_loss"] * float(result.chi_net_p))
    return out


# ------------------------------------------- the closed forms, verified the long way


@pytest.mark.parametrize("profile,label", [(UNIFORM, "uniform"), (NONUNIFORM, "nonuniform")])
def test_closed_forms_agree_with_an_explicit_eigendecomposition(profile: dict, label: str) -> None:
    """Guard the reference itself: closed form == ``sum_k gamma_k^2 p_k`` from ``eigh``."""
    case = build_onehot_case(seed=SEED, **profile)
    for p_diag, target, name in (
        (None, case["closed_form_raw"], "raw"),
        (case["p_newton"], case["closed_form_newton"], "newton"),
        (case["p_adam"], case["closed_form_adam"], "adam"),
    ):
        ref = lnp_from_explicit_kernel(case["jac"], case["u"], p_diag)
        assert abs(ref["gamma_sq_sum"] - 1.0) < 1e-12, (label, name)
        assert abs(ref["chi_pos_spectral"] - ref["chi_pos_ratio"]) < 1e-13, (label, name)
        rel = abs(ref["chi_pos_spectral"] - target) / abs(target)
        assert rel < 1e-13, (label, name, ref["chi_pos_spectral"], target, rel)


# ------------------------------------------------------------------- the bridge


@pytest.mark.parametrize("micro", [1, 3, 12])
@pytest.mark.parametrize("profile,label", [(UNIFORM, "uniform"), (NONUNIFORM, "nonuniform")])
def test_newton_preconditioner_gives_one_over_m_through_the_shipped_code(
    micro: int, profile: dict, label: str
) -> None:
    """``P = (J^T J)^+`` ⇒ ``chi_pos^P = 1/M`` and ``Tr(Theta_P) = M``."""
    case = build_onehot_case(seed=SEED, **profile)
    got = _run(case, case["p_newton"], method="per_sequence_cv", n_h=4, seed=7, micro=micro)
    target = case["closed_form_newton"]
    assert got["chi_pos_P"] == pytest.approx(target, rel=FP32_GRAD_REL), (label, micro, got)
    assert got["chi_net_P"] == pytest.approx(float(case["m_modes"]), rel=1e-5), (label, got)


@pytest.mark.parametrize("micro", [1, 3, 12])
@pytest.mark.parametrize("profile,label", [(UNIFORM, "uniform"), (NONUNIFORM, "nonuniform")])
def test_full_batch_adam_matches_its_closed_form_through_the_shipped_code(
    micro: int, profile: dict, label: str
) -> None:
    """``chi_pos^P = (sum pi s)/((sum pi s^2)(sum 1/s))``, and raw matches its own form."""
    case = build_onehot_case(seed=SEED, **profile)
    got = _run(case, case["p_adam"], method="per_sequence_cv", n_h=4, seed=11, micro=micro)
    assert got["chi_pos_P"] == pytest.approx(case["closed_form_adam"], rel=FP32_GRAD_REL), (
        label,
        micro,
        got["chi_pos_P"],
        case["closed_form_adam"],
    )
    assert got["chi_pos"] == pytest.approx(case["closed_form_raw"], rel=FP32_GRAD_REL), (
        label,
        micro,
        got["chi_pos"],
        case["closed_form_raw"],
    )
    assert got["chi_net"] == pytest.approx(case["chi_net"], rel=1e-6)


def test_hutchinson_is_exact_when_the_kernel_is_diagonal() -> None:
    """Uniform multiplicities ⇒ ``Theta_P`` diagonal ⇒ zero Rademacher variance."""
    case = build_onehot_case(seed=SEED, **UNIFORM)
    theta_p = (case["jac"] * case["p_adam"].unsqueeze(0)) @ case["jac"].T
    assert _rademacher_trace_std(theta_p, 1) == 0.0, "precondition: diagonal kernel"
    got = _run(case, case["p_adam"], method="hutchinson", n_h=3, seed=5, micro=1)
    assert got["chi_pos_P"] == pytest.approx(case["closed_form_adam"], rel=FP32_GRAD_REL)


@pytest.mark.slow
def test_hutchinson_matches_the_closed_form_within_the_derived_bound() -> None:
    """Non-uniform multiplicities: block-diagonal ``Theta_P``, so statistical."""
    case = build_onehot_case(seed=SEED, **NONUNIFORM)
    theta_p = (case["jac"] * case["p_adam"].unsqueeze(0)) @ case["jac"].T
    n_h = 4096
    std = _rademacher_trace_std(theta_p, n_h)
    assert std > 0.0, "this branch is meant to be the stochastic one"
    trace = float(torch.diagonal(theta_p).sum())
    got = _run(case, case["p_adam"], method="hutchinson", n_h=n_h, seed=3, micro=case["b"])
    assert abs(got["chi_net_P"] - trace) <= 5.0 * std, (got["chi_net_P"], trace, std)
    target = case["closed_form_adam"]
    tol = target * (5.0 * std / trace) + 1e-9
    assert abs(got["chi_pos_P"] - target) <= tol, (got["chi_pos_P"], target, tol, std)


def test_alignment_matrix_memory_path_is_on_the_same_ruler() -> None:
    """The opt-in ``O(B*P)`` path must reach the same closed form."""
    case = build_onehot_case(seed=SEED, **NONUNIFORM)
    a = _run(case, case["p_adam"], method="per_sequence_cv", n_h=4, seed=11, micro=case["b"])
    b = _run(
        case,
        case["p_adam"],
        method="per_sequence_cv",
        n_h=4,
        seed=11,
        micro=case["b"],
        store_alignment=True,
    )
    for key in ("chi_net", "chi_net_P", "chi_pos_P"):
        assert b[key] == pytest.approx(a[key], rel=1e-5), (key, a[key], b[key])
    assert b["chi_pos_P"] == pytest.approx(case["closed_form_adam"], rel=1e-5)


def test_one_over_m_is_a_ceiling_attained_only_at_equal_residuals() -> None:
    """``chi_pos^P <= 1/M`` (Cauchy-Schwarz twice), with equality iff all ``s_k`` equal."""
    equal = build_onehot_case(multiplicities=[3, 2, 1, 1], residuals=[1.5, 1.5, 1.5, 1.5], seed=1)
    got = _run(equal, equal["p_adam"], method="per_sequence_cv", n_h=2, seed=2, micro=2)
    assert got["chi_pos_P"] == pytest.approx(1.0 / equal["m_modes"], rel=FP32_GRAD_REL)

    spread = build_onehot_case(
        multiplicities=[3, 2, 1, 1], residuals=[10.0, 1.0, 0.1, 0.01], seed=1
    )
    got2 = _run(spread, spread["p_adam"], method="per_sequence_cv", n_h=2, seed=2, micro=2)
    assert got2["chi_pos_P"] < 1.0 / spread["m_modes"]
    assert math.isfinite(got2["chi_pos_P"])
