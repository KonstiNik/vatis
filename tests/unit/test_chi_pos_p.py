"""Exact-answer tests for the preconditioned ruler chi_pos^P (WP0 component 03).

These are the acceptance criteria of ``03_estimator.md`` §Tests. Every one of
them compares vatis's *shipping* estimator path against a value known in closed
form, on an fp64 toy, with a fixed seed:

| test                              | exact answer                              |
| --------------------------------- | ----------------------------------------- |
| T-identity                        | ``P = I`` reproduces the raw columns       |
| T-GN                              | ``P = (J^T J)^+`` gives ``chi_pos^P = 1/M``|
| T-bruteforce                      | explicit ``Theta_P = J P J^T`` by eigendecomposition |
| T-flattening-direction            | ``P = diag(pi)^-1`` gives ``1/M``; raw is lower on a tail residual |
| T-batchsize                       | ``E[components]`` independent of ``|A|``   |

Two facts about *tolerances* apply throughout and are stated once here:

1. **Where exactness comes from, and where it cannot.** The estimators
   accumulate in fp64 but return fp32 scalars (``ChiNetResult.chi_net`` /
   ``chi_net_p``), and :func:`vatis.core.observables.chi_pos` casts its three
   inputs to fp32 and divides in fp32. That is the repo's pre-existing
   convention and T-identity *requires* it (a fp64 ``chi_net_p`` would differ
   from the fp32 ``chi_net`` by ~1e-8 relative and could never match at 1e-10).
   So an assertion on a public fp32 scalar cannot be tighter than a few
   multiples of ``2^-24`` relative — :func:`fp32_output_slack`. Where the spec
   asks for 1e-8, the tests therefore assert *both*: the fp32 public value
   against ``max(1e-8, fp32 slack)``, and the same quantity recomputed from the
   fp64 components against 1e-8 (which it beats by ~8 orders of magnitude). The
   achieved numbers are in the assertion messages.

2. **Where the stochastic tolerance comes from.** ``chi_net`` (and hence
   ``chi_net^P``) is a Hutchinson estimate. Its ``n``-probe standard deviation is
   derived in :func:`rademacher_trace_std`:
   ``std = sqrt(2 * sum_{i != j} A_ij^2 / n)`` with ``A = Theta_P`` for
   ``hutchinson`` and ``A = (I - Pi) Theta_P (I - Pi)`` for ``per_sequence_cv``.
   Note that this **vanishes for diagonal ``A``** — which is exactly the T-GN and
   T-flattening situation (``Theta_P = I``), so those two are equalities, not
   statistical statements. For the genuinely stochastic brute-force case the
   tests use ``k = 6`` standard deviations, computed from the materialized
   ``Theta_P`` inside the test.
"""

from __future__ import annotations

import statistics

import pytest
import torch

from tests.fixtures.preconditioned_toys import (
    Fp64MLP,
    LinearJacobianModel,
    all_valid_mask,
    ce_loss,
    ce_loss_direction,
    diagonal_spectrum_jacobian,
    fp32_output_slack,
    gauss_newton_jacobian,
    index_forward,
    loss_direction_projector,
    make_gaussian_class_batch,
    make_linear_loss,
    materialize_jacobian,
    mse_loss,
    preconditioned_reference,
    rademacher_trace_std,
    tuple_forward,
)
from vatis.core.chi_net import HutchinsonEstimator, PerSequenceControlVariateEstimator
from vatis.core.observables import chi_loss_cross_entropy, chi_pos
from vatis.core.preconditioner import (
    Preconditioner,
    weighted_dot_fp64,
    weighted_sum_sq_fp64,
)

# Number of standard deviations used for the stochastic brute-force bounds.
# With a fixed seed the tests are deterministic; k = 6 states how much of the
# sampling distribution the bound covers, so a future reseeding stays green
# without the bound being vacuous.
K_SIGMA = 6.0


def _flat_grad(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    """``grad_theta loss`` as one fp64 vector, in ``params`` order.

    Mirrors what the analyzer builds (``_zero_param_vector`` /
    ``_add_grads_into_flat``) except that it stays fp64, so these tests measure
    the preconditioned *math* rather than the analyzer's fp32 gradient buffer.
    """
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    parts = [
        torch.zeros(p.numel(), dtype=torch.float64)
        if g is None
        else g.detach().to(dtype=torch.float64).reshape(-1)
        for p, g in zip(params, grads, strict=True)
    ]
    return torch.cat(parts)


# ====================================================================== T-identity


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_identity_preconditioner_reproduces_raw_columns(method: str) -> None:
    """T-identity, core level: ``P = I`` -> the P triple equals the raw triple.

    Required atol on an fp64 toy MLP: **1e-10**. The identity ``P`` is
    materialized as a vector of ones (not short-circuited) precisely so that the
    preconditioned code path — the per-parameter ``P`` views in the estimator,
    the weighted reductions, the combinator — is what produces the numbers.
    Because multiplying by ``1.0`` is exact in IEEE-754 and the weighted
    reductions keep the raw accumulation order, the agreement should be
    *bit-for-bit*, i.e. far better than 1e-10; the assertion messages report
    what was achieved.
    """
    model = Fp64MLP(in_dim=6, hidden=8, n_out=4, seed=0)
    params = list(model.parameters())
    x, y = make_gaussian_class_batch(
        n=8, in_dim=6, teacher=torch.eye(6, 4, dtype=torch.float64), seed=1
    )
    batch = (x, y)
    identity = Preconditioner.identity(
        sum(p.numel() for p in params), dtype=torch.float64, descriptor="identity"
    )

    flat = _flat_grad(ce_loss(model(x), batch), params)
    delta_loss = float((flat * flat).sum())
    delta_loss_p = float(weighted_sum_sq_fp64(flat, identity.diag))

    estimator = (
        HutchinsonEstimator(n_hutchinson=16)
        if method == "hutchinson"
        else PerSequenceControlVariateEstimator(n_hutchinson=16)
    )
    gen = torch.Generator().manual_seed(4242)
    result = estimator.compute(
        model,
        batch,
        tuple_forward,
        loss_fn=ce_loss,
        valid_mask_fn=all_valid_mask,
        params=params,
        micro_batch_size=8,
        generator=gen,
        preconditioner=identity,
    )
    assert result.chi_net_p is not None
    chi_net = float(result.chi_net)
    chi_net_p = float(result.chi_net_p)
    chi_loss = float(chi_loss_cross_entropy(model(x).detach(), y))
    pos = float(chi_pos(delta_loss, chi_loss, chi_net))
    pos_p = float(chi_pos(delta_loss_p, chi_loss, chi_net_p))

    assert abs(delta_loss_p - delta_loss) <= 1e-10, (
        f"delta_loss_P != delta_loss at P=I: {delta_loss_p!r} vs {delta_loss!r} "
        f"(abs {abs(delta_loss_p - delta_loss):.3e})"
    )
    assert abs(chi_net_p - chi_net) <= 1e-10, (
        f"chi_net_P != chi_net at P=I: {chi_net_p!r} vs {chi_net!r} "
        f"(abs {abs(chi_net_p - chi_net):.3e})"
    )
    assert abs(pos_p - pos) <= 1e-10, (
        f"chi_pos_P != chi_pos at P=I: {pos_p!r} vs {pos!r} (abs {abs(pos_p - pos):.3e})"
    )
    # Sanity: the toy is non-degenerate, so this is not 0 == 0.
    assert chi_net > 0.0 and delta_loss > 0.0 and abs(pos) > 1e-6


def test_preconditioner_does_not_perturb_raw_chi_net() -> None:
    """Passing a (non-identity) P must leave ``chi_net`` bit-identical.

    The no-regression guard for "no change to raw-ruler numerics": the P
    accumulators are additive, so for a fixed seed the raw value must not move
    at all when the preconditioned ruler is switched on.
    """
    model = Fp64MLP(in_dim=6, hidden=8, n_out=4, seed=3)
    params = list(model.parameters())
    x, y = make_gaussian_class_batch(
        n=6, in_dim=6, teacher=torch.eye(6, 4, dtype=torch.float64), seed=2
    )
    batch = (x, y)
    gp = torch.Generator().manual_seed(17)
    n_params = sum(p.numel() for p in params)
    precond = Preconditioner.from_diagonal(
        torch.rand(n_params, generator=gp, dtype=torch.float64) * 4.0 + 0.01
    )

    def run(p: Preconditioner | None) -> float:
        gen = torch.Generator().manual_seed(99)
        res = PerSequenceControlVariateEstimator(n_hutchinson=8).compute(
            model,
            batch,
            tuple_forward,
            loss_fn=ce_loss,
            valid_mask_fn=all_valid_mask,
            params=params,
            micro_batch_size=3,
            generator=gen,
            preconditioner=p,
        )
        return float(res.chi_net)

    without = run(None)
    with_p = run(precond)
    assert with_p == without, f"raw chi_net moved when P was enabled: {with_p!r} vs {without!r}"


# =========================================================================== T-GN


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gn_preconditioner_gives_one_over_m(seed: int) -> None:
    """T-GN: ``P = (J^T J)^+`` on a full-row-rank random linear model -> ``1/M``.

    ``N = 64 > M = m|A| = 4 * 10 = 40``, ``rank(J) = M``. See
    :func:`gauss_newton_jacobian` for the construction and for why it keeps
    ``(J^T J)^+`` diagonal (vatis's ``P`` is diagonal by design; non-diagonal
    ``P`` is out of scope for WP0). ``Theta_P = J (J^T J)^+ J^T = I_M``, so

        chi_pos^P = u^T I u / (||u||^2 * Tr I) = 1/M   for ANY loss gradient u.

    The test runs ``seed``-many different random loss gradients (the
    parametrization) and asserts 1/M at atol 1e-8.

    This is an *equality*, not a statistical statement, even though ``chi_net^P``
    goes through Hutchinson: for ``Theta_P = I`` the Rademacher estimate is
    ``z^T I z = sum_i z_i^2 = M`` for **every** probe, and
    :func:`rademacher_trace_std` returns exactly 0 (no off-diagonal terms). The
    test asserts that zero-variance property directly, so a regression that made
    the P weighting probe-dependent would fail here rather than hide behind a
    tolerance.
    """
    n_samples, m, n_params = 10, 4, 64
    jac, live, d = gauss_newton_jacobian(n_samples=n_samples, m=m, n_params=n_params, seed=11)
    big_m = n_samples * m

    # P = (J^T J)^+, both analytically and via torch.linalg.pinv — the two must
    # agree, and pinv must come out diagonal (that is the construction's claim).
    gram = jac.T @ jac
    pinv = torch.linalg.pinv(gram, hermitian=True)
    p_diag = torch.zeros(n_params, dtype=torch.float64)
    p_diag[live] = 1.0 / d**2
    off = pinv - torch.diag(torch.diagonal(pinv))
    assert float(off.abs().max()) < 1e-10, (
        f"(J^T J)^+ is not diagonal for this construction (max off-diagonal "
        f"{float(off.abs().max()):.3e}); the GN P would be outside vatis's scope"
    )
    assert float((torch.diagonal(pinv) - p_diag).abs().max()) < 1e-10

    model = LinearJacobianModel(jac.reshape(n_samples, m, n_params))
    params = [model.theta]
    index = torch.arange(n_samples)
    precond = Preconditioner.from_diagonal(p_diag, descriptor="gn")

    gw = torch.Generator().manual_seed(500 + seed)
    w = torch.randn(n_samples, m, generator=gw, dtype=torch.float64)
    loss_fn = make_linear_loss(w)
    reference = preconditioned_reference(jac, w.reshape(-1), p_diag)
    assert abs(reference["chi_net_p"] - big_m) < 1e-9, "Theta_P should be I_M, Tr = M"

    chi_loss = float((w * w).sum())
    flat = _flat_grad(loss_fn(model(index), index), params)
    delta_loss_p = float(weighted_sum_sq_fp64(flat, p_diag))

    gen = torch.Generator().manual_seed(7 + seed)
    result = HutchinsonEstimator(n_hutchinson=8).compute(
        model,
        index,
        index_forward,
        loss_fn=loss_fn,
        valid_mask_fn=all_valid_mask,
        params=params,
        micro_batch_size=n_samples,
        generator=gen,
        preconditioner=precond,
    )
    assert result.chi_net_p is not None
    chi_net_p = float(result.chi_net_p)
    theta_p = (jac * p_diag.unsqueeze(0)) @ jac.T
    probe_std = rademacher_trace_std(theta_p, 8)
    assert probe_std < 1e-12 * big_m, (
        f"Theta_P = I must make the Hutchinson estimate variance-free; the derived "
        f"std is {probe_std:.3e}, which is more than the round-off level of the "
        "materialized Theta_P, so this test's exactness claim is unfounded"
    )
    trace_slack = fp32_output_slack(float(big_m))
    assert abs(chi_net_p - big_m) <= trace_slack, (
        f"Tr(Theta_P) must be M={big_m} (variance-free probe), got {chi_net_p!r} "
        f"(abs {abs(chi_net_p - big_m):.3e}, fp32 slack {trace_slack:.3e})"
    )

    pos_p_fp32 = float(chi_pos(delta_loss_p, chi_loss, chi_net_p))
    pos_p_fp64 = delta_loss_p / (chi_loss * chi_net_p)
    target = 1.0 / big_m
    slack = max(1e-8, fp32_output_slack(target))
    assert abs(pos_p_fp32 - target) <= slack, (
        f"chi_pos_P != 1/M: {pos_p_fp32!r} vs {target!r} "
        f"(abs {abs(pos_p_fp32 - target):.3e}, allowed {slack:.3e})"
    )
    assert abs(pos_p_fp64 - target) <= 1e-8, (
        f"chi_pos_P (fp64 components) != 1/M: {pos_p_fp64!r} vs {target!r} "
        f"(abs {abs(pos_p_fp64 - target):.3e})"
    )
    # The raw ruler on the same model is NOT 1/M — otherwise the test would pass
    # for a P that was silently ignored.
    raw_pos = reference["delta_loss"] / (chi_loss * reference["chi_net"])
    assert abs(raw_pos - target) > 1e-4, (
        f"raw chi_pos happens to equal 1/M ({raw_pos!r}); the test cannot "
        "distinguish a working P from an ignored one"
    )


def test_gn_dense_reference_also_gives_one_over_m() -> None:
    """The ``chi_pos^P = 1/M`` claim, verified for a fully dense random ``J``.

    Pure-math companion to the test above: for *any* ``M x N`` ``J`` of full row
    rank with ``N > M``, ``J (J^T J)^+ J^T = I_M`` and hence
    ``chi_pos^P = 1/M``. The Gauss-Newton ``P`` is then dense, which is outside
    the diagonal-``P`` scope of WP0 — so this asserts the mathematical statement
    (from the materialized Jacobian), while the previous test asserts that
    vatis's shipping estimator reproduces it. Together they cover both halves.
    """
    g = torch.Generator().manual_seed(31)
    big_m, n_params = 40, 64
    jac = torch.randn(big_m, n_params, generator=g, dtype=torch.float64)
    assert int(torch.linalg.matrix_rank(jac).item()) == big_m
    p_dense = torch.linalg.pinv(jac.T @ jac, hermitian=True)
    theta_p = jac @ p_dense @ jac.T
    eye = torch.eye(big_m, dtype=torch.float64)
    assert float((theta_p - eye).abs().max()) < 1e-8
    for seed in range(4):
        gu = torch.Generator().manual_seed(900 + seed)
        u = torch.randn(big_m, generator=gu, dtype=torch.float64)
        chi_loss = float((u * u).sum())
        delta_loss_p = float(u @ theta_p @ u)
        pos_p = delta_loss_p / (chi_loss * float(torch.diagonal(theta_p).sum()))
        assert abs(pos_p - 1.0 / big_m) <= 1e-8, (
            f"dense-P reference: chi_pos_P {pos_p!r} != 1/M {1.0 / big_m!r}"
        )


# =================================================================== T-bruteforce


def test_bruteforce_exact_agreement_zero_variance_cv() -> None:
    """T-bruteforce, exact branch: estimator == explicit ``Theta_P`` at atol 1e-8.

    Tiny fp64 MLP (183 parameters, well under the spec's 1e3) with **one output
    per sample** and ``|A| = 20``, so ``M = m|A| = 20 <= 40``. The Jacobian is
    materialized by autograd, ``Theta_P = J P J^T`` formed for a random diagonal
    ``P``, and ``delta_loss^P`` / ``Tr(Theta_P)`` / ``chi_pos^P`` obtained by
    explicit eigendecomposition (:func:`preconditioned_reference`).

    Why this branch is *exact* rather than statistical, with no exact-trace mode
    added to production code: with ``m = 1`` each per-sample output block is
    one-dimensional, so ``per_sequence_cv``'s control-variate projector ``Pi``
    is the identity in output space, ``w = (I - Pi) v = 0`` for every probe, the
    Hutchinson residual is identically zero, and the estimator returns its
    *exact* term ``sum_b ||P^{1/2} g_b||^2 / ||u_b||^2 = Tr(J P J^T)``. The test
    checks the zero-variance precondition (``rademacher_trace_std == 0``) rather
    than assuming it.

    Achieved numbers are in the assertion messages. The only residual is the
    fp32 cast of the public scalars (see the module docstring), so each
    assertion comes in an fp32 form and an fp64-components form.
    """
    model = Fp64MLP(in_dim=6, hidden=8, n_out=1, seed=5)
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    assert n_params <= 1000, f"toy must stay under 1e3 parameters, got {n_params}"

    gb = torch.Generator().manual_seed(1)
    n_samples = 20
    x = torch.randn(n_samples, 6, generator=gb, dtype=torch.float64)
    y = torch.randn(n_samples, 1, generator=gb, dtype=torch.float64)
    batch = (x, y)

    jac = materialize_jacobian(model, batch, params)
    assert tuple(jac.shape) == (n_samples, n_params)
    logits = model(x).detach()
    u = (2.0 / n_samples) * (logits.reshape(-1) - y.reshape(-1))

    gp = torch.Generator().manual_seed(9)
    p_diag = torch.rand(n_params, generator=gp, dtype=torch.float64) * 3.0 + 0.05
    precond = Preconditioner.from_diagonal(p_diag)
    reference = preconditioned_reference(jac, u, p_diag)

    # The LNP factorization itself: spectral form == ratio form.
    assert abs(reference["chi_pos_p_spectral"] - reference["chi_pos_p_ratio"]) <= 1e-12

    chi_loss = reference["chi_loss"]
    flat = _flat_grad(mse_loss(model(x), batch), params)
    delta_loss_p = float(weighted_sum_sq_fp64(flat, p_diag))

    theta_p = (jac * p_diag.unsqueeze(0)) @ jac.T
    pi = loss_direction_projector(u, n_samples, 1)
    eye = torch.eye(n_samples, dtype=torch.float64)
    residual = (eye - pi) @ theta_p @ (eye - pi)
    assert rademacher_trace_std(residual, 1) == 0.0, (
        "with m = 1 the control-variate residual must be identically zero; the "
        "derived std was non-zero, so the exactness claim is unfounded"
    )

    gen = torch.Generator().manual_seed(0)
    result = PerSequenceControlVariateEstimator(n_hutchinson=1).compute(
        model,
        batch,
        tuple_forward,
        loss_fn=mse_loss,
        valid_mask_fn=all_valid_mask,
        params=params,
        micro_batch_size=n_samples,
        generator=gen,
        preconditioner=precond,
    )
    assert result.chi_net_p is not None
    chi_net_p = float(result.chi_net_p)

    assert abs(delta_loss_p - reference["delta_loss_p"]) <= 1e-8, (
        f"delta_loss_P != u^T Theta_P u: {delta_loss_p!r} vs "
        f"{reference['delta_loss_p']!r} (abs {abs(delta_loss_p - reference['delta_loss_p']):.3e})"
    )
    trace_slack = fp32_output_slack(reference["chi_net_p"])
    assert abs(chi_net_p - reference["chi_net_p"]) <= trace_slack, (
        f"chi_net_P != Tr(Theta_P): {chi_net_p!r} vs {reference['chi_net_p']!r} "
        f"(abs {abs(chi_net_p - reference['chi_net_p']):.3e}, fp32 slack {trace_slack:.3e})"
    )
    exact_pos = reference["chi_pos_p_spectral"]
    pos_fp32 = float(chi_pos(delta_loss_p, chi_loss, chi_net_p))
    pos_fp64 = delta_loss_p / (chi_loss * reference["chi_net_p"])
    pos_slack = max(1e-8, fp32_output_slack(exact_pos))
    assert abs(pos_fp32 - exact_pos) <= pos_slack, (
        f"chi_pos_P != eigendecomposition value: {pos_fp32!r} vs {exact_pos!r} "
        f"(abs {abs(pos_fp32 - exact_pos):.3e}, allowed {pos_slack:.3e})"
    )
    assert abs(pos_fp64 - exact_pos) <= 1e-8, (
        f"chi_pos_P (fp64 components) != eigendecomposition value: {pos_fp64!r} vs "
        f"{exact_pos!r} (abs {abs(pos_fp64 - exact_pos):.3e})"
    )


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_bruteforce_matches_explicit_jacobian_stochastic(method: str) -> None:
    """T-bruteforce, stochastic branch: multi-output MLP, derived tolerance.

    Tiny fp64 MLP (164 parameters) with ``m = 4`` outputs and ``|A| = 10``, so
    ``M = m|A| = 40`` — the spec's ceiling. Cross-entropy, so ``u`` is the real
    closed-form loss gradient. ``Theta_P = J P J^T`` is materialized and
    eigendecomposed; the estimator must match.

    Here the production path *is* stochastic (``m = 4`` leaves a rank-3
    per-sample subspace the control variate does not cover, and plain Hutchinson
    covers nothing exactly), and no exact-trace mode is added to production for
    the test's benefit. The tolerance is derived instead, in full:

        one probe:  z^T A z with E = Tr(A), Var = 2 * sum_{i != j} A_ij^2
        n probes:   std = sqrt(Var / n)                       [rademacher_trace_std]
        A         = Theta_P                        for hutchinson
        A         = (I - Pi) Theta_P (I - Pi)      for per_sequence_cv, whose
                    exact term sum_b ||P^{1/2} g_b||^2 / ||u_b||^2 is deterministic

        |chi_net_P_hat - Tr(Theta_P)| <= K_SIGMA * std                       (i)

    and, since ``delta_loss^P`` and ``chi_loss`` are deterministic, ``chi_pos^P``
    inherits that interval exactly (it is ``c / chi_net_P_hat`` with ``c`` fixed):

        chi_pos^P_hat in [ c / (Tr + K*std),  c / (Tr - K*std) ]              (ii)

    with ``K_SIGMA = 6``. ``n_hutchinson = 4096`` puts the resulting relative
    band at a few percent; the achieved deviations (reported in the assertion
    messages, both in absolute terms and as a multiple of ``std``) are ~1-2
    sigma.
    """
    model = Fp64MLP(in_dim=6, hidden=8, n_out=4, seed=0)
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    assert n_params <= 1000, f"toy must stay under 1e3 parameters, got {n_params}"

    gb = torch.Generator().manual_seed(1)
    n_samples, m = 10, 4
    x = torch.randn(n_samples, 6, generator=gb, dtype=torch.float64)
    y = torch.randint(0, m, (n_samples,), generator=gb)
    batch = (x, y)
    assert n_samples * m <= 40

    jac = materialize_jacobian(model, batch, params)
    u = ce_loss_direction(model(x), y)
    gp = torch.Generator().manual_seed(9)
    p_diag = torch.rand(n_params, generator=gp, dtype=torch.float64) * 3.0 + 0.05
    precond = Preconditioner.from_diagonal(p_diag)
    reference = preconditioned_reference(jac, u, p_diag)
    assert abs(reference["chi_pos_p_spectral"] - reference["chi_pos_p_ratio"]) <= 1e-12

    # chi_loss from vatis's closed form must agree with ||u||^2 — otherwise the
    # reference and the estimator are not measuring the same loss direction.
    # (chi_loss_cross_entropy returns fp32, hence the fp32 slack.)
    chi_loss_vatis = float(chi_loss_cross_entropy(model(x).detach(), y))
    assert abs(chi_loss_vatis - reference["chi_loss"]) <= fp32_output_slack(
        reference["chi_loss"]
    ), (
        f"vatis chi_loss {chi_loss_vatis!r} != ||u||^2 {reference['chi_loss']!r} "
        f"(abs {abs(chi_loss_vatis - reference['chi_loss']):.3e})"
    )

    flat = _flat_grad(ce_loss(model(x), batch), params)
    delta_loss_p = float(weighted_sum_sq_fp64(flat, p_diag))
    assert abs(delta_loss_p - reference["delta_loss_p"]) <= 1e-8, (
        f"delta_loss_P != u^T Theta_P u: {delta_loss_p!r} vs "
        f"{reference['delta_loss_p']!r} (abs {abs(delta_loss_p - reference['delta_loss_p']):.3e})"
    )

    n_h = 4096
    theta_p = (jac * p_diag.unsqueeze(0)) @ jac.T
    if method == "hutchinson":
        estimator: object = HutchinsonEstimator(n_hutchinson=n_h)
        probe_matrix = theta_p
    else:
        estimator = PerSequenceControlVariateEstimator(n_hutchinson=n_h)
        pi = loss_direction_projector(u, n_samples, m)
        eye = torch.eye(n_samples * m, dtype=torch.float64)
        probe_matrix = (eye - pi) @ theta_p @ (eye - pi)
    std = rademacher_trace_std(probe_matrix, n_h)
    assert std > 0.0, "this branch is meant to be the stochastic one"

    gen = torch.Generator().manual_seed(4)
    result = estimator.compute(  # type: ignore[attr-defined]
        model,
        batch,
        tuple_forward,
        loss_fn=ce_loss,
        valid_mask_fn=all_valid_mask,
        params=params,
        micro_batch_size=n_samples,
        generator=gen,
        preconditioner=precond,
    )
    assert result.chi_net_p is not None
    chi_net_p = float(result.chi_net_p)
    trace = reference["chi_net_p"]
    bound = K_SIGMA * std + fp32_output_slack(trace)
    deviation = abs(chi_net_p - trace)
    assert deviation <= bound, (
        f"{method}: chi_net_P {chi_net_p!r} vs Tr(Theta_P) {trace!r} — deviation "
        f"{deviation:.4e} = {deviation / std:.2f} sigma, bound {bound:.4e} "
        f"({K_SIGMA} sigma at n={n_h}, sigma={std:.4e})"
    )

    # (ii) propagate the same interval through chi_pos^P.
    numerator = delta_loss_p / reference["chi_loss"]
    lo = numerator / (trace + K_SIGMA * std)
    hi = numerator / (trace - K_SIGMA * std)
    pos = float(chi_pos(delta_loss_p, reference["chi_loss"], chi_net_p))
    exact_pos = reference["chi_pos_p_spectral"]
    assert lo - 1e-12 <= pos <= hi + 1e-12, (
        f"{method}: chi_pos_P {pos!r} outside the derived band [{lo!r}, {hi!r}] "
        f"around the exact {exact_pos!r}"
    )
    assert abs(pos - exact_pos) <= abs(exact_pos) * (K_SIGMA * std / trace) + 1e-9, (
        f"{method}: chi_pos_P {pos!r} vs exact {exact_pos!r} "
        f"(abs {abs(pos - exact_pos):.3e}, relative band "
        f"{K_SIGMA * std / trace:.3%})"
    )


# ========================================================= T-flattening-direction


def test_flattening_direction_gives_one_over_m_and_beats_raw() -> None:
    """T-flattening-direction: ``P = diag(pi)^-1`` flattens the spectrum to 1/M.

    Diagonal linear toy: ``J = diag(sqrt(pi))``, so ``Theta = diag(pi)`` *is* the
    prescribed imbalanced spectrum (a ``k^-2`` power law over ``M = 40`` modes,
    randomly permuted so no mode ordering is special). With ``P = diag(pi)^-1``,
    ``Theta_P = I_M``, hence ``chi_pos^P = 1/M`` exactly and independently of the
    residual — the ideal-optimizer signature of ``background.md`` §4.5 ("flat
    spectrum -> a horizontal line at 1/M").

    Direction check for D3 (``background.md`` §4.6): with a **tail-concentrated**
    residual (all weight on the smallest-``pi`` quarter of the modes) the raw
    ``chi_pos = sum_k gamma_k^2 p_k`` sits far *below* ``1/M``, because ``gamma``
    is on modes with ``p_k << 1/M``. So ``chi_pos < chi_pos^P``. The contrasting
    bulk-concentrated residual is checked too, so the assertion is about the
    residual's position in the spectrum and not an artifact of the toy.
    """
    n_samples, m = 10, 4
    big_m = n_samples * m
    g = torch.Generator().manual_seed(3)
    spectrum = torch.tensor([1.0 / (k + 1) ** 2.0 for k in range(big_m)], dtype=torch.float64)[
        torch.randperm(big_m, generator=g)
    ]
    jac = diagonal_spectrum_jacobian(spectrum)
    p_diag = 1.0 / spectrum
    precond = Preconditioner.from_diagonal(p_diag)
    model = LinearJacobianModel(jac.reshape(n_samples, m, big_m))
    params = [model.theta]
    index = torch.arange(n_samples)
    ascending = torch.argsort(spectrum)

    def run(weights: torch.Tensor) -> tuple[float, float, float, float]:
        loss_fn = make_linear_loss(weights)
        chi_loss = float((weights * weights).sum())
        flat = _flat_grad(loss_fn(model(index), index), params)
        delta_loss = float((flat * flat).sum())
        delta_loss_p = float(weighted_sum_sq_fp64(flat, p_diag))
        gen = torch.Generator().manual_seed(5)
        res = HutchinsonEstimator(n_hutchinson=8).compute(
            model,
            index,
            index_forward,
            loss_fn=loss_fn,
            valid_mask_fn=all_valid_mask,
            params=params,
            micro_batch_size=n_samples,
            generator=gen,
            preconditioner=precond,
        )
        assert res.chi_net_p is not None
        return (
            float(chi_pos(delta_loss, chi_loss, float(res.chi_net))),
            float(chi_pos(delta_loss_p, chi_loss, float(res.chi_net_p))),
            float(res.chi_net_p),
            delta_loss_p / (chi_loss * float(res.chi_net_p)),
        )

    # Theta_P = I again makes the trace estimate variance-free (up to the
    # round-off of the materialized Theta_P).
    theta_p = (jac * p_diag.unsqueeze(0)) @ jac.T
    assert rademacher_trace_std(theta_p, 8) < 1e-12 * big_m

    tail = torch.zeros(big_m, dtype=torch.float64)
    tail[ascending[: big_m // 4]] = 1.0
    raw_tail, p_tail, trace_p, p_tail_fp64 = run(tail.reshape(n_samples, m))

    target = 1.0 / big_m
    assert abs(trace_p - big_m) <= fp32_output_slack(float(big_m)), (
        f"Tr(Theta_P) must be M={big_m}, got {trace_p!r}"
    )
    slack = max(1e-8, fp32_output_slack(target))
    assert abs(p_tail - target) <= slack, (
        f"chi_pos_P != 1/M with P = diag(pi)^-1: {p_tail!r} vs {target!r} "
        f"(abs {abs(p_tail - target):.3e}, allowed {slack:.3e})"
    )
    assert abs(p_tail_fp64 - target) <= 1e-8, (
        f"chi_pos_P (fp64 components) != 1/M: {p_tail_fp64!r} vs {target!r} "
        f"(abs {abs(p_tail_fp64 - target):.3e})"
    )
    assert raw_tail < p_tail, (
        f"tail-concentrated residual: raw chi_pos {raw_tail!r} should be BELOW "
        f"chi_pos_P {p_tail!r} (the preconditioner flattens the tail away)"
    )
    assert raw_tail < target / 10.0, (
        f"tail-concentrated raw chi_pos {raw_tail!r} should be far below 1/M {target!r}"
    )

    # Contrast: a bulk-concentrated residual puts raw chi_pos ABOVE 1/M, and
    # chi_pos_P is unchanged at 1/M (it does not depend on the residual at all).
    bulk = torch.zeros(big_m, dtype=torch.float64)
    bulk[ascending[-(big_m // 4) :]] = 1.0
    raw_bulk, p_bulk, _, _ = run(bulk.reshape(n_samples, m))
    assert raw_bulk > target, (
        f"bulk-concentrated raw chi_pos {raw_bulk!r} should exceed 1/M {target!r}"
    )
    assert abs(p_bulk - target) <= slack, (
        f"chi_pos_P must be 1/M for ANY residual; bulk case gave {p_bulk!r}"
    )


# ==================================================================== T-batchsize

# Declared parameters of the statistical test below.
BATCHSIZE_REPEATS = 12
BATCHSIZE_SIGMA_K = 4.0
BATCHSIZE_RELATIVE_BOUND = 0.20


def test_batch_size_independence_of_components() -> None:
    """T-batchsize: ``E[components]`` independent of ``|A| in {8, 32, 128}``.

    Setup: iid Gaussian inputs (``x ~ N(0, I)``, ``in_dim = 8``) with labels
    ``argmax(x @ teacher)`` for a fixed random teacher, evaluated on a fixed
    fp32 MLP. Measured components: the sigma-normalized ``chi_loss`` and
    ``chi_net`` (and its P counterpart), ``delta_loss``/``delta_loss^P``, and
    ``chi_pos``/``chi_pos^P``.

    **Cross pairs, not self pairs — and why.** vatis's normalization
    (``chi_loss * sqrt(N_A N_B)``, ``chi_net / sqrt(N_A N_B)``) makes the two
    magnitude components ``|A|``-independent by construction. ``delta_loss`` is
    different: for a *self* pair ``delta_loss = ||mean_b g_b||^2`` has
    expectation ``||mu||^2 + tr(Sigma_g)/|A|``, an irreducible ``O(1/|A|)``
    minibatch term (the "mini-batch stochasticity smears it" remark of
    ``background.md`` §4.7), so ``E[chi_pos]`` genuinely *does* move with ``|A|``
    at small ``|A|`` and asserting otherwise would be asserting something false.
    For a *cross* pair with independent batches ``A`` and ``B``,
    ``E[<g_A, g_B>] = ||mu||^2`` exactly, with no ``1/|A|`` term — so the cross
    observables are the ones for which ``|A|``-independence is a true statement,
    and they are what this test asserts. The self-pair trend is *reported* (via
    the message of the final assertion) rather than asserted, so the ``O(1/|A|)``
    term stays visible instead of being quietly excluded.

    Statistical tolerance, declared: ``BATCHSIZE_REPEATS = 12`` independent
    ``(A, B)`` pairs per ``|A|`` (seeded), and two bounds, both of which must
    hold for every component:

    - a standard-error test, ``|mean_n - mean_128| <= 4 * sqrt(sem_n^2 +
      sem_128^2)`` with ``sem = stdev / sqrt(12)`` — the actual statistical
      statement, self-calibrating to each component's spread;
    - a fixed relative backstop of 20% from the ``|A| = 128`` mean, so the test
      cannot pass merely because the spread became huge.
    """
    in_dim, hidden, n_classes = 8, 12, 4
    torch.manual_seed(0)
    gm = torch.Generator().manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, n_classes)
    )
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gm) * 0.6)
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    teacher = torch.randn(in_dim, n_classes, generator=torch.Generator().manual_seed(77))
    gp = torch.Generator().manual_seed(5)
    p_diag = torch.rand(n_params, generator=gp) * 2.0 + 0.1
    precond = Preconditioner.from_diagonal(p_diag)

    def one_batch(n: int, seed: int) -> tuple[float, float, float, torch.Tensor]:
        x, y = make_gaussian_class_batch(
            n=n, in_dim=in_dim, teacher=teacher, seed=seed, dtype=torch.float32
        )
        batch = (x, y)
        chi_loss = float(chi_loss_cross_entropy(model(x).detach(), y))
        gen = torch.Generator().manual_seed(seed * 31 + 7)
        res = PerSequenceControlVariateEstimator(n_hutchinson=16).compute(
            model,
            batch,
            tuple_forward,
            loss_fn=ce_loss,
            valid_mask_fn=all_valid_mask,
            params=params,
            micro_batch_size=n,
            generator=gen,
            preconditioner=precond,
        )
        assert res.chi_net_p is not None
        grads = torch.autograd.grad(ce_loss(model(x), batch), params)
        flat = torch.cat([g.reshape(-1) for g in grads])
        return chi_loss, float(res.chi_net), float(res.chi_net_p), flat

    def cross_measurement(n: int, seed: int) -> dict[str, float]:
        cl_a, cn_a, cnp_a, g_a = one_batch(n, seed)
        cl_b, cn_b, cnp_b, g_b = one_batch(n, seed + 500_000)
        norm = float(n)  # sqrt(N_A * N_B) with N_A = N_B = n
        cl = (cl_a * cl_b) ** 0.5
        cn = (cn_a * cn_b) ** 0.5
        cnp = (cnp_a * cnp_b) ** 0.5
        delta = float((g_a.to(torch.float64) * g_b.to(torch.float64)).sum())
        delta_p = float(weighted_dot_fp64(g_a, g_b, p_diag))
        return {
            "chi_loss_normalized": cl * norm,
            "chi_net_normalized": cn / norm,
            "chi_net_P_normalized": cnp / norm,
            "delta_loss": delta,
            "delta_loss_P": delta_p,
            "chi_pos": float(chi_pos(delta, cl, cn)),
            "chi_pos_P": float(chi_pos(delta_p, cl, cnp)),
        }

    sizes = (8, 32, 128)
    keys = (
        "chi_loss_normalized",
        "chi_net_normalized",
        "chi_net_P_normalized",
        "delta_loss",
        "delta_loss_P",
        "chi_pos",
        "chi_pos_P",
    )
    stats: dict[int, dict[str, tuple[float, float]]] = {}
    for n in sizes:
        samples: dict[str, list[float]] = {k: [] for k in keys}
        for r in range(BATCHSIZE_REPEATS):
            out = cross_measurement(n, seed=1000 * n + r)
            for k in keys:
                samples[k].append(out[k])
        stats[n] = {
            k: (statistics.mean(v), statistics.stdev(v) / BATCHSIZE_REPEATS**0.5)
            for k, v in samples.items()
        }

    failures: list[str] = []
    report: list[str] = []
    for k in keys:
        ref_mean, ref_sem = stats[128][k]
        for n in (8, 32):
            mean, sem = stats[n][k]
            deviation = abs(mean - ref_mean)
            sigma = (sem**2 + ref_sem**2) ** 0.5
            rel = deviation / abs(ref_mean)
            report.append(
                f"{k} |A|={n}: mean={mean:.6g} vs |A|=128 {ref_mean:.6g} "
                f"({rel:.2%}, {deviation / sigma:.2f} sigma)"
            )
            if deviation > BATCHSIZE_SIGMA_K * sigma:
                failures.append(
                    f"{k}: |A|={n} mean {mean:.6g} vs |A|=128 {ref_mean:.6g} is "
                    f"{deviation / sigma:.2f} sigma > {BATCHSIZE_SIGMA_K}"
                )
            if rel > BATCHSIZE_RELATIVE_BOUND:
                failures.append(
                    f"{k}: |A|={n} mean {mean:.6g} vs |A|=128 {ref_mean:.6g} is "
                    f"{rel:.2%} > {BATCHSIZE_RELATIVE_BOUND:.0%}"
                )
    assert not failures, (
        "batch-size dependence detected in "
        + str(len(failures))
        + " comparison(s):\n"
        + "\n".join(failures)
        + "\nall comparisons:\n"
        + "\n".join(report)
    )
