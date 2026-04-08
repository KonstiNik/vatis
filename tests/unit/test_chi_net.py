"""Unit tests for the chi_net estimators (Hutchinson + per_seq_cv).

The convergence test runs an EXACT chi_net by iterating over the full output
dimension on a tiny model — so we keep the toy model deliberately small
(``S * V <= 1024``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.fixtures.tiny_transformer import (
    TinyTransformer,
    causal_lm_loss,
    causal_lm_valid_mask,
    make_tiny_lm_batch,
)
from vatis.core.chi_net import (
    HutchinsonEstimator,
    OpacusEstimator,
    PerSequenceControlVariateEstimator,
    select_chi_net_method,
)
from vatis.core.observables import (
    chi_loss_cross_entropy,
    chi_pos,
    delta_loss_cross,
    delta_loss_self,
)


def _build_small_lm() -> tuple[TinyTransformer, dict[str, torch.Tensor]]:
    torch.manual_seed(42)
    # Even smaller than the default to make the exact-trace test cheap.
    model = TinyTransformer(
        vocab_size=16, seq_len=8, d_model=16, n_layers=1, n_heads=2, d_ff=32
    ).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    batch = make_tiny_lm_batch(batch_size=2, seq_len=8, vocab_size=16, seed=0)
    return model, batch


def _exact_chi_net(model: TinyTransformer, batch: dict[str, torch.Tensor]) -> float:
    """Compute the exact ``Tr(M)`` by iterating over output dimensions.

    Sums ``||grad_theta logits[b,s,v]||^2`` over all valid (b, s, v) entries.
    Cost is ``S * V`` backwards per valid sequence — kept feasible by the
    tiny-model fixture.
    """
    params = list(model.parameters())
    logits = model(batch["input_ids"])
    mask = batch["labels"] != -100

    total = torch.zeros((), dtype=torch.float64)
    for b in range(logits.shape[0]):
        for s in range(logits.shape[1]):
            if not mask[b, s]:
                continue
            for v in range(logits.shape[2]):
                grads = torch.autograd.grad(
                    logits[b, s, v], params, retain_graph=True, allow_unused=True
                )
                for g in grads:
                    if g is None:
                        continue
                    total = total + (g.to(torch.float64) ** 2).sum()
    return float(total)


def _build_full_jacobian_and_u(
    model: TinyTransformer,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Materialize the explicit per-token Jacobian and the loss-direction vector.

    Returns:
        J_full: ``(R, P)`` fp64 tensor where ``R = n_valid_tokens * V`` and
            ``P`` is the total parameter count. Row ``r = idx(b,s,v)`` is
            ``∇_theta logits[b,s,v]`` flattened across all parameters; only
            valid (b,s) tokens are included (in row-major batch order).
        u_full: ``(R,)`` fp64 tensor with ``u_r = (1/N) * (softmax(z_{b,s})_v
            - δ_{v=y_{b,s}})``, the closed-form ``∂L/∂z`` for mean-reduced CE
            on ``N`` valid tokens. Same row ordering as ``J_full``.
        n_valid: number of valid (b,s) tokens (the ``N`` in the formula
            above).

    Cost: one autograd backward per (valid b, s, v) entry — only feasible
    on the smallest toy fixture (``S*V*B`` small). Output dtype is fp64
    so the matrix-form arithmetic downstream stays well-conditioned.
    """
    params = list(model.parameters())
    logits = model(batch["input_ids"])  # (B, S, V) — float (model dtype)
    labels = batch["labels"]
    mask = labels != -100  # (B, S)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        raise ValueError("batch has zero valid tokens; can't build a Jacobian")

    v_dim = logits.shape[-1]

    # u in closed form: (1/N) * (softmax - onehot) at valid positions.
    probs = F.softmax(logits.detach().to(dtype=torch.float64), dim=-1)
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    onehot = F.one_hot(safe_labels, num_classes=v_dim).to(dtype=torch.float64)
    diff = (probs - onehot) / float(n_valid)  # (B, S, V)

    j_rows: list[torch.Tensor] = []
    u_rows: list[torch.Tensor] = []
    for b in range(logits.shape[0]):
        for s in range(logits.shape[1]):
            if not bool(mask[b, s]):
                continue
            for v in range(v_dim):
                grads = torch.autograd.grad(
                    logits[b, s, v],
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                flat_parts: list[torch.Tensor] = []
                for p, g in zip(params, grads, strict=False):
                    if g is None:
                        flat_parts.append(torch.zeros(p.numel(), dtype=torch.float64))
                    else:
                        flat_parts.append(g.detach().to(torch.float64).reshape(-1))
                j_rows.append(torch.cat(flat_parts))
                u_rows.append(diff[b, s, v])

    j_full = torch.stack(j_rows, dim=0)  # (R, P)
    u_full = torch.stack(u_rows, dim=0)  # (R,)
    return j_full, u_full, n_valid


def _fwd(model: TinyTransformer, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(batch["input_ids"])


def test_hutchinson_unbiased_converges_to_exact() -> None:
    model, batch = _build_small_lm()
    exact = _exact_chi_net(model, batch)

    g = torch.Generator(device="cpu")
    g.manual_seed(123)
    est = HutchinsonEstimator(n_hutchinson=1024)
    res = est.compute(
        model,
        batch,
        _fwd,
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        params=list(model.parameters()),
        micro_batch_size=2,
        generator=g,
    )
    rel_err = abs(float(res.chi_net) - exact) / exact
    # 1024 probes on a tiny model should comfortably get to <2% rel err.
    assert rel_err < 0.02, f"Hutchinson rel err {rel_err:.4f} > 0.02"


def test_per_seq_cv_unbiased_converges_to_exact() -> None:
    model, batch = _build_small_lm()
    exact = _exact_chi_net(model, batch)

    g = torch.Generator(device="cpu")
    g.manual_seed(123)
    est = PerSequenceControlVariateEstimator(n_hutchinson=128)
    res = est.compute(
        model,
        batch,
        _fwd,
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        params=list(model.parameters()),
        micro_batch_size=2,
        generator=g,
    )
    rel_err = abs(float(res.chi_net) - exact) / exact
    # CV should be close to exact with way fewer probes than plain Hutchinson.
    assert rel_err < 0.05, f"per_seq_cv rel err {rel_err:.4f} > 0.05"


def test_per_seq_cv_records_alignment_matrix() -> None:
    model, batch = _build_small_lm()
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    est = PerSequenceControlVariateEstimator(n_hutchinson=4)
    res = est.compute(
        model,
        batch,
        _fwd,
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        params=list(model.parameters()),
        micro_batch_size=2,
        generator=g,
    )
    align = res.extras["alignment_matrix"]
    b = batch["input_ids"].shape[0]
    assert align.shape == (b, b)
    # Symmetric because <g_b, g_b'> = <g_b', g_b>.
    assert torch.allclose(align, align.T, atol=1e-6)
    # Diagonal entries are non-negative (squared norms).
    assert torch.all(align.diag() >= 0)


def test_select_chi_net_method_auto() -> None:
    assert select_chi_net_method(8, None) == "per_sequence_cv"
    assert select_chi_net_method(32, None) == "per_sequence_cv"
    assert select_chi_net_method(33, None) == "hutchinson"
    assert select_chi_net_method(1000, None) == "hutchinson"


def test_select_chi_net_method_user_override() -> None:
    assert select_chi_net_method(8, "hutchinson") == "hutchinson"
    assert select_chi_net_method(1000, "per_sequence_cv") == "per_sequence_cv"
    assert select_chi_net_method(8, "opacus") == "opacus"


def test_opacus_estimator_raises_not_implemented() -> None:
    est = OpacusEstimator()
    model, batch = _build_small_lm()
    with pytest.raises(NotImplementedError, match="opacus"):
        est.compute(
            model,
            batch,
            _fwd,
            loss_fn=causal_lm_loss,
            valid_mask_fn=causal_lm_valid_mask,
            params=list(model.parameters()),
            micro_batch_size=2,
        )


def test_per_seq_cv_memory_check_fires_for_oversized_config() -> None:
    """The startup memory check must reject configurations that would
    cache more than the configured fraction of available memory in
    per-sample gradient vectors.

    We pick a deliberately huge n_params (1B) so the check fires regardless
    of how much RAM/VRAM is on the host.
    """
    huge_n_params = 1_000_000_000  # 1B params -> 32 * 1B * 4 = 128 GB at B=32
    with pytest.raises(ValueError, match="per_sequence_cv would need"):
        PerSequenceControlVariateEstimator.check_memory_feasible(
            b_total=32,
            n_params=huge_n_params,
            device=None,  # use host RAM
        )


def test_per_seq_cv_memory_check_passes_for_sensible_config() -> None:
    """The check should not fire for normal toy-model configurations."""
    # 4 * 1M * 4 = 16 MB — trivially fits in any host.
    PerSequenceControlVariateEstimator.check_memory_feasible(
        b_total=4,
        n_params=1_000_000,
        device=None,
    )


def test_per_seq_cv_memory_check_fires_in_compute() -> None:
    """The estimator's compute() should also raise (not just the static
    helper) — this is the path the analyzer hits.
    """
    model, batch = _build_small_lm()
    # memory_fraction=0 forces the check to always fire (any positive
    # estimated bytes exceeds 0% of available).
    est = PerSequenceControlVariateEstimator(n_hutchinson=2, memory_fraction=1e-30)
    with pytest.raises(ValueError, match="per_sequence_cv would need"):
        est.compute(
            model,
            batch,
            _fwd,
            loss_fn=causal_lm_loss,
            valid_mask_fn=causal_lm_valid_mask,
            params=list(model.parameters()),
            micro_batch_size=2,
        )


def test_per_seq_cv_memory_check_does_not_fire_in_compute_for_toy() -> None:
    """The toy fixture must always pass the default memory check — otherwise
    the existing convergence test would fail."""
    model, batch = _build_small_lm()
    est = PerSequenceControlVariateEstimator(n_hutchinson=2)
    res = est.compute(
        model,
        batch,
        _fwd,
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        params=list(model.parameters()),
        micro_batch_size=2,
    )
    assert float(res.chi_net) > 0


def test_per_seq_cv_memory_fraction_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="memory_fraction"):
        PerSequenceControlVariateEstimator(memory_fraction=0.0)
    with pytest.raises(ValueError, match="memory_fraction"):
        PerSequenceControlVariateEstimator(memory_fraction=1.5)


def test_chi_pos_matches_exact_ntk() -> None:
    """End-to-end exact-NTK ground truth for chi_pos.

    Build the explicit per-token Jacobian ``J`` and the loss-direction
    vector ``u`` on the smallest toy fixture, then verify that vatis's
    ``chi_pos`` combinator output (using exact ``chi_loss``, exact
    ``chi_net``, and the autograd ``delta_loss``) matches the analytic
    Rayleigh quotient

        chi_pos = (u^T Θ u) / (||u||^2 * Tr Θ)         where Θ = J J^T

    to relative tolerance ``1e-5``. Tests both the math of the combinator
    and the equivalence between vatis's three building blocks
    (``chi_loss_cross_entropy``, ``_exact_chi_net``, ``delta_loss_self``)
    and the explicit-Jacobian formulation.
    """
    model, batch = _build_small_lm()
    params = list(model.parameters())

    j_full, u_full, _n_valid = _build_full_jacobian_and_u(model, batch)

    # Exact reference values from the explicit Jacobian.
    chi_loss_exact = float((u_full * u_full).sum())  # ||u||^2
    chi_net_exact = float((j_full * j_full).sum())  # ||J||_F^2 = Tr(J J^T)
    jt_u = j_full.T @ u_full  # (P,) — exact grad_theta L
    delta_loss_exact = float((jt_u * jt_u).sum())  # ||J^T u||^2 = u^T Θ u
    chi_pos_exact = delta_loss_exact / (chi_loss_exact * chi_net_exact)

    # Vatis's individual primitives — each one independently exact.
    logits_for_chi_loss = model(batch["input_ids"]).detach()
    chi_loss_vatis = float(chi_loss_cross_entropy(logits_for_chi_loss, batch["labels"]))
    chi_net_vatis = _exact_chi_net(model, batch)  # the existing exact helper
    loss = causal_lm_loss(model(batch["input_ids"]), batch)
    delta_loss_vatis = float(delta_loss_self(loss, params))
    chi_pos_vatis = float(chi_pos(delta_loss_vatis, chi_loss_vatis, chi_net_vatis))

    # Sanity-check the building blocks first so a failure points at the
    # responsible primitive.
    assert abs(chi_loss_vatis - chi_loss_exact) <= max(1e-12, abs(chi_loss_exact) * 1e-6), (
        f"chi_loss mismatch: vatis={chi_loss_vatis}, exact={chi_loss_exact}"
    )
    assert abs(chi_net_vatis - chi_net_exact) <= max(1e-12, abs(chi_net_exact) * 1e-6), (
        f"chi_net mismatch: vatis={chi_net_vatis}, exact={chi_net_exact}"
    )
    assert abs(delta_loss_vatis - delta_loss_exact) <= max(1e-12, abs(delta_loss_exact) * 1e-5), (
        f"delta_loss mismatch: vatis={delta_loss_vatis}, exact={delta_loss_exact}"
    )

    # Then the combined chi_pos to the requested 1e-5 tolerance.
    rel_err = abs(chi_pos_vatis - chi_pos_exact) / max(abs(chi_pos_exact), 1e-30)
    assert rel_err < 1e-5, (
        f"chi_pos rel err {rel_err:.2e} > 1e-5 (vatis={chi_pos_vatis}, exact={chi_pos_exact})"
    )


def test_delta_loss_cross_matches_exact_ntk_jacobian_product() -> None:
    """Cross delta_loss agrees with the explicit Jacobian product.

    For two batches A and B drawn from the same model:

        delta_loss(A, B) = <∇_θ L^A, ∇_θ L^B>
                         = u_A^T J_A J_B^T u_B

    The first equality is what vatis computes (gradient dot product); the
    second is the analytic Jacobian product. They should agree to
    relative tolerance ``1e-5``.
    """
    model, batch_a = _build_small_lm()
    params = list(model.parameters())
    batch_b = make_tiny_lm_batch(batch_size=2, seq_len=8, vocab_size=16, seed=7)

    # Path (a): vatis's gradient dot product.
    loss_a = causal_lm_loss(model(batch_a["input_ids"]), batch_a)
    loss_b = causal_lm_loss(model(batch_b["input_ids"]), batch_b)
    delta_loss_vatis = float(delta_loss_cross(loss_a, loss_b, params))

    # Path (b): explicit Jacobian product.
    j_a, u_a, _ = _build_full_jacobian_and_u(model, batch_a)
    j_b, u_b, _ = _build_full_jacobian_and_u(model, batch_b)
    # u_A^T (J_A J_B^T) u_B  ==  (J_A^T u_A) . (J_B^T u_B)
    g_a = j_a.T @ u_a  # (P,) exact grad_theta L_A
    g_b = j_b.T @ u_b  # (P,) exact grad_theta L_B
    delta_loss_exact = float((g_a * g_b).sum())

    rel_err = abs(delta_loss_vatis - delta_loss_exact) / max(abs(delta_loss_exact), 1e-30)
    assert rel_err < 1e-5, (
        f"delta_loss(A, B) rel err {rel_err:.2e} > 1e-5 "
        f"(vatis={delta_loss_vatis}, exact={delta_loss_exact})"
    )


def test_per_seq_cv_lower_variance_than_hutchinson_at_same_n() -> None:
    """Per-sample CV should have lower variance per probe than plain Hutchinson.

    We don't compute the variance directly — instead we run both at the same
    n_hutchinson with multiple seeds and check that the CV estimator's mean
    distance to the true value is no worse than Hutchinson's, in fact better.
    """
    model, batch = _build_small_lm()
    exact = _exact_chi_net(model, batch)

    n_h = 8
    n_seeds = 5

    h_errors = []
    cv_errors = []
    for seed in range(n_seeds):
        gh = torch.Generator(device="cpu")
        gh.manual_seed(seed * 7 + 1)
        h = HutchinsonEstimator(n_hutchinson=n_h).compute(
            model,
            batch,
            _fwd,
            loss_fn=causal_lm_loss,
            valid_mask_fn=causal_lm_valid_mask,
            params=list(model.parameters()),
            micro_batch_size=2,
            generator=gh,
        )
        h_errors.append(abs(float(h.chi_net) - exact))

        gc = torch.Generator(device="cpu")
        gc.manual_seed(seed * 7 + 1)
        c = PerSequenceControlVariateEstimator(n_hutchinson=n_h).compute(
            model,
            batch,
            _fwd,
            loss_fn=causal_lm_loss,
            valid_mask_fn=causal_lm_valid_mask,
            params=list(model.parameters()),
            micro_batch_size=2,
            generator=gc,
        )
        cv_errors.append(abs(float(c.chi_net) - exact))

    h_mean = sum(h_errors) / len(h_errors)
    c_mean = sum(cv_errors) / len(cv_errors)
    # The CV estimator should be at least as accurate; we use a slightly loose
    # bound (1.1×) because variance is itself noisy across only 5 seeds.
    assert c_mean <= h_mean * 1.1, (
        f"per_seq_cv mean abs err {c_mean:.4f} not better than hutchinson {h_mean:.4f}"
    )
