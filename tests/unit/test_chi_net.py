"""Unit tests for the chi_net estimators (Hutchinson + per_seq_cv).

The convergence test runs an EXACT chi_net by iterating over the full output
dimension on a tiny model — so we keep the toy model deliberately small
(``S * V <= 1024``).
"""

from __future__ import annotations

import pytest
import torch

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
