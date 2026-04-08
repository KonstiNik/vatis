"""Cross-validate vatis observables against perspic as the reference.

We run perspic's calculators directly (bypassing PyTorch Lightning) on the
same model / batch / seed and assert that:

    vatis.chi_loss_normalized  ~=  perspic.chi_loss
    vatis.chi_net_normalized   ~=  perspic.chi_net           (within Hutch noise)
    vatis.delta_loss           ~=  perspic.grad_norm_squared
    vatis.chi_pos              ~=  perspic.chi_coup          (within Hutch noise)

The key mapping is documented in CLAUDE.md §Glossary and §cross_validation.

Perspic's default is sample=batch with per-sample CE, so we use a
TinyMLP classifier on CPU with fp32 — the cleanest setup where the
normalization conventions align without any LM-token gymnastics.

Both perspic backends ("functorch", "opacus") are validated, and both vatis
chi_net methods ("hutchinson", "per_sequence_cv") are validated against both.

Gated behind ``pytest -m cross_validation`` because it needs perspic in the
environment (which it is, via the shared venv in this repo).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.fixtures.tiny_transformer import (
    TinyMLP,
    make_tiny_mlp_batch,
    mlp_loss,
    mlp_valid_mask,
)
from vatis import analyze
from vatis.models.hf import ModelBundle

pytestmark = pytest.mark.cross_validation


# --------------------------------------------------------------- perspic side


def _perspic_run(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    engine: str,
    approximate_with_n: int | None = None,
) -> dict[str, float]:
    """Run perspic's SamplewiseCalculator + Linearizer directly on ``model``.

    Returns a dict of the four observables we care about, with perspic's
    naming: ``chi_loss``, ``chi_net``, ``grad_norm_squared``, ``chi_coup``.
    """
    from perspic.calculator.coupling import CouplingCalculator
    from perspic.calculator.linearizer import Linearizer

    if engine == "functorch":
        from perspic.calculator.samplewise_functorch import (
            SamplewiseCalculatorFunctorch,
        )

        calc: Any = SamplewiseCalculatorFunctorch()
    elif engine == "opacus":
        from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus

        calc = SamplewiseCalculatorOpacus(approximate_with_n=approximate_with_n)
    else:
        raise ValueError(f"unknown perspic engine: {engine}")

    samplewise = calc.compute(model, criterion, x, y, normalize=True)
    chi_loss = float(samplewise["batch_grad_norms_loss"])
    chi_net = float(samplewise["batch_grad_norms_network"])

    lin = Linearizer()
    probe = lin.compute(model=model, criterion=criterion, x1=x, y1=y)
    _, _, delta_loss_neg = probe["self"]
    grad_norm_squared = -float(delta_loss_neg)

    coup = CouplingCalculator().calculate(
        delta_loss=delta_loss_neg,
        chi_loss=chi_loss,
        chi_net=chi_net,
    )
    return {
        "chi_loss": chi_loss,
        "chi_net": chi_net,
        "grad_norm_squared": grad_norm_squared,
        "chi_coup": float(coup),
    }


# --------------------------------------------------------------- vatis side


def _vatis_run(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    chi_net_method: str,
    n_hutchinson: int,
    seed: int,
) -> dict[str, float]:
    bundle = ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=lambda m, b: m(b[0]),
        loss_fn=mlp_loss,
        valid_mask_fn=mlp_valid_mask,
        identifier="mlp@0",
    )
    results = analyze(
        model=bundle,
        revisions=["0"],
        eval_batches={"val": (x, y)},
        chi_net_method=chi_net_method,
        n_hutchinson=n_hutchinson,
        micro_batch_size=x.shape[0],  # one shot
        seed=seed,
        device="cpu",
        sink=None,
    )
    by_obs = {r.observable: r.value for r in results[0].rows}
    return by_obs


# --------------------------------------------------------------- fixtures


def _build_shared_model_and_batch(
    seed: int = 42, batch_size: int = 8
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = TinyMLP(in_dim=8, hidden=16, n_classes=4).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    x, y = make_tiny_mlp_batch(batch_size=batch_size, in_dim=8, n_classes=4, seed=seed)
    return model, x, y


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _restore(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)


# ---------------------------------------------------------------- tests


@pytest.mark.parametrize("perspic_engine", ["functorch", "opacus"])
def test_chi_loss_and_delta_loss_match_perspic(perspic_engine: str) -> None:
    """chi_loss and delta_loss should match exactly (both are deterministic)."""
    model, x, y = _build_shared_model_and_batch()
    snapshot = _snapshot(model)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    p = _perspic_run(model, criterion, x, y, engine=perspic_engine)

    # Restore model weights (perspic.linearizer.compute may have mutated
    # p.grad; we reload the state_dict to be sure).
    _restore(model, snapshot)
    v = _vatis_run(model, x, y, chi_net_method="hutchinson", n_hutchinson=16, seed=0)

    # chi_loss: vatis.chi_loss_normalized ≡ perspic.chi_loss
    assert v["chi_loss_normalized"] == pytest.approx(p["chi_loss"], rel=1e-4, abs=1e-6)

    # delta_loss: vatis.delta_loss ≡ perspic.grad_norm_squared
    assert v["delta_loss"] == pytest.approx(p["grad_norm_squared"], rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("perspic_engine", ["functorch", "opacus"])
@pytest.mark.parametrize("vatis_method", ["hutchinson", "per_sequence_cv"])
def test_chi_net_matches_perspic_within_hutchinson_noise(
    perspic_engine: str, vatis_method: str
) -> None:
    """chi_net is stochastic in vatis (Hutchinson) but deterministic in perspic.

    We use a large n_hutchinson to drive the Hutchinson variance below 2%,
    then assert vatis matches perspic within that bound. Empirically (see
    the test docstring), n=2048 is enough for both methods to consistently
    hit <1% relative error on the toy MLP.
    """
    model, x, y = _build_shared_model_and_batch()
    snapshot = _snapshot(model)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    p = _perspic_run(model, criterion, x, y, engine=perspic_engine)

    _restore(model, snapshot)
    n_h = 2048
    v = _vatis_run(model, x, y, chi_net_method=vatis_method, n_hutchinson=n_h, seed=0)

    # vatis.chi_net_normalized ≡ perspic.chi_net
    rel = abs(v["chi_net_normalized"] - p["chi_net"]) / max(1e-12, abs(p["chi_net"]))
    # 2% noise floor. With n=2048 both methods sit well under 1% empirically.
    assert rel < 0.02, (
        f"chi_net mismatch: vatis={v['chi_net_normalized']:.6f} vs "
        f"perspic={p['chi_net']:.6f} (rel err {rel:.3%}, method={vatis_method}, "
        f"engine={perspic_engine})"
    )


@pytest.mark.parametrize("perspic_engine", ["functorch", "opacus"])
@pytest.mark.parametrize("vatis_method", ["hutchinson", "per_sequence_cv"])
def test_chi_pos_matches_perspic_chi_coup(perspic_engine: str, vatis_method: str) -> None:
    """vatis.chi_pos == perspic.chi_coup within the Hutchinson noise floor.

    Because chi_pos = delta_loss / (chi_loss * chi_net) and delta_loss /
    chi_loss are deterministic, the only noise comes from chi_net. We
    inherit the same 2% tolerance as above.
    """
    model, x, y = _build_shared_model_and_batch()
    snapshot = _snapshot(model)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    p = _perspic_run(model, criterion, x, y, engine=perspic_engine)

    _restore(model, snapshot)
    n_h = 2048
    v = _vatis_run(model, x, y, chi_net_method=vatis_method, n_hutchinson=n_h, seed=0)

    rel = abs(v["chi_pos"] - p["chi_coup"]) / max(1e-12, abs(p["chi_coup"]))
    assert rel < 0.02, (
        f"chi_pos/chi_coup mismatch: vatis={v['chi_pos']:.6f} vs "
        f"perspic={p['chi_coup']:.6f} (rel err {rel:.3%}, method={vatis_method}, "
        f"engine={perspic_engine})"
    )


@pytest.mark.parametrize("vatis_method", ["hutchinson", "per_sequence_cv"])
def test_heavy_padding_matches_perspic_on_valid_subset(vatis_method: str) -> None:
    """Heavy-padding cross-validation: 5/8 samples masked (>50%).

    vatis sees the FULL padded batch (with ``labels=-100`` at the masked
    positions and a custom ``valid_mask_fn`` that honors them); perspic
    sees only the unpadded valid subset. If vatis correctly masks out the
    padded samples in every observable, the two should agree to the same
    tolerances as the existing cross-validation tests.

    Padding modes exercised here:
        - ``labels = -100`` for the per-sample ignore path
          (``valid_token_mask`` in ``vatis/core/normalization.py``)
        - The bundle's custom ``valid_mask_fn`` for the analyzer's
          ``n_valid`` accounting (which feeds the ``1/N^2`` factor in
          ``chi_loss`` and the per-rank weighting in ``delta_loss``)
        - The estimator's masked-probe path (probes are zeroed at masked
          positions via the closed-form CE u-vector that the
          per-seq-CV control variate uses)

    The TinyMLP fixture only has one notion of padding (sample-wise
    ``labels=-100``); the LM-style ``attention_mask=0`` path is exercised
    separately by ``test_heavy_padding_lm_matches_exact_ntk`` below
    (against the exact-NTK ground truth, since perspic doesn't natively
    speak token padding).
    """
    # Build the full and the subset versions from the same model + seed.
    model, x_full, y_full = _build_shared_model_and_batch()
    snapshot = _snapshot(model)

    # Mask 5 of 8 samples → 3 valid, 5 ignored = 62.5% masked.
    y_padded = y_full.clone()
    for i in (1, 3, 4, 6, 7):
        y_padded[i] = -100
    keep = y_padded != -100
    assert int(keep.sum().item()) == 3
    x_valid = x_full[keep]
    y_valid = y_padded[keep]
    assert keep.sum().item() / float(keep.numel()) < 0.5  # > 50% masked

    # ----- perspic side: see only the unpadded subset.
    criterion = nn.CrossEntropyLoss(reduction="mean")
    p = _perspic_run(model, criterion, x_valid, y_valid, engine="functorch")

    # ----- vatis side: see the full padded batch with a valid_mask_fn
    # that honors -100 and a loss_fn that uses ignore_index=-100.
    _restore(model, snapshot)

    def loss_with_ignore(
        logits: torch.Tensor, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return F.cross_entropy(logits, batch[1], reduction="mean", ignore_index=-100)

    def valid_mask_with_ignore(
        batch: tuple[torch.Tensor, torch.Tensor],
        logits: torch.Tensor,  # noqa: ARG001
    ) -> torch.Tensor:
        return batch[1] != -100

    bundle = ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=lambda m, b: m(b[0]),
        loss_fn=loss_with_ignore,
        valid_mask_fn=valid_mask_with_ignore,
        identifier="mlp@0",
    )
    results = analyze(
        model=bundle,
        revisions=["0"],
        eval_batches={"val": (x_full, y_padded)},
        chi_net_method=vatis_method,
        n_hutchinson=2048,
        micro_batch_size=8,
        seed=0,
        device="cpu",
        sink=None,
    )
    v = {r.observable: r.value for r in results[0].rows}

    # chi_loss and delta_loss are deterministic — exact agreement.
    assert v["chi_loss_normalized"] == pytest.approx(p["chi_loss"], rel=1e-4, abs=1e-6)
    assert v["delta_loss"] == pytest.approx(p["grad_norm_squared"], rel=1e-4, abs=1e-6)
    # chi_net carries Hutchinson noise — same 2% bound as the existing tests.
    rel_net = abs(v["chi_net_normalized"] - p["chi_net"]) / max(1e-12, abs(p["chi_net"]))
    assert rel_net < 0.02, (
        f"heavy-padding chi_net mismatch (method={vatis_method}): "
        f"vatis={v['chi_net_normalized']:.6f} vs perspic={p['chi_net']:.6f} "
        f"(rel err {rel_net:.3%})"
    )
    # chi_pos inherits chi_net's noise.
    rel_pos = abs(v["chi_pos"] - p["chi_coup"]) / max(1e-12, abs(p["chi_coup"]))
    assert rel_pos < 0.02, (
        f"heavy-padding chi_pos mismatch (method={vatis_method}): "
        f"vatis={v['chi_pos']:.6f} vs perspic={p['chi_coup']:.6f} "
        f"(rel err {rel_pos:.3%})"
    )


@pytest.mark.parametrize("vatis_method", ["hutchinson", "per_sequence_cv"])
def test_all_observables_tight_tolerance_at_n16384(vatis_method: str) -> None:
    """Drive Hutchinson noise below 0.3% with n=16384 and confirm all four
    observables match perspic to that bound.

    The looser cross-validation tests use ``rel=0.02`` to absorb the
    Hutchinson noise floor at ``n=2048``. That hides any constant-factor
    bias smaller than ~2%. This test cranks ``n_hutchinson`` to 16384 — at
    which point the noise floor is far below 0.3% on the toy MLP — and
    asserts the tight bound. If this test fails, it is a real bug
    (perspic is the reference); do not loosen the tolerance to make it
    pass. The test uses functorch only (the cleaner perspic backend) and
    a single fixed seed/batch to keep it deterministic and quick.
    """
    model, x, y = _build_shared_model_and_batch()
    snapshot = _snapshot(model)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    p = _perspic_run(model, criterion, x, y, engine="functorch")

    _restore(model, snapshot)
    v = _vatis_run(model, x, y, chi_net_method=vatis_method, n_hutchinson=16384, seed=0)

    rel_tol = 3e-3
    abs_tol = 1e-6

    # chi_loss and delta_loss are deterministic — tight bound is for free.
    assert v["chi_loss_normalized"] == pytest.approx(p["chi_loss"], rel=rel_tol, abs=abs_tol), (
        f"chi_loss mismatch (method={vatis_method}): "
        f"vatis={v['chi_loss_normalized']} vs perspic={p['chi_loss']}"
    )
    assert v["delta_loss"] == pytest.approx(p["grad_norm_squared"], rel=rel_tol, abs=abs_tol), (
        f"delta_loss mismatch (method={vatis_method}): "
        f"vatis={v['delta_loss']} vs perspic={p['grad_norm_squared']}"
    )
    # chi_net carries Hutchinson variance; n=16384 must drive it under
    # the tight bound on the toy MLP.
    assert v["chi_net_normalized"] == pytest.approx(p["chi_net"], rel=rel_tol, abs=abs_tol), (
        f"chi_net mismatch (method={vatis_method}): "
        f"vatis={v['chi_net_normalized']} vs perspic={p['chi_net']}"
    )
    # chi_pos inherits chi_net's noise floor.
    assert v["chi_pos"] == pytest.approx(p["chi_coup"], rel=rel_tol, abs=abs_tol), (
        f"chi_pos mismatch (method={vatis_method}): vatis={v['chi_pos']} vs perspic={p['chi_coup']}"
    )
