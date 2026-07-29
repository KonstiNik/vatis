"""Regression tests: the ``valid_mask_fn`` / ``labels`` intersection must reach
**every** accumulator, not only ``chi_loss``.

History. v1.2 task 1 fixed ``chi_loss`` (and ``n_valid``) to use the
intersection of the bundle's ``valid_mask_fn`` with ``labels != ignore_index``.
The two other accumulators were left on the un-intersected mask, so with a
``valid_mask_fn`` that disagrees with the labels — the very configuration that
fix was written for, "e.g. all-True for an MLP with -100 labels" — vatis
reported:

- ``delta_loss`` rescaled by roughly ``(n_vmask / n_labels)^2`` **and dependent
  on ``micro_batch_size``** (measured 2.34x / 2.27x / 2.12x at micro = 1 / 2 / 4
  on this fixture's shape), because the per-micro-batch weight that reassembles
  ``grad_theta L`` was counted over ``valid_mask_fn`` while the global
  denominator used the intersection;
- ``chi_net`` (and ``chi_net^P``) as the trace over **all** output positions,
  because the Hutchinson probe was masked by ``valid_mask_fn`` alone — +37% on
  this fixture. A position whose label is ``ignore_index`` contributes nothing
  to the loss, so its Jacobian rows have no business in ``Tr(Theta)``.

``chi_pos = delta_loss / (chi_loss * chi_net)``, so the two composed into an
``O(1)``-wrong ``chi_pos`` with no non-finite value and no warning. Neither fired
on the shipped loaders (``causal_lm_valid_mask`` returns ``labels != -100``, so
the intersection is a no-op there) — which is exactly why they survived.

Ground truth here is the explicit per-token Jacobian
(``test_chi_net._build_full_jacobian_and_u``), which includes rows for valid
positions only, and the exact autograd gradient of the mean-reduced CE.
"""

from __future__ import annotations

import pytest
import torch

from tests.fixtures.tiny_transformer import (
    TinyTransformer,
    causal_lm_loss,
    make_tiny_lm_batch,
)
from tests.unit.test_chi_net import _build_full_jacobian_and_u
from vatis import analyze
from vatis.analyzer import Analyzer
from vatis.core.chi_net.base import ChiNetResult
from vatis.core.preconditioner import Preconditioner
from vatis.models.hf import ModelBundle

FP64 = torch.float64


def _all_true_mask(
    batch: dict[str, torch.Tensor],  # noqa: ARG001
    logits: torch.Tensor,
) -> torch.Tensor:
    """The disagreement source: claims every output position is valid."""
    return torch.ones(logits.shape[:-1], dtype=torch.bool, device=logits.device)


def _build_padded_fixture() -> tuple[
    TinyTransformer, dict[str, torch.Tensor], list[torch.nn.Parameter]
]:
    """The smallest fixture the explicit Jacobian is affordable on, heavily padded."""
    torch.manual_seed(42)
    model = TinyTransformer(
        vocab_size=16, seq_len=8, d_model=16, n_layers=1, n_heads=2, d_ff=32
    ).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    batch = make_tiny_lm_batch(batch_size=4, seq_len=8, vocab_size=16, pad_fraction=0.375, seed=0)
    n_valid = int((batch["labels"] != -100).sum().item())
    n_total = batch["labels"].numel()
    # The masks must genuinely disagree, or these tests prove nothing.
    assert 0 < n_valid < n_total, (n_valid, n_total)
    return model, batch, list(model.parameters())


def _bundle(
    model: TinyTransformer, params: list[torch.nn.Parameter], *, all_true: bool
) -> ModelBundle:
    from tests.fixtures.tiny_transformer import causal_lm_valid_mask

    return ModelBundle(
        model=model,
        params=params,
        forward_fn=lambda m, b: m(b["input_ids"]),
        loss_fn=causal_lm_loss,
        valid_mask_fn=_all_true_mask if all_true else causal_lm_valid_mask,
        identifier="toy@step0",
    )


def _exact_delta_loss(
    model: TinyTransformer, batch: dict[str, torch.Tensor], params: list[torch.nn.Parameter]
) -> float:
    """``||grad_theta L||^2`` for the mean-reduced CE over ``labels != -100``."""
    loss = causal_lm_loss(model(batch["input_ids"]), batch)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return sum(float((g.detach().to(FP64) ** 2).sum()) for g in grads if g is not None)


@pytest.mark.parametrize("micro", [1, 2, 4])
def test_delta_loss_is_exact_and_micro_batch_invariant_when_masks_disagree(micro: int) -> None:
    """``delta_loss`` must be ``||grad_theta L||^2`` at every micro-batch size."""
    model, batch, params = _build_padded_fixture()
    exact = _exact_delta_loss(model, batch, params)
    results = analyze(
        model=_bundle(model, params, all_true=True),
        revisions=["step0"],
        eval_batches={"lm": batch},
        chi_net_method="hutchinson",
        n_hutchinson=2,
        micro_batch_size=micro,
        sink=None,
    )
    rows = {r.observable: r.value for r in results[0].rows}
    assert rows["delta_loss"] == pytest.approx(exact, rel=1e-5), (
        f"delta_loss must not depend on micro_batch_size or on the un-intersected "
        f"mask count: got {rows['delta_loss']} at micro={micro}, exact {exact} "
        f"(ratio {rows['delta_loss'] / exact:.6f})"
    )


def test_chi_net_excludes_positions_the_labels_ignore() -> None:
    """``chi_net`` is the trace over the positions ``chi_loss`` uses, not all of them."""
    model, batch, params = _build_padded_fixture()
    j_full, _u_full, n_valid = _build_full_jacobian_and_u(model, batch)
    chi_net_exact = float((j_full * j_full).sum())
    results = analyze(
        model=_bundle(model, params, all_true=True),
        revisions=["step0"],
        eval_batches={"lm": batch},
        chi_net_method="hutchinson",
        n_hutchinson=1024,
        micro_batch_size=4,
        sink=None,
        seed=7,
    )
    rows = {r.observable: r.value for r in results[0].rows}
    n_reported = next(r.n_valid_a for r in results[0].rows)
    assert n_reported == n_valid
    # 1024 probes put the Hutchinson noise well under 5%; the pre-fix error was
    # +37%, which no amount of probes would remove.
    assert rows["chi_net"] == pytest.approx(chi_net_exact, rel=0.05), (
        f"chi_net = {rows['chi_net']}, exact over valid rows = {chi_net_exact} "
        f"(ratio {rows['chi_net'] / chi_net_exact:.4f})"
    )


def test_chi_net_p_excludes_the_same_positions() -> None:
    """The preconditioned ruler inherits the mask, so it must inherit the fix.

    Reference: ``Tr(J P J^T) = sum_i P_i ||J_{:, i}||^2`` from the explicit
    Jacobian, with a ``P`` that has real dynamic range (so a wrong column set
    cannot cancel).
    """
    model, batch, params = _build_padded_fixture()
    j_full, _u_full, _n = _build_full_jacobian_and_u(model, batch)
    n_params = sum(p.numel() for p in params)
    gen = torch.Generator().manual_seed(3)
    p_diag = torch.exp(torch.randn(n_params, generator=gen, dtype=FP64) * 2.0)
    chi_net_p_exact = float(((j_full * j_full) * p_diag.unsqueeze(0)).sum())

    results = analyze(
        model=_bundle(model, params, all_true=True),
        revisions=["step0"],
        eval_batches={"lm": batch},
        preconditioner=Preconditioner.from_diagonal(
            p_diag.to(dtype=torch.float32), descriptor="explicit"
        ),
        chi_net_method="hutchinson",
        n_hutchinson=1024,
        micro_batch_size=4,
        sink=None,
        seed=7,
    )
    rows = {r.observable: r.value for r in results[0].rows}
    assert rows["chi_net_P"] == pytest.approx(chi_net_p_exact, rel=0.10), (
        f"chi_net_P = {rows['chi_net_P']}, exact = {chi_net_p_exact} "
        f"(ratio {rows['chi_net_P'] / chi_net_p_exact:.4f})"
    )


def test_estimator_and_chi_loss_agree_on_the_valid_count() -> None:
    """The tripwire's precondition: both paths see the same mask, so equal counts.

    Checked for both estimators and several micro-batch sizes, with the
    disagreeing ``valid_mask_fn`` in place.
    """
    model, batch, params = _build_padded_fixture()
    n_valid = int((batch["labels"] != -100).sum().item())
    for method in ("hutchinson", "per_sequence_cv"):
        for micro in (1, 3, 4):
            analyzer = Analyzer(
                eval_batches={"lm": batch},
                chi_net_method=method,
                n_hutchinson=2,
                micro_batch_size=micro,
                sink=None,
            )
            results = analyzer.run([(_bundle(model, params, all_true=True), "step0")])
            assert next(r.n_valid_a for r in results[0].rows) == n_valid, (method, micro)


def test_valid_count_tripwire_fires_when_the_masks_drift_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guard that is never exercised is not a guard.

    Force the estimator to report a different valid-token count and assert the
    analyzer refuses to publish, rather than emitting a ``chi_pos`` rescaled by
    the ratio of the two masks.
    """
    model, batch, params = _build_padded_fixture()
    import vatis.analyzer as analyzer_mod

    real_build = analyzer_mod.build_estimator

    def lying_build(method: str, **kwargs: object) -> object:
        estimator = real_build(method, **kwargs)  # type: ignore[arg-type]
        real_compute = estimator.compute

        def compute(*args: object, **kw: object) -> ChiNetResult:
            result = real_compute(*args, **kw)  # type: ignore[arg-type]
            result.n_valid_tokens += 1
            return result

        estimator.compute = compute  # type: ignore[method-assign]
        return estimator

    monkeypatch.setattr(analyzer_mod, "build_estimator", lying_build)
    with pytest.raises(RuntimeError, match="valid output positions"):
        analyze(
            model=_bundle(model, params, all_true=False),
            revisions=["step0"],
            eval_batches={"lm": batch},
            chi_net_method="hutchinson",
            n_hutchinson=1,
            micro_batch_size=4,
            sink=None,
        )
