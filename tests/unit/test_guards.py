"""Cheap guards against silent wrong answers.

Three failure modes the correctness audit found could not be *seen* by a user:

1. a ``loss_fn`` that is not mean-reduced over the valid tokens ``chi_loss``
   normalizes by rescales ``chi_pos`` by exactly ``N_valid^2``, with no warning
   and no non-finite value — the only symptom is ``|chi_pos| > 1``, which nothing
   checked;
2. ``chi_pos`` divided in fp32, so an out-of-fp32-range denominator returned
   ``0.0``, indistinguishable from a true zero;
3. ``Analyzer`` unconditionally set ``requires_grad=True`` on
   ``ModelBundle.params``, so measuring a caller's model unfroze it.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F

from tests.fixtures.tiny_transformer import (
    TinyTransformer,
    causal_lm_loss,
    causal_lm_valid_mask,
    make_tiny_lm_batch,
)
from vatis import analyze
from vatis.core.observables import chi_pos, chi_pos_exceeds_cauchy_schwarz
from vatis.models.hf import ModelBundle

FP64 = torch.float64


def _bundle(model: TinyTransformer, loss_fn) -> ModelBundle:
    return ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=lambda m, b: m(b["input_ids"]),
        loss_fn=loss_fn,
        valid_mask_fn=causal_lm_valid_mask,
        identifier="toy@step0",
    )


def _model() -> TinyTransformer:
    torch.manual_seed(0)
    model = TinyTransformer(
        vocab_size=16, seq_len=8, d_model=16, n_layers=1, n_heads=2, d_ff=32
    ).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


# ------------------------------------------------------------------ chi_pos range


def test_chi_pos_divides_in_fp64_and_keeps_out_of_fp32_range_values() -> None:
    """The old fp32 division turned these into a silent ``0.0``."""
    assert float(chi_pos(1.0, 1e30, 1e30)) == pytest.approx(1e-60, rel=1e-12)
    assert float(chi_pos(1.0, 1e-30, 1e-30)) == pytest.approx(1e60, rel=1e-12)
    # and it no longer costs fp32 round-off on ordinary values
    assert float(chi_pos(0.1234567890123, 1.0, 1.0)) == 0.1234567890123
    assert chi_pos(1.0, 2.0, 4.0).dtype is torch.float64


def test_chi_pos_is_zero_only_when_the_denominator_is_exactly_zero() -> None:
    """chi_loss and chi_net are non-negative, so a zero product is genuinely
    degenerate (no valid tokens, or an identically zero Jacobian)."""
    assert float(chi_pos(1.0, 0.0, 5.0)) == 0.0
    assert float(chi_pos(1.0, 5.0, 0.0)) == 0.0
    assert float(chi_pos(0.0, 0.0, 0.0)) == 0.0
    # a tiny-but-nonzero denominator still divides
    assert float(chi_pos(1e-20, 1e-15, 1e-14)) == pytest.approx(1e9, rel=1e-12)
    # non-finite inputs propagate rather than being masked into a plausible 0
    assert not torch.isfinite(chi_pos(float("nan"), 1.0, 1.0))


def test_cauchy_schwarz_predicate() -> None:
    assert not chi_pos_exceeds_cauchy_schwarz(0.5)
    assert not chi_pos_exceeds_cauchy_schwarz(-1.0)
    assert not chi_pos_exceeds_cauchy_schwarz(1.4)  # inside the stochastic slack
    assert chi_pos_exceeds_cauchy_schwarz(1.6)
    assert chi_pos_exceeds_cauchy_schwarz(-4.9)
    assert chi_pos_exceeds_cauchy_schwarz(1.05, slack=0.0)


# --------------------------------------------------- the sum-reduced loss warning


def test_non_mean_reduced_loss_fn_warns_that_chi_pos_left_its_range() -> None:
    """A sum-reduced CE inflates chi_pos by exactly ``N_valid^2``; that must warn.

    ``chi_loss`` always uses the mean-reduced closed form (a documented v1
    restriction), so ``delta_loss`` and ``chi_loss`` end up describing different
    losses. The audit measured chi_pos going 7.587e-02 -> 4.856e+00 on an
    8-valid-token batch, i.e. exactly 64x, with nothing to alert the user.
    """
    model = _model()
    # n_valid = 56 here, so the N^2 = 3136 inflation carries chi_pos from ~1.1e-03
    # to ~3.4e+00 and crosses the bound. See
    # test_the_range_warning_is_a_backstop_not_a_detector for why the batch has to
    # be this big.
    batch = make_tiny_lm_batch(batch_size=8, seq_len=8, vocab_size=16, seed=9)
    n_valid = int((batch["labels"] != -100).sum())
    assert n_valid == 56

    def sum_reduced(logits: torch.Tensor, b: dict[str, torch.Tensor]) -> torch.Tensor:
        return causal_lm_loss(logits, b) * n_valid

    with pytest.warns(UserWarning, match=r"outside the \[-1, 1\] range"):
        rows = analyze(
            model=_bundle(model, sum_reduced),
            revisions=["step0"],
            eval_batches={"lm": batch},
            chi_net_method="hutchinson",
            n_hutchinson=8,
            micro_batch_size=2,
            sink=None,
        )[0].rows
    values = {r.observable: r.value for r in rows}
    assert abs(values["chi_pos"]) > 1.5

    # and the same run with the correct loss must NOT warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = analyze(
            model=_bundle(_model(), causal_lm_loss),
            revisions=["step0"],
            eval_batches={"lm": batch},
            chi_net_method="hutchinson",
            n_hutchinson=8,
            micro_batch_size=2,
            sink=None,
        )[0].rows
    assert abs(next(r.value for r in ok if r.observable == "chi_pos")) <= 1.5


def test_the_range_warning_is_a_backstop_not_a_detector() -> None:
    """Honest limit of the guard: it fires only when ``N^2 * chi_pos > 1.5``.

    The inconsistency scales the *published* value by ``N_valid^2``, so the guard
    sees it only if the result actually leaves ``[-1, 1]``. On a realistic batch
    (``N`` in the thousands) that is true for essentially any ``chi_pos``. On a tiny
    toy it is not: measured on this fixture at ``B=2, S=8`` — ``N = 14``,
    ``N^2 = 196``, ``chi_pos = 4.34e-03`` — the sum-reduced run lands at
    ``8.5e-01``, still inside the bound, and **no warning fires** even though the
    numbers are just as wrong. Recorded so nobody reads a silent run as a clean
    bill of health.
    """
    small = make_tiny_lm_batch(batch_size=2, seq_len=8, vocab_size=16, seed=9)
    n_small = int((small["labels"] != -100).sum())
    assert n_small == 14

    def sum_reduced(logits: torch.Tensor, b: dict[str, torch.Tensor]) -> torch.Tensor:
        return causal_lm_loss(logits, b) * n_small

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = analyze(
            model=_bundle(_model(), sum_reduced),
            revisions=["step0"],
            eval_batches={"lm": small},
            chi_net_method="hutchinson",
            n_hutchinson=4,
            micro_batch_size=2,
            sink=None,
        )[0].rows
    value = next(r.value for r in rows if r.observable == "chi_pos")
    assert 0.0 < abs(value) < 1.5, value
    assert not [w for w in caught if w.category is UserWarning]


def test_the_ratio_between_the_two_losses_is_exactly_n_valid_squared() -> None:
    """Pin the mechanism, so the warning's explanation stays true."""
    batch = make_tiny_lm_batch(batch_size=8, seq_len=8, vocab_size=16, seed=9)
    n_valid = int((batch["labels"] != -100).sum())

    def sum_reduced(logits: torch.Tensor, b: dict[str, torch.Tensor]) -> torch.Tensor:
        return causal_lm_loss(logits, b) * n_valid

    out = {}
    for label, fn in (("mean", causal_lm_loss), ("sum", sum_reduced)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = analyze(
                model=_bundle(_model(), fn),
                revisions=["step0"],
                eval_batches={"lm": batch},
                chi_net_method="hutchinson",
                n_hutchinson=4,
                micro_batch_size=2,
                sink=None,
            )[0].rows
        out[label] = {r.observable: r.value for r in rows}
    assert out["sum"]["chi_loss"] == out["mean"]["chi_loss"]
    assert out["sum"]["chi_net"] == out["mean"]["chi_net"]
    ratio = out["sum"]["chi_pos"] / out["mean"]["chi_pos"]
    assert ratio == pytest.approx(float(n_valid**2), rel=1e-5)


# -------------------------------------------------------------- requires_grad


def test_analyzer_restores_requires_grad_on_a_caller_supplied_bundle() -> None:
    """Measuring a model must not mutate it."""
    model = _model()
    params = list(model.parameters())
    frozen = params[1]
    frozen.requires_grad_(False)
    batch = make_tiny_lm_batch(batch_size=2, seq_len=8, vocab_size=16, seed=1)
    flags_before = [p.requires_grad for p in params]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyze(
            model=_bundle(model, causal_lm_loss),
            revisions=["step0"],
            eval_batches={"lm": batch},
            chi_net_method="hutchinson",
            n_hutchinson=2,
            micro_batch_size=2,
            sink=None,
        )
    assert [p.requires_grad for p in params] == flags_before
    assert not frozen.requires_grad, "the analyzer left a caller-frozen parameter unfrozen"


def test_requires_grad_is_restored_even_when_the_run_raises() -> None:
    """The restore is in a ``finally``, so a failure mid-checkpoint cannot leak it."""
    model = _model()
    params = list(model.parameters())
    params[0].requires_grad_(False)

    def exploding_loss(logits: torch.Tensor, b: dict[str, torch.Tensor]) -> torch.Tensor:
        raise RuntimeError("boom")

    batch = make_tiny_lm_batch(batch_size=2, seq_len=8, vocab_size=16, seed=1)
    with pytest.raises(RuntimeError, match="boom"):
        analyze(
            model=_bundle(model, exploding_loss),
            revisions=["step0"],
            eval_batches={"lm": batch},
            chi_net_method="hutchinson",
            n_hutchinson=2,
            micro_batch_size=2,
            sink=None,
        )
    assert not params[0].requires_grad


def test_analyzer_still_computes_gradients_for_initially_frozen_params() -> None:
    """Restoring the flags must not mean the frozen parameter went unmeasured.

    ``ModelBundle.params`` is the declared measurement set, so everything in it is
    measured during the run whatever its incoming flag was — the flag is only put
    back afterwards.
    """
    batch = make_tiny_lm_batch(batch_size=2, seq_len=8, vocab_size=16, seed=1)
    model_a = _model()
    rows_a = analyze(
        model=_bundle(model_a, causal_lm_loss),
        revisions=["step0"],
        eval_batches={"lm": batch},
        chi_net_method="hutchinson",
        n_hutchinson=4,
        micro_batch_size=2,
        sink=None,
        seed=3,
    )[0].rows
    model_b = _model()
    list(model_b.parameters())[1].requires_grad_(False)
    rows_b = analyze(
        model=_bundle(model_b, causal_lm_loss),
        revisions=["step0"],
        eval_batches={"lm": batch},
        chi_net_method="hutchinson",
        n_hutchinson=4,
        micro_batch_size=2,
        sink=None,
        seed=3,
    )[0].rows
    a = {r.observable: r.value for r in rows_a}
    b = {r.observable: r.value for r in rows_b}
    for observable, value in a.items():
        assert b[observable] == pytest.approx(value, rel=1e-9), observable


def test_chi_loss_closed_form_is_unaffected_by_the_fp64_chi_pos_change() -> None:
    """Guard: only the combinator moved, not the components."""
    model = _model()
    batch = make_tiny_lm_batch(batch_size=3, seq_len=8, vocab_size=16, pad_fraction=0.25, seed=4)
    rows = analyze(
        model=_bundle(model, causal_lm_loss),
        revisions=["step0"],
        eval_batches={"lm": batch},
        chi_net_method="hutchinson",
        n_hutchinson=4,
        micro_batch_size=1,
        sink=None,
    )[0].rows
    values = {r.observable: r.value for r in rows}
    mask = batch["labels"] != -100
    n_valid = int(mask.sum())
    with torch.no_grad():
        logits = model(batch["input_ids"])
    probs = F.softmax(logits.to(FP64), dim=-1)
    safe = batch["labels"].clone()
    safe[~mask] = 0
    onehot = F.one_hot(safe, num_classes=16).to(FP64)
    diff = (probs - onehot) * mask.unsqueeze(-1).to(FP64)
    expected = float((diff * diff).sum()) / n_valid**2
    assert values["chi_loss"] == pytest.approx(expected, rel=1e-6)
    assert values["chi_pos"] == pytest.approx(
        values["delta_loss"] / (values["chi_loss"] * values["chi_net"]), rel=1e-12
    )
