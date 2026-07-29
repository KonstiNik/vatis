"""Does a preconditioned observable land on the *right parameter*?

``test_local_bundle.py`` checks that ``flat_state("exp_avg_sq")`` carries the
right constants in the right segments — i.e. alignment at the level of the flat
vector. This file closes the last link: that the **observable** weights each
parameter by that parameter's own entry of ``P``, end to end through
``Analyzer``.

The construction: give every parameter a distinct ``exp_avg_sq`` constant, so
``P`` is piecewise-constant with a known value per parameter, and assert

    delta_loss^P == sum_p P_p * ||grad_p||^2      (grad_p by name, from autograd)

Every case also asserts that *swapping* two of the per-parameter weights changes
the answer, so a permutation could have been caught — a green run with an
indistinguishable fixture would prove nothing.

Covered: the writer-side default order; a ``param_names.json`` that permutes it
(including the two identically-shaped hidden layers, which no shape check can
tell apart); tied weights; and the one misalignment vatis provably *cannot* see.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.fixtures.synthetic_bundle import (
    BundleTinyLM,
    bundle_metadata,
    handwritten_optim_state,
    make_batch_a,
    named_parameter_order,
    permuted_parameter_order,
    tiny_lm_factory,
    write_checkpoint,
    write_run_files,
)
from vatis.analyzer import Analyzer
from vatis.core.preconditioner import Preconditioner, weighted_sum_sq_fp64
from vatis.models.local_bundle import load_local_bundle

FP64 = torch.float64
STEP = 7
BETA2, EPS = 0.95, 1e-8

# Distinct, well-separated constants so P is piecewise-constant per parameter and
# a swap between any two segments is unmistakable.
CONSTANTS = {
    "embed.weight": 1.0,
    "fc1.weight": 4.0,
    "fc1.bias": 25.0,
    "fc2.weight": 100.0,
    "fc2.bias": 400.0,
    "head.weight": 2500.0,
}


def _expected_p_value(constant: float) -> float:
    """``1/(sqrt(v_hat) + eps)`` for a constant ``exp_avg_sq``."""
    v_hat = constant / (1.0 - BETA2**STEP)
    return 1.0 / (v_hat**0.5 + EPS)


def _write(
    tmp_path: Path,
    *,
    order: list[str] | None,
    tie_head: bool = False,
    param_indices: list[int] | None = None,
    state_order: list[str] | None = None,
) -> tuple[Path, BundleTinyLM, list[str]]:
    """Write a one-checkpoint bundle; ``order`` is ``param_names.json``'s order."""
    model = BundleTinyLM(tie_head=tie_head)
    run_dir = tmp_path / "bundle"
    written = write_run_files(
        run_dir,
        model,
        param_order=order,
        metadata=bundle_metadata(model),
    )
    optim = handwritten_optim_state(
        model,
        state_order if state_order is not None else written,
        step=STEP,
        exp_avg_sq_constants=CONSTANTS,
        betas=(0.9, BETA2),
        eps=EPS,
        param_indices=param_indices,
    )
    write_checkpoint(
        run_dir,
        0,
        model=model,
        optim_state=optim,
        batch_a=make_batch_a(seed=0),
        meta={"step": 0},
    )
    return run_dir, model, written


def _check_alignment(bundle, precond: Preconditioner, batch, *, label: str) -> None:
    """``delta_loss^P == sum_p P_p ||grad_p||^2``, with a swap-detectability guard."""
    logits = bundle.forward_fn(bundle.model, batch)
    loss = bundle.loss_fn(logits, batch)
    grads = torch.autograd.grad(loss, bundle.params, allow_unused=True)

    names = [n for n, _ in bundle.model.named_parameters()]
    by_id = {id(p): n for n, p in bundle.model.named_parameters()}
    reference = 0.0
    for param, grad in zip(bundle.params, grads, strict=True):
        if grad is None:
            continue
        constant = CONSTANTS[by_id[id(param)]]
        reference += _expected_p_value(constant) * float((grad.detach().to(FP64) ** 2).sum())

    analyzer = Analyzer(
        eval_batches={"a": batch},
        preconditioner=precond,
        chi_net_method="hutchinson",
        n_hutchinson=2,
        micro_batch_size=1,
        sink=None,
    )
    rows = {r.observable: r.value for r in analyzer.run([(bundle, "step0")])[0].rows}
    assert rows["delta_loss_P"] == pytest.approx(reference, rel=1e-5), (
        f"{label}: delta_loss_P = {rows['delta_loss_P']}, expected {reference}; param order {names}"
    )

    # Would a permutation have been visible at all? Re-weight with the
    # per-parameter constants assigned in REVERSED parameter order — a genuine
    # permutation of P over parameters, and one that works whatever the parameter
    # shapes are (unlike swapping two fixed segments, which is only representable
    # when the two happen to have the same numel). If this does not move the
    # answer, the fixture could not have caught a misalignment and the assertion
    # above proves nothing.
    flat = torch.cat(
        [
            torch.zeros(p.numel(), dtype=torch.float32) if g is None else g.reshape(-1).float()
            for p, g in zip(bundle.params, grads, strict=True)
        ]
    )
    constants = [CONSTANTS[by_id[id(p)]] for p in bundle.params]
    permuted = torch.cat(
        [
            torch.full((p.numel(),), _expected_p_value(c), dtype=precond.diag.dtype)
            for p, c in zip(bundle.params, reversed(constants), strict=True)
        ]
    )
    ok = float(weighted_sum_sq_fp64(flat, precond.diag))
    perturbed = float(weighted_sum_sq_fp64(flat, permuted))
    assert abs(ok - perturbed) > 1e-3 * ok, (
        f"{label}: reversing P over parameters did not move delta_loss^P — the "
        f"fixture cannot distinguish a misalignment, so the test is vacuous "
        f"({ok} vs {perturbed})"
    )


def _source(run_dir: Path):
    return load_local_bundle(run_dir, tiny_lm_factory, dtype="fp32", device="cpu")


def test_alignment_holds_for_the_writer_side_default_order(tmp_path: Path) -> None:
    run_dir, model, order = _write(tmp_path, order=None)
    assert order == named_parameter_order(model)
    source = _source(run_dir)
    ckpt = source.checkpoints[0]
    precond = source.adamw_preconditioner()(ckpt.identifier)
    assert precond is not None
    _check_alignment(ckpt.model_bundle, precond, ckpt.batch_a, label="default order")


def test_alignment_holds_when_param_names_json_permutes_the_model_order(tmp_path: Path) -> None:
    """The flat vectors must follow ``param_names.json``, not ``named_parameters()``.

    The permutation reverses the order, which also swaps the two identically
    shaped hidden layers relative to each other — the case no shape check can
    detect.
    """
    model_probe = BundleTinyLM()
    order = permuted_parameter_order(model_probe)
    run_dir, _model, written = _write(tmp_path, order=order)
    assert written == order != named_parameter_order(model_probe)

    source = _source(run_dir)
    ckpt = source.checkpoints[0]
    bundle = ckpt.model_bundle
    # bundle.params is the *named* lookup in file order
    named = dict(bundle.model.named_parameters())
    for param, name in zip(bundle.params, order, strict=True):
        assert param is named[name], name

    precond = source.adamw_preconditioner()(ckpt.identifier)
    assert precond is not None
    # P must carry each parameter's own constant in that parameter's segment
    offset = 0
    for name in order:
        n = named[name].numel()
        segment = precond.diag[offset : offset + n]
        assert torch.allclose(
            segment, torch.full_like(segment, _expected_p_value(CONSTANTS[name])), rtol=1e-6
        ), name
        offset += n
    _check_alignment(bundle, precond, ckpt.batch_a, label="permuted order")


def test_alignment_holds_with_tied_weights(tmp_path: Path) -> None:
    """A shared Parameter is listed once and its gradient is the sum of both uses."""
    run_dir, model, order = _write(tmp_path, order=None, tie_head=True)
    assert "head.weight" not in dict(model.named_parameters())
    assert "head.weight" in model.state_dict()
    assert "head.weight" not in order

    source = _source(run_dir)
    ckpt = source.checkpoints[0]
    bundle = ckpt.model_bundle
    assert sum(1 for p in bundle.params if p is bundle.model.embed.weight) == 1
    precond = source.adamw_preconditioner()(ckpt.identifier)
    assert precond is not None
    _check_alignment(bundle, precond, ckpt.batch_a, label="tied")


def test_same_shape_permutation_between_json_and_optimizer_is_undetectable(
    tmp_path: Path,
) -> None:
    """**A documented residual risk, pinned rather than asserted away.**

    ``param_names.json`` is *defined* to be index-aligned with the saved
    optimizer state, and vatis has nothing independent to check that against. If
    the writer swaps two identically-shaped parameters between the two files,
    every loader check passes, the flat ``P`` vector is **bit-identical** (the
    constants live at *indices*), and only the parameter each segment is applied
    to changes — so a P observable moves silently. The guarantee rests entirely on
    the bundle writer (WP0 component 01).

    This test exists so the risk is on record and measured; if a future version
    gains a cross-check that rejects this bundle, the test will fail and should be
    replaced by the rejection test.
    """
    model_probe = BundleTinyLM()
    truth = named_parameter_order(model_probe)
    swapped = list(truth)
    i, j = truth.index("fc1.weight"), truth.index("fc2.weight")
    swapped[i], swapped[j] = swapped[j], swapped[i]

    values = {}
    for tag, order in (("truth", truth), ("swapped", swapped)):
        # the optimizer state is written in the TRUE order in both cases; only
        # param_names.json disagrees.
        run_dir, _model, _written = _write(tmp_path / tag, order=order, state_order=truth)
        source = _source(run_dir)
        ckpt = source.checkpoints[0]
        bundle = ckpt.model_bundle  # must not raise: nothing can see the swap
        precond = source.adamw_preconditioner()(ckpt.identifier)
        assert precond is not None
        analyzer = Analyzer(
            eval_batches={"a": ckpt.batch_a},
            preconditioner=precond,
            chi_net_method="hutchinson",
            n_hutchinson=2,
            micro_batch_size=1,
            sink=None,
        )
        rows = {r.observable: r.value for r in analyzer.run([(bundle, "s")])[0].rows}
        values[tag] = (rows["delta_loss_P"], precond.diag.clone())

    assert torch.equal(values["truth"][1], values["swapped"][1]), (
        "the flat P is index-keyed, so a same-shape name swap cannot change it"
    )
    rel = abs(values["truth"][0] - values["swapped"][0]) / values["truth"][0]
    assert rel > 1e-3, (
        "the swap must move the observable, otherwise this fixture does not "
        "demonstrate the risk it documents"
    )


def test_optimizer_index_list_must_be_zero_based_and_ordered(tmp_path: Path) -> None:
    """A ``param_groups`` index list that is not ``[0..n-1]`` is refused."""
    from vatis.core.param_layout import LayoutMismatchError

    model_probe = BundleTinyLM()
    n = len(named_parameter_order(model_probe))
    run_dir, _model, _written = _write(tmp_path, order=None, param_indices=[1, 0, *range(2, n)])
    source = _source(run_dir)
    with pytest.raises(LayoutMismatchError, match="indices"):
        _ = source.checkpoints[0].optimizer_state
