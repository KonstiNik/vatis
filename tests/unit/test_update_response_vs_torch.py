"""The update-response rule against what ``torch.optim`` really does (fp64).

``vatis.core.update_response`` rebuilds one optimizer step from the state saved
before it and splits ``grad L^B . dtheta`` into a gradient, a history and a
weight-decay part. Every assertion here holds that against a real optimizer:
the tiny transformer in fp64, a real ``step()``, the weights before and after.

Three kinds of check, because a formula can be wrong in three ways:

1. **The whole update.** The predicted ``theta_{t+1} - theta_t`` equals the one
   torch applied, element by element — and so the three terms sum to
   ``g . dtheta_actual`` for any ``g``. Over several steps including the first,
   with weight decay, tied and untied weights, per-group hyper-parameters, a
   learning-rate schedule and gradient clipping.
2. **Each part separately.** A mistake that moves mass between the parts leaves
   the sum intact. The parts are therefore pinned by *ablating the real
   optimizer*: the same step re-run with the weight decay switched off, and
   with the first moment zeroed, isolates exactly the decay part and the history
   part of what torch did — no formula is typed in twice.
3. **What must be refused.** Optimizers and options the rule does not cover.

The tolerances are fp64 round-off: the update is ``O(lr)`` and is read off as a
difference of weights of ``O(1)``, so a relative ``1e-9`` on a dot product is
loose by several orders and still far below every modelling error these tests
are meant to expose (the smallest of them, a decay that is coupled instead of
decoupled, moves the result by percent).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn

from tests.fixtures.update_response_toys import (
    RecordedStep,
    TrainingRecord,
    decay_groups,
    flat,
    held_out_batch,
    make_model,
    optimizer_parameters,
    record_steps,
)
from vatis.core.param_layout import ParameterLayout
from vatis.core.preconditioner import Preconditioner, weighted_dot_fp64
from vatis.core.update_response import (
    AdamHyper,
    OptimizerUpdate,
    SGDHyper,
    UpdateResponseTerms,
    weight_delta_dot_fp64,
)
from vatis.models.local_bundle import (
    BundleMetadata,
    BundleSchemaError,
    OptimizerStateView,
    optimizer_update_from_state,
)

N_STEPS = 6
RTOL = 1e-9

OptimizerFactory = Callable[[nn.Module], torch.optim.Optimizer]


def _adamw(**kwargs: Any) -> OptimizerFactory:
    defaults: dict[str, Any] = {"lr": 3e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1}
    defaults.update(kwargs)
    return lambda model: torch.optim.AdamW(model.parameters(), **defaults)


def _adam(**kwargs: Any) -> OptimizerFactory:
    defaults: dict[str, Any] = {"lr": 3e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1}
    defaults.update(kwargs)
    return lambda model: torch.optim.Adam(model.parameters(), **defaults)


def _sgd(**kwargs: Any) -> OptimizerFactory:
    defaults: dict[str, Any] = {"lr": 0.05}
    defaults.update(kwargs)
    return lambda model: torch.optim.SGD(model.parameters(), **defaults)


def _rule(
    record: TrainingRecord,
    step: RecordedStep,
    *,
    clip_grad_norm: float | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OptimizerUpdate:
    """The rule vatis builds from the state saved before ``step``."""
    layout = ParameterLayout.from_records(
        [[name, list(shape)] for name, shape in zip(record.names, record.shapes, strict=True)]
    )
    view = OptimizerStateView(step.state_dict, layout)
    kwargs.setdefault("checkpoint_step", step.index)
    return optimizer_update_from_state(
        view,
        metadata=None if metadata is None else BundleMetadata(raw=metadata),
        clip_grad_norm=clip_grad_norm,
        **kwargs,
    )


def _assert_update_reproduced(
    record: TrainingRecord, *, clip_grad_norm: float | None = None
) -> None:
    """Predicted update == applied update, elementwise, and the terms sum to it."""
    batch_b = held_out_batch()
    for step in record.steps:
        rule = _rule(record, step, clip_grad_norm=clip_grad_norm)
        predicted = rule.predicted_delta(step.grad_update, step.params_before)
        actual = step.delta
        scale = float(actual.abs().max())
        assert scale > 0.0
        worst = float((predicted - actual).abs().max())
        assert worst <= 1e-10 * scale, (
            f"update {step.index + 1}: predicted and applied update differ by {worst:.3e} "
            f"(largest applied entry {scale:.3e})"
        )
        _loss, grad_b = record.loss_and_grad(step.params_before, batch_b)
        terms = rule.response(
            step.grad_update, {"A": step.grad_update, "B": grad_b}, step.params_before
        )
        for name, grad in (("A", step.grad_update), ("B", grad_b)):
            reference = float(grad @ actual)
            assert terms[name].total == pytest.approx(reference, rel=RTOL), (
                f"update {step.index + 1}, batch {name}: grad + hist + wd = "
                f"{terms[name].total!r}, g . dtheta_actual = {reference!r}"
            )
            from_weights = float(weight_delta_dot_fp64(grad, step.params_after, step.params_before))
            assert from_weights == pytest.approx(reference, rel=1e-12)


# ---------------------------------------------------------------- the whole update


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_adamw(), id="golden-like"),
        pytest.param(_adamw(betas=(0.9, 0.999), weight_decay=1.0), id="strong-decay"),
        pytest.param(_adamw(weight_decay=0.0), id="no-decay"),
        pytest.param(_adamw(betas=(0.0, 0.999)), id="no-momentum"),
        pytest.param(_adamw(eps=1e-3), id="large-eps"),
        pytest.param(_adamw(foreach=False), id="single-tensor-path"),
    ],
)
@pytest.mark.parametrize("tie_head", [True, False], ids=["tied", "untied"])
def test_adamw_update_is_reproduced_from_the_pre_step_state(
    factory: OptimizerFactory, tie_head: bool
) -> None:
    _assert_update_reproduced(record_steps(factory, N_STEPS, tie_head=tie_head))


def test_adamw_first_update_starts_from_an_empty_state() -> None:
    """Update 1 has no saved state at all: zero moments, exponent 1, sign descent."""
    record = record_steps(_adamw(weight_decay=0.0, eps=1e-12), 1)
    step = record.steps[0]
    assert step.state_dict["state"] == {}
    rule = _rule(record, step)
    assert rule.step == 0 and rule.first_moment is None and rule.second_moment is None
    predicted = rule.predicted_delta(step.grad_update, step.params_before)
    # v_hat_1 = g**2, so the first step is -lr * g / (|g| + eps): -lr * sign(g)
    # wherever |g| is well above eps.
    moved = step.grad_update.abs() > 1e-5
    assert int(moved.sum()) > 1000
    assert torch.allclose(
        predicted[moved], -3e-3 * torch.sign(step.grad_update[moved]), rtol=1e-6, atol=0.0
    )
    terms = rule.response(step.grad_update, {"A": step.grad_update}, step.params_before)["A"]
    assert terms.hist == 0.0 and terms.wd == 0.0
    assert terms.total == pytest.approx(float(step.grad_update @ step.delta), rel=RTOL)


def test_per_group_weight_decay_and_learning_rate_are_exact() -> None:
    """No decay on biases / norms, and a different rate there: per parameter, exact."""
    record = record_steps(
        lambda model: torch.optim.AdamW(
            decay_groups(model, weight_decay=0.5, lr_no_decay=1e-3),
            lr=3e-3,
            betas=(0.9, 0.95),
        ),
        N_STEPS,
    )
    rule = _rule(record, record.steps[-1])
    decays = {(h.lr, h.weight_decay) for h in rule.hypers}
    assert decays == {(3e-3, 0.5), (1e-3, 0.0)}
    _assert_update_reproduced(record)


def test_per_group_betas_and_eps_are_exact() -> None:
    def factory(model: nn.Module) -> torch.optim.Optimizer:
        groups = decay_groups(model, weight_decay=0.1)
        groups[1].update(betas=(0.8, 0.99), eps=1e-6)
        return torch.optim.AdamW(groups, lr=3e-3, betas=(0.9, 0.95))

    _assert_update_reproduced(record_steps(factory, N_STEPS))


def test_scheduled_learning_rate_is_the_one_saved_before_the_step() -> None:
    """A warm-up that changes the rate every step: the saved group's ``lr`` is the one used."""
    record = record_steps(
        _adamw(),
        N_STEPS,
        make_scheduler=lambda opt: torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=N_STEPS
        ),
    )
    rates = [step.lr for step in record.steps]
    assert len(set(rates)) == N_STEPS, "the schedule did not move the rate"
    for step in record.steps:
        assert _rule(record, step).hypers[0].lr == step.lr
    _assert_update_reproduced(record)


@pytest.mark.parametrize("max_norm", [0.05, 0.5, 1e6], ids=["tight", "sometimes", "inactive"])
def test_global_norm_clipping_is_applied_as_torch_applies_it(max_norm: float) -> None:
    record = record_steps(_adamw(), N_STEPS, clip_grad_norm=max_norm)
    active = 0
    for step in record.steps:
        rule = _rule(record, step, clip_grad_norm=max_norm)
        seen = float(step.grad_seen.norm()) / float(step.grad_update.norm())
        assert rule.clip_coefficient(step.grad_update) == pytest.approx(seen, rel=1e-12)
        active += seen < 1.0
    assert (active > 0) == (max_norm < 1e6)
    _assert_update_reproduced(record, clip_grad_norm=max_norm)


def test_an_unstated_clip_is_visible_as_a_wrong_update() -> None:
    """The rule does not hide a clip it was not told about (the bundle must state it)."""
    record = record_steps(_adamw(), N_STEPS, clip_grad_norm=0.05)
    step = record.steps[-1]
    unclipped = _rule(record, step).predicted_delta(step.grad_update, step.params_before)
    relative = float((unclipped - step.delta).norm() / step.delta.norm())
    assert relative > 1e-3


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_adam(), id="coupled-decay"),
        pytest.param(_adam(weight_decay=0.0), id="no-decay"),
        pytest.param(_adam(betas=(0.5, 0.9), weight_decay=1.0), id="strong-decay"),
    ],
)
def test_adam_with_coupled_decay_is_reproduced(factory: OptimizerFactory) -> None:
    record = record_steps(factory, N_STEPS)
    assert all(not h.decoupled_weight_decay for h in _rule(record, record.steps[0]).hypers)
    _assert_update_reproduced(record)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_sgd(), id="plain"),
        pytest.param(_sgd(weight_decay=0.1), id="decay"),
        pytest.param(_sgd(momentum=0.9), id="momentum"),
        pytest.param(_sgd(momentum=0.9, dampening=0.3, weight_decay=0.1), id="dampening"),
        pytest.param(_sgd(momentum=0.9, nesterov=True, weight_decay=0.1), id="nesterov"),
    ],
)
def test_sgd_update_is_reproduced(factory: OptimizerFactory) -> None:
    _assert_update_reproduced(record_steps(factory, N_STEPS))


def test_sgd_with_per_group_momentum_is_reproduced() -> None:
    def factory(model: nn.Module) -> torch.optim.Optimizer:
        groups = decay_groups(model, weight_decay=0.1, lr_no_decay=0.01)
        groups[1]["momentum"] = 0.0
        return torch.optim.SGD(groups, lr=0.05, momentum=0.9)

    record = record_steps(factory, N_STEPS)
    rule = _rule(record, record.steps[-1])
    assert rule.first_moment is not None
    assert any(m is None for m in rule.first_moment) and any(
        m is not None for m in rule.first_moment
    )
    _assert_update_reproduced(record)


# ------------------------------------------------------- each part, by ablation


def _replay(
    record: TrainingRecord,
    step: RecordedStep,
    make_optimizer: OptimizerFactory,
    *,
    edit_state: Callable[[dict[str, Any]], None] | None = None,
    edit_groups: Callable[[dict[str, Any]], None] | None = None,
) -> torch.Tensor:
    """Re-run ``step`` with the real optimizer after editing its saved state.

    Returns the update the edited optimizer applies to ``theta_t`` when handed
    the gradient the original step saw.
    """
    model = make_model(dtype=record.dtype, tie_head=record.tie_head, seed=record.seed)
    optimizer = make_optimizer(model)
    params = optimizer_parameters(optimizer)
    state = copy.deepcopy(step.state_dict)
    if edit_state is not None:
        edit_state(state)
    optimizer.load_state_dict(state)
    if edit_groups is not None:
        for group in optimizer.param_groups:
            edit_groups(group)
    offset = 0
    with torch.no_grad():
        for p, before in zip(params, step.params_before, strict=True):
            p.copy_(before)
            n = p.numel()
            p.grad = step.grad_seen[offset : offset + n].reshape(p.shape).clone()
            offset += n
    optimizer.step()
    return flat(params) - flat(step.params_before)


def _zero_first_moment(state: dict[str, Any]) -> None:
    for entry in state["state"].values():
        for key in ("exp_avg", "momentum_buffer"):
            if entry.get(key) is not None:
                entry[key] = torch.zeros_like(entry[key])


def _no_decay(group: dict[str, Any]) -> None:
    group["weight_decay"] = 0.0


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_adamw(weight_decay=0.5), id="adamw"),
        pytest.param(
            lambda model: torch.optim.AdamW(
                decay_groups(model, weight_decay=0.5, lr_no_decay=1e-3), lr=3e-3
            ),
            id="adamw-groups",
        ),
        pytest.param(_sgd(momentum=0.9, weight_decay=0.1), id="sgd-momentum"),
        pytest.param(_sgd(momentum=0.9, dampening=0.3, weight_decay=0.1), id="sgd-dampening"),
    ],
)
def test_each_part_matches_an_ablation_of_the_real_optimizer(factory: OptimizerFactory) -> None:
    """``wd`` = what switching the decay off removes; ``hist`` = what zeroing ``m`` removes.

    Both updates are linear in the decay and in the first moment at fixed
    gradient (decoupled AdamW; SGD), so the differences of two real steps are
    exactly the two parts, and the rest is the gradient part.
    """
    record = record_steps(factory, N_STEPS)
    batch_b = held_out_batch()
    replayed_any_history = False
    for step in record.steps:
        full = _replay(record, step, factory)
        assert float((full - step.delta).abs().max()) <= 1e-12 * float(step.delta.abs().max())
        without_decay = _replay(record, step, factory, edit_groups=_no_decay)
        without_history = _replay(record, step, factory, edit_state=_zero_first_moment)
        _loss, grad_b = record.loss_and_grad(step.params_before, batch_b)
        terms = _rule(record, step).response(step.grad_update, {"B": grad_b}, step.params_before)[
            "B"
        ]
        wd = float(grad_b @ (full - without_decay))
        hist = float(grad_b @ (full - without_history))
        grad = float(grad_b @ full) - wd - hist
        assert terms.wd == pytest.approx(wd, rel=1e-7, abs=1e-15)
        assert terms.hist == pytest.approx(hist, rel=1e-7, abs=1e-15)
        assert terms.grad == pytest.approx(grad, rel=1e-7, abs=1e-15)
        replayed_any_history |= hist != 0.0
    assert replayed_any_history


def test_the_three_adamw_terms_are_the_documented_formulas() -> None:
    """The split written out once by hand, on the optimizer's own internals.

    Deliberately redundant with the ablation test: this is the formula a reader
    finds in the docs, with ``P_{t+1}`` from the second moment torch holds
    *after* the step and the bias corrections at exponent ``t + 1``.
    """
    lr, beta1, beta2, eps, decay = 3e-3, 0.9, 0.95, 1e-8, 0.5
    factory = _adamw(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=decay)
    record = record_steps(factory, N_STEPS)
    batch_b = held_out_batch()
    for step, following in zip(record.steps[:-1], record.steps[1:], strict=True):
        t = step.index
        second_after = flat(
            [following.state_dict["state"][i]["exp_avg_sq"] for i in range(len(record.names))]
        )
        first_before = (
            flat([step.state_dict["state"][i]["exp_avg"] for i in range(len(record.names))])
            if step.state_dict["state"]
            else torch.zeros_like(second_after)
        )
        p_after = Preconditioner.adamw(second_after, step=t + 1, beta2=beta2, eps=eps).diag
        _loss, grad_b = record.loss_and_grad(step.params_before, batch_b)
        bias1 = 1.0 - beta1 ** (t + 1)
        expected = UpdateResponseTerms(
            grad=-lr
            * (1 - beta1)
            / bias1
            * float(weighted_dot_fp64(grad_b, step.grad_update, p_after)),
            hist=-lr * beta1 / bias1 * float(weighted_dot_fp64(grad_b, first_before, p_after)),
            wd=-lr * decay * float(grad_b @ flat(step.params_before)),
        )
        terms = _rule(record, step).response(step.grad_update, {"B": grad_b}, step.params_before)[
            "B"
        ]
        assert terms.grad == pytest.approx(expected.grad, rel=RTOL)
        assert terms.hist == pytest.approx(expected.hist, rel=RTOL, abs=1e-18)
        assert terms.wd == pytest.approx(expected.wd, rel=RTOL)


def test_delta_loss_p_is_not_the_response_to_the_step() -> None:
    """``-lr * delta_loss_P`` (pre-step ``P_t``, no momentum, no decay) is a different number.

    Not a tolerance on how different — that depends on the run — only that the
    two are not interchangeable on an ordinary AdamW trajectory, which is why
    the new rows exist.
    """
    record = record_steps(_adamw(), 12)
    gaps = []
    for step in record.steps[1:]:
        second = flat([step.state_dict["state"][i]["exp_avg_sq"] for i in range(len(record.names))])
        p_before = Preconditioner.adamw(second, step=step.index, beta2=0.95, eps=1e-8).diag
        hypothetical = -step.lr * float(
            weighted_dot_fp64(step.grad_update, step.grad_update, p_before)
        )
        actual = float(step.grad_update @ step.delta)
        gaps.append(abs(hypothetical / actual - 1.0))
    assert min(gaps) > 0.5


# ------------------------------------------------------------- the rule by hand


def test_a_hand_built_rule_needs_no_bundle_plumbing() -> None:
    """``OptimizerUpdate`` assembled directly from the optimizer's attributes."""
    record = record_steps(_adamw(weight_decay=0.2), 4)
    step = record.steps[-1]
    n = len(record.names)
    rule = OptimizerUpdate(
        family="adam",
        hypers=[AdamHyper(lr=3e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.2)] * n,
        step=step.index,
        first_moment=[step.state_dict["state"][i]["exp_avg"] for i in range(n)],
        second_moment=[step.state_dict["state"][i]["exp_avg_sq"] for i in range(n)],
    )
    predicted = rule.predicted_delta(step.grad_update, step.params_before)
    assert float((predicted - step.delta).abs().max()) <= 1e-10 * float(step.delta.abs().max())


def test_chunking_does_not_change_the_result() -> None:
    record = record_steps(_adamw(), 3)
    step = record.steps[-1]
    rule = _rule(record, step)
    whole = rule.predicted_delta(step.grad_update, step.params_before)
    pieces = torch.empty_like(whole)
    n_items = 0
    for start, stop, d_grad, d_hist, d_wd in rule.iter_parts(
        step.grad_update, step.params_before, chunk=37
    ):
        pieces[start:stop] = d_grad + d_hist + d_wd
        assert stop - start <= 37
        n_items += 1
    assert n_items > len(record.names)
    assert torch.equal(whole, pieces)


def test_fp32_inputs_are_accumulated_in_fp64() -> None:
    """The reductions promote: an fp32 gradient loses nothing beyond its own rounding."""
    record = record_steps(_adamw(), 3)
    step = record.steps[-1]
    rule = _rule(record, step)
    grad32 = step.grad_update.to(torch.float32)
    terms = rule.response(grad32, {"A": grad32}, step.params_before)["A"]
    reference = rule.response(
        grad32.to(torch.float64), {"A": grad32.to(torch.float64)}, step.params_before
    )["A"]
    assert terms == reference


def test_sgd_coefficients() -> None:
    assert SGDHyper(lr=1.0).coefficients(has_buffer=False) == (1.0, 0.0)
    assert SGDHyper(lr=1.0, momentum=0.9).coefficients(has_buffer=False) == (1.0, 0.0)
    assert SGDHyper(lr=1.0, momentum=0.9, dampening=0.25).coefficients(has_buffer=True) == (
        0.75,
        0.9,
    )
    on_grad, on_buffer = SGDHyper(lr=1.0, momentum=0.5, nesterov=True).coefficients(has_buffer=True)
    assert (on_grad, on_buffer) == (1.5, 0.25)


# ------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda m: torch.optim.RAdam(m.parameters(), lr=1e-3), id="RAdam"),
        pytest.param(lambda m: torch.optim.NAdam(m.parameters(), lr=1e-3), id="NAdam"),
        pytest.param(lambda m: torch.optim.Adamax(m.parameters(), lr=1e-3), id="Adamax"),
        pytest.param(lambda m: torch.optim.RMSprop(m.parameters(), lr=1e-3), id="RMSprop"),
        pytest.param(lambda m: torch.optim.Adagrad(m.parameters(), lr=1e-3), id="Adagrad"),
    ],
)
def test_other_optimizers_are_refused(factory: OptimizerFactory) -> None:
    record = record_steps(factory, 2)
    for step in record.steps:  # also before the first update, where every state is empty
        with pytest.raises(BundleSchemaError, match="Adam, AdamW\\s+and SGD"):
            _rule(record, step)


@pytest.mark.parametrize("flag", ["amsgrad", "maximize"])
def test_amsgrad_and_maximize_are_refused(flag: str) -> None:
    record = record_steps(_adamw(**{flag: True}), 2)
    for step in record.steps:
        with pytest.raises(ValueError, match=f"{flag}=True"):
            _rule(record, step)


def test_sgd_maximize_is_refused() -> None:
    record = record_steps(_sgd(maximize=True), 2)
    with pytest.raises(ValueError, match="maximize=True"):
        _rule(record, record.steps[1])


def test_metadata_naming_another_optimizer_is_refused() -> None:
    record = record_steps(_adamw(), 2)
    with pytest.raises(BundleSchemaError, match="do\\s+not belong together"):
        _rule(record, record.steps[1], metadata={"optimizer": "sgd"})
    sgd = record_steps(_sgd(momentum=0.9), 2)
    with pytest.raises(BundleSchemaError, match="do\\s+not belong together"):
        _rule(sgd, sgd.steps[1], metadata={"optimizer_class": "AdamW"})


def test_a_step_counter_that_contradicts_the_directory_is_refused_or_warned() -> None:
    record = record_steps(_adamw(), 3)
    step = record.steps[2]
    with pytest.raises(BundleSchemaError, match="has applied 2 update"):
        _rule(record, step, checkpoint_step=5)
    with pytest.warns(UserWarning, match="used as it is"):
        rule = _rule(record, step, checkpoint_step=5, on_step_mismatch="warn")
    assert rule.step == 2
    # An empty state at a later step is the same contradiction.
    with pytest.raises(BundleSchemaError, match="has applied 0 update"):
        _rule(record, record.steps[0], checkpoint_step=4)


def test_sgd_buffers_must_match_the_directory_step() -> None:
    record = record_steps(_sgd(momentum=0.9), 3)
    with pytest.raises(BundleSchemaError, match="holds momentum buffers"):
        _rule(record, record.steps[2], checkpoint_step=0)
    with pytest.raises(BundleSchemaError, match="holds no momentum buffer"):
        _rule(record, record.steps[0], checkpoint_step=3)
    # Without momentum there is no state to contradict the step.
    plain = record_steps(_sgd(), 3)
    _rule(plain, plain.steps[2], checkpoint_step=17)


def test_a_recorded_learning_rate_that_differs_from_the_saved_one_is_refused() -> None:
    record = record_steps(_adamw(), 2)
    step = record.steps[1]
    _rule(record, step, recorded_lr=3e-3)
    _rule(record, step, recorded_lr=3e-3 * (1 + 1e-9))
    with pytest.raises(BundleSchemaError, match="lr = 0.003 .*lr = 0.0031"):
        _rule(record, step, recorded_lr=3.1e-3)


def test_partial_state_is_refused() -> None:
    record = record_steps(_adamw(), 3)
    step = record.steps[2]
    del step.state_dict["state"][0]
    with pytest.raises(BundleSchemaError, match="for some\\s+parameters but not for all"):
        _rule(record, step)
    sgd = record_steps(_sgd(momentum=0.9), 3)
    sgd.steps[2].state_dict["state"][0]["momentum_buffer"] = None
    with pytest.raises(BundleSchemaError, match="never received one"):
        _rule(sgd, sgd.steps[2])


def test_decoupling_is_read_from_the_group_then_from_the_name_never_guessed() -> None:
    """A group without ``decoupled_weight_decay`` (older torch) and a non-zero decay."""
    record = record_steps(_adamw(weight_decay=0.3), 3)
    step = record.steps[2]
    for group in step.state_dict["param_groups"]:
        del group["decoupled_weight_decay"]
    with pytest.raises(ValueError, match="Adam .* or AdamW"):
        _rule(record, step)
    assert _rule(record, step, metadata={"optimizer": "adamw"}).hypers[0].decoupled_weight_decay
    coupled = _rule(record, step, metadata={"optimizer": "adam"})
    assert not coupled.hypers[0].decoupled_weight_decay
    assert coupled.descriptor == "adam_update"
    # With zero decay the two optimizers coincide and nothing has to be known.
    for group in step.state_dict["param_groups"]:
        group["weight_decay"] = 0.0
    assert _rule(record, step).descriptor == "adamw_update"


def test_descriptors_name_the_update_rule() -> None:
    for factory, descriptor in (
        (_adamw(), "adamw_update"),
        (_adam(), "adam_update"),
        (_sgd(momentum=0.9), "sgd_update"),
    ):
        record = record_steps(factory, 2)
        assert _rule(record, record.steps[1]).descriptor == descriptor


def test_rule_validation() -> None:
    hyper = AdamHyper(lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8)
    zeros = [torch.zeros(3)]
    with pytest.raises(ValueError, match="family"):
        OptimizerUpdate(
            family="rmsprop", hypers=[hyper], step=0, first_moment=None, second_moment=None
        )
    with pytest.raises(ValueError, match="both moments or neither"):
        OptimizerUpdate(
            family="adam", hypers=[hyper], step=1, first_moment=zeros, second_moment=None
        )
    with pytest.raises(ValueError, match="before the first update"):
        OptimizerUpdate(
            family="adam", hypers=[hyper], step=3, first_moment=None, second_moment=None
        )
    with pytest.raises(TypeError, match="AdamHyper"):
        OptimizerUpdate(
            family="adam", hypers=[SGDHyper(lr=0.1)], step=0, first_moment=None, second_moment=None
        )
    with pytest.raises(ValueError, match="clip_grad_norm"):
        OptimizerUpdate(
            family="adam",
            hypers=[hyper],
            step=0,
            first_moment=None,
            second_moment=None,
            clip_grad_norm=0.0,
        )
    rule = OptimizerUpdate(
        family="adam", hypers=[hyper], step=0, first_moment=None, second_moment=None
    )
    with pytest.raises(ValueError, match="flatten to 3"):
        rule.response(torch.zeros(4), {}, [torch.zeros(3)])
    with pytest.raises(ValueError, match="covers 1 parameters"):
        rule.response(torch.zeros(3), {}, [torch.zeros(2), torch.zeros(1)])


def test_a_non_finite_update_is_refused_not_reported() -> None:
    """``eps = 0`` on a coordinate with zero gradient and zero history: ``0 / 0``."""
    rule = OptimizerUpdate(
        family="adam",
        hypers=[AdamHyper(lr=1e-3, beta1=0.9, beta2=0.99, eps=0.0)],
        step=0,
        first_moment=None,
        second_moment=None,
    )
    grad = torch.tensor([1.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="not finite"):
        rule.response(grad, {"A": grad}, [torch.ones(2, dtype=torch.float64)])


# ------------------------------------------------ a parameter without a gradient


def _run_with_an_idle_parameter(
    make_optimizer: Callable[[list[nn.Parameter]], torch.optim.Optimizer],
    *,
    set_to_none: bool,
    n_steps: int = 4,
) -> list[dict[str, Any]]:
    """Two parameters; ``b`` is in the graph at steps 0 and 1 only.

    Each record holds the weights and the ``state_dict()`` before the step, the
    gradient as a flat vector (zeros where torch held ``None``), whether ``b``'s
    gradient was ``None``, and the weights after.
    """
    a = nn.Parameter(torch.linspace(0.5, 1.5, 3, dtype=torch.float64))
    b = nn.Parameter(torch.linspace(-1.0, 1.0, 4, dtype=torch.float64))
    optimizer = make_optimizer([a, b])
    target = torch.arange(1.0, 4.0, dtype=torch.float64)
    records = []
    for t in range(n_steps):
        optimizer.zero_grad(set_to_none=set_to_none)
        loss = ((a - target) ** 2).sum()
        if t < 2:
            loss = loss + (b**3).sum()
        loss.backward()
        records.append(
            {
                "before": [a.detach().clone(), b.detach().clone()],
                "state": copy.deepcopy(optimizer.state_dict()),
                "grad": torch.cat(
                    [a.grad, torch.zeros_like(b) if b.grad is None else b.grad]
                ).clone(),
                "b_has_no_gradient": b.grad is None,
            }
        )
        optimizer.step()
        records[-1]["after"] = [a.detach().clone(), b.detach().clone()]
    return records


def _idle_rule(record: dict[str, Any], step: int) -> OptimizerUpdate:
    layout = ParameterLayout.from_records([["a", [3]], ["b", [4]]])
    return optimizer_update_from_state(
        OptimizerStateView(record["state"], layout), checkpoint_step=step
    )


_IDLE_OPTIMIZERS = [
    pytest.param(lambda ps: torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.5), id="adamw"),
    pytest.param(lambda ps: torch.optim.SGD(ps, lr=1e-2, momentum=0.9, weight_decay=0.5), id="sgd"),
]


@pytest.mark.parametrize("make_optimizer", _IDLE_OPTIMIZERS)
def test_torch_skips_a_parameter_whose_gradient_is_none(make_optimizer: Any) -> None:
    """No momentum, no weight decay: the parameter is not touched — and the rule,
    which sees a zero gradient, would move it. Told which parameter got no
    gradient, the rule refuses instead."""
    record = _run_with_an_idle_parameter(make_optimizer, set_to_none=True)[2]
    assert record["b_has_no_gradient"]
    assert torch.equal(record["after"][1], record["before"][1])  # torch: untouched
    assert not torch.equal(record["after"][0], record["before"][0])

    rule = _idle_rule(record, 2)
    predicted = rule.predicted_delta(record["grad"], record["before"])
    assert float(predicted[3:].abs().max()) > 1e-3  # momentum and decay, never applied
    with pytest.raises(ValueError, match=r"parameter 1 \(b\) receives no gradient"):
        rule.predicted_delta(record["grad"], record["before"], no_gradient={1: "b"})
    with pytest.raises(ValueError, match="receives no gradient"):
        rule.response(record["grad"], {"A": record["grad"]}, record["before"], no_gradient={1: "b"})


@pytest.mark.parametrize("make_optimizer", _IDLE_OPTIMIZERS)
def test_a_zero_gradient_is_not_a_missing_one(make_optimizer: Any) -> None:
    """``zero_grad(set_to_none=False)`` leaves zeros in ``.grad``: torch then does
    apply momentum and decay, and the rule reproduces that step exactly."""
    record = _run_with_an_idle_parameter(make_optimizer, set_to_none=False)[2]
    assert not record["b_has_no_gradient"]
    actual = flat(record["after"]) - flat(record["before"])
    assert float(actual[3:].abs().max()) > 1e-3
    predicted = _idle_rule(record, 2).predicted_delta(record["grad"], record["before"])
    assert float((predicted - actual).abs().max()) <= 1e-10 * float(actual.abs().max())


def test_adam_step_counters_diverge_after_a_skipped_step() -> None:
    """One checkpoint later the skip is visible in the state, and refused."""
    records = _run_with_an_idle_parameter(
        lambda ps: torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.5), set_to_none=True
    )
    with pytest.raises(BundleSchemaError, match="different 'step' counters"):
        _idle_rule(records[3], 3)


def test_no_gradient_is_exact_when_nothing_would_move_the_parameter() -> None:
    """Plain SGD, decay on ``a`` only: skipped or not, ``b`` does not move."""
    record = _run_with_an_idle_parameter(
        lambda ps: torch.optim.SGD(
            [{"params": [ps[0]], "weight_decay": 0.5}, {"params": [ps[1]], "weight_decay": 0.0}],
            lr=1e-2,
        ),
        set_to_none=True,
    )[2]
    assert record["b_has_no_gradient"]
    actual = flat(record["after"]) - flat(record["before"])
    predicted = _idle_rule(record, 2).predicted_delta(
        record["grad"], record["before"], no_gradient={1: "b"}
    )
    assert float(actual[3:].abs().max()) == 0.0 and float(predicted[3:].abs().max()) == 0.0
    assert float((predicted - actual).abs().max()) <= 1e-10 * float(actual.abs().max())
