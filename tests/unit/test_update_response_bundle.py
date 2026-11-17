"""The update-response rows through the bundle path: train, write a bundle, analyze it.

``test_update_response_vs_torch.py`` pins the rule itself in fp64. This file
pins everything between a bundle on disk and a row in the result: which state is
read, which learning rate, which batch is taken for the update batch, what the
analyzer does with ``theta_{t+1}``, and what is refused.

The reference is never the loader. :func:`tests.fixtures.update_response_toys.write_bundle`
trains the tiny transformer with a real ``torch.optim`` optimizer in fp32, writes
checkpoints in the bundle schema, and keeps the run in memory — the weights
before and after every step. A row is held against ``g . (theta_{t+1} - theta_t)``
of *that* record, with ``g`` from plain autograd.

Tolerances: the bundle path is fp32 (weights, optimizer state, and the
analyzer's gradient vector), so agreement with the fp32 run is limited by fp32
round-off in a dot product over ~20k coordinates, measured at ``1e-8 .. 5e-7``
relative on these bundles. The assertions use ``1e-5``: twenty times the worst
case seen, and three orders below the smallest modelling error they are there to
catch.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn

from tests.fixtures.update_response_toys import (
    TrainingRecord,
    decay_groups,
    edit_json,
    held_out_batch,
    write_bundle,
)
from tests.utils.bundle_fixture import DEFAULT_MODEL_FACTORY, lm_loss
from vatis.analyzer import (
    ALL_OBSERVABLES,
    P_OBSERVABLES,
    REALIZED_CHANGE_OBSERVABLES,
    UPDATE_RESPONSE_OBSERVABLES,
    UPDATE_STEP_OBSERVABLES,
    Analyzer,
    analyze,
)
from vatis.core.update_response import AdamHyper, OptimizerUpdate
from vatis.models.hf import ModelBundle, causal_lm_loss, causal_lm_valid_mask
from vatis.models.local_bundle import (
    BundleSchemaError,
    LocalBundleSource,
    load_local_bundle,
    plain_lm_forward,
)
from vatis.sinks.base import ResultRow

N_STEPS = 9
# 0..3 consecutive (so 0, 1, 2 have a stored successor), 6 isolated, 8 and 9 again
# consecutive with 9 the final weights.
BUNDLE_STEPS = (0, 1, 2, 3, 6, 8, 9)
WITH_SUCCESSOR = (0, 1, 2, 8)
RTOL = 1e-5

OptimizerFactory = Callable[[nn.Module], torch.optim.Optimizer]


def _adamw(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)


def _write(
    tmp_path: Path, make_optimizer: OptimizerFactory = _adamw, **kwargs: Any
) -> tuple[Path, TrainingRecord]:
    run_dir = tmp_path / "run"
    kwargs.setdefault("n_steps", N_STEPS)
    kwargs.setdefault("bundle_steps", BUNDLE_STEPS)
    record = write_bundle(run_dir, make_optimizer, **kwargs)
    return run_dir, record


def _source(run_dir: Path, **kwargs: Any) -> LocalBundleSource:
    return load_local_bundle(run_dir, DEFAULT_MODEL_FACTORY, **kwargs)


def _run(
    source: LocalBundleSource,
    *,
    update_response: Any = "default",
    preconditioner: Any = None,
    eval_batches: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[int, list[ResultRow]]:
    """Analyze the bundle; rows per step."""
    if update_response == "default":
        update_response = source.update_response()
    if eval_batches is None:
        eval_batches = {"train": source.batch_a_spec(), "val": held_out_batch()}
    kwargs.setdefault("micro_batch_size", 2)
    analyzer = Analyzer(
        eval_batches=eval_batches,
        update_response=update_response,
        preconditioner=preconditioner,
        chi_net_method="hutchinson",
        n_hutchinson=2,
        dtype="fp32",
        **kwargs,
    )
    results = analyzer.run(source.model_specs())
    return {step: result.rows for step, result in zip(source.steps, results, strict=True)}


def _value(rows: list[ResultRow], batch_b: str, observable: str, batch_a: str = "train") -> float:
    found = [
        r.value
        for r in rows
        if (r.batch_a, r.batch_b, r.observable) == (batch_a, batch_b, observable)
    ]
    assert len(found) == 1, f"{len(found)} rows for ({batch_a}, {batch_b}, {observable})"
    return found[0]


def _has(rows: list[ResultRow], observable: str) -> bool:
    return any(r.observable == observable for r in rows)


def _reference(record: TrainingRecord, step: int, batch: str) -> float:
    """``g . (theta_{t+1} - theta_t)`` from the recorded run and plain autograd."""
    recorded = record.steps[step]
    if batch == "train":
        grad = recorded.grad_update
    else:
        _loss, grad = record.loss_and_grad(recorded.params_before, held_out_batch())
    return float(grad @ recorded.delta)


def _loss_fp64(record: TrainingRecord, params: list[torch.Tensor], batch: dict[str, Any]) -> float:
    model = record.model_at(params)
    with torch.no_grad():
        return float(lm_loss(model(batch["input_ids"]).to(torch.float64), batch))


def _assert_matches_the_run(
    record: TrainingRecord,
    rows_by_step: dict[int, list[ResultRow]],
    *,
    descriptor: str = "adamw_update",
) -> None:
    for step, rows in rows_by_step.items():
        for batch in ("train", "val"):
            reference = _reference(record, step, batch)
            total = _value(rows, batch, "loss_change_first_order")
            assert total == pytest.approx(reference, rel=RTOL), (
                f"step {step}, batch {batch}: update_response = {total!r}, the run did "
                f"g . dtheta = {reference!r}"
            )
            parts = sum(
                _value(rows, batch, f"loss_change_first_order_{part}")
                for part in ("grad", "hist", "wd")
            )
            assert total == pytest.approx(parts, rel=1e-12, abs=1e-18)
        for row in rows:
            if row.observable in UPDATE_RESPONSE_OBSERVABLES:
                assert row.preconditioner == descriptor


# ------------------------------------------------------------------ the rows


@pytest.fixture(scope="module")
def adamw_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, TrainingRecord]:
    return _write(tmp_path_factory.mktemp("adamw"))


def test_update_response_is_the_first_order_change_of_the_step_the_run_took(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    run_dir, record = adamw_run
    rows_by_step = _run(_source(run_dir))
    assert tuple(rows_by_step) == BUNDLE_STEPS
    _assert_matches_the_run(record, rows_by_step)
    # The history part is absent at the first update and present afterwards; the
    # decay part is there from the start.
    assert _value(rows_by_step[0], "val", "loss_change_first_order_hist") == 0.0
    assert _value(rows_by_step[1], "val", "loss_change_first_order_hist") != 0.0
    assert _value(rows_by_step[0], "val", "loss_change_first_order_wd") != 0.0


def test_rows_are_keyed_by_update_batch_and_evaluated_batch(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    run_dir, _record = adamw_run
    rows_by_step = _run(_source(run_dir))
    for step, rows in rows_by_step.items():
        new = [
            r
            for r in rows
            if r.observable
            in UPDATE_RESPONSE_OBSERVABLES + UPDATE_STEP_OBSERVABLES + REALIZED_CHANGE_OBSERVABLES
        ]
        expected = set(UPDATE_RESPONSE_OBSERVABLES)
        if step in WITH_SUCCESSOR:
            expected |= set(REALIZED_CHANGE_OBSERVABLES)
        # The step's learning rate and clip factor sit on the update batch's own pair.
        step_rows = {"update_lr", "update_clip_factor"}
        for batch in ("train", "val"):
            got = {r.observable for r in new if (r.batch_a, r.batch_b) == ("train", batch)}
            assert got == expected | (step_rows if batch == "train" else set()), (
                f"step {step}, pair (train, {batch})"
            )
        assert {(r.batch_a, r.batch_b) for r in new} == {("train", "train"), ("train", "val")}
        for row in new:
            # The run's setting, as on every row; no probe stream, no timing.
            assert row.n_hutchinson == 2 and row.hutchinson_seed is None
            assert row.wallclock_s is None
            assert row.preconditioner == (
                "adamw_update"
                if row.observable in UPDATE_RESPONSE_OBSERVABLES + UPDATE_STEP_OBSERVABLES
                else "none"
            )
            assert (
                row.n_valid_a == 4 * 15
                and row.n_valid_b == (4 if row.batch_b == "train" else 8) * 15
            )
        # One row per (pair, observable): the long format pivots without duplicates.
        keys = [(r.batch_a, r.batch_b, r.observable) for r in rows]
        assert len(keys) == len(set(keys))


def test_existing_rows_are_unchanged_bit_for_bit(adamw_run: tuple[Path, TrainingRecord]) -> None:
    run_dir, _record = adamw_run
    source = _source(run_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the step-0 P skip
        without = _run(source, update_response=None, preconditioner=source.preconditioner())
        with_response = _run(source, preconditioner=source.preconditioner())
    for step in BUNDLE_STEPS:
        old = [r for r in without[step]]
        kept = [
            r
            for r in with_response[step]
            if r.observable
            not in UPDATE_RESPONSE_OBSERVABLES
            + UPDATE_STEP_OBSERVABLES
            + REALIZED_CHANGE_OBSERVABLES
        ]
        assert len(old) == len(kept) > 0
        for a, b in zip(old, kept, strict=True):
            assert (a.batch_a, a.batch_b, a.observable, a.preconditioner) == (
                b.batch_a,
                b.batch_b,
                b.observable,
                b.preconditioner,
            )
            assert a.value == b.value
            assert a.hutchinson_seed == b.hutchinson_seed


def test_step_zero_has_response_rows_although_it_has_no_p_rows(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    run_dir, _record = adamw_run
    source = _source(run_dir, steps=[0, 1])
    with pytest.warns(UserWarning, match="SKIPPED"):
        rows_by_step = _run(source, preconditioner=source.preconditioner())
    assert not any(_has(rows_by_step[0], name) for name in P_OBSERVABLES)
    assert all(_has(rows_by_step[0], name) for name in UPDATE_RESPONSE_OBSERVABLES)
    assert all(_has(rows_by_step[1], name) for name in P_OBSERVABLES)


# ----------------------------------------------------------- the realized change


def test_realized_change_and_residual_where_the_bundle_holds_the_next_step(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    run_dir, record = adamw_run
    rows_by_step = _run(_source(run_dir))
    batches = {"val": held_out_batch()}
    for step in BUNDLE_STEPS:
        rows = rows_by_step[step]
        if step not in WITH_SUCCESSOR:
            assert not any(_has(rows, name) for name in REALIZED_CHANGE_OBSERVABLES)
            continue
        recorded = record.steps[step]
        for batch in ("train", "val"):
            from_weights = _value(rows, batch, "loss_change_first_order_weights")
            assert from_weights == pytest.approx(_reference(record, step, batch), rel=RTOL)
            # ... and the rebuilt update agrees with the stored one.
            assert _value(rows, batch, "loss_change_first_order") == pytest.approx(
                from_weights, rel=RTOL
            )
            change = _value(rows, batch, "loss_change")
            residual = _value(rows, batch, "loss_change_residual")
            assert residual == change - from_weights
        reference_change = _loss_fp64(record, recorded.params_after, batches["val"]) - _loss_fp64(
            record, recorded.params_before, batches["val"]
        )
        assert _value(rows, "val", "loss_change") == pytest.approx(
            reference_change, rel=1e-4, abs=1e-7
        )
        # A descent step on this toy: the loss falls, by less than first order says.
        assert _value(rows, "val", "loss_change") < 0.0
        assert (
            0.0
            < _value(rows, "val", "loss_change_residual")
            < abs(_value(rows, "val", "loss_change_first_order_weights"))
        )


def test_the_successor_is_found_on_disk_even_when_it_is_not_selected(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    run_dir, record = adamw_run
    rows_by_step = _run(_source(run_dir, steps=[1, 6]))
    assert set(rows_by_step) == {1, 6}
    assert _value(rows_by_step[1], "val", "loss_change_first_order_weights") == pytest.approx(
        _reference(record, 1, "val"), rel=RTOL
    )
    assert not _has(rows_by_step[6], "loss_change")


def test_realized_false_switches_the_realized_rows_off(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    run_dir, _record = adamw_run
    source = _source(run_dir, steps=[1])
    rows = _run(source, update_response=source.update_response(realized=False))[1]
    assert not any(_has(rows, name) for name in REALIZED_CHANGE_OBSERVABLES)
    assert all(_has(rows, name) for name in UPDATE_RESPONSE_OBSERVABLES)


@pytest.mark.parametrize("weight_dtype", ["bf16"])
def test_rounded_weight_files_give_no_realized_rows(tmp_path: Path, weight_dtype: str) -> None:
    """bf16 weight files: the difference of two of them is rounding, so it is not reported."""
    run_dir, record = _write(tmp_path, weight_dtype=weight_dtype)
    source = _source(run_dir)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows_by_step = _run(source)
    skipped = [str(w.message) for w in caught if "SKIPPED" in str(w.message)]
    assert len(skipped) == len(WITH_SUCCESSOR)
    assert all("BF16" in message and "fp32 weight files" in message for message in skipped)
    for step, rows in rows_by_step.items():
        assert not any(_has(rows, name) for name in REALIZED_CHANGE_OBSERVABLES)
        assert all(_has(rows, name) for name in UPDATE_RESPONSE_OBSERVABLES), step

    # Why: what such a difference would have been. Element by element, the
    # stored weight change of one step carries the rounding of both files (14%
    # of the step on this toy, whose learning rate is large; most of the step
    # on a real bundle early in training, see docs/SPEC.md) ...
    step = record.steps[1]
    stored = torch.cat(
        [
            (after.to(torch.bfloat16).double() - before.to(torch.bfloat16).double()).reshape(-1)
            for after, before in zip(step.params_after, step.params_before, strict=True)
        ]
    )
    assert float((stored - step.delta).norm() / step.delta.norm()) > 0.05
    # ... while the rows that ARE reported only use the rounded weights as the
    # point where the gradient is taken, which costs percent, not everything.
    for batch in ("train", "val"):
        assert _value(rows_by_step[1], batch, "loss_change_first_order") == pytest.approx(
            _reference(record, 1, batch), rel=0.1
        )


def test_one_rounded_endpoint_is_enough_to_refuse(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    run_dir, _record = _write(tmp_path)
    path = run_dir / "checkpoints" / "step_00000002" / "model.safetensors"
    save_file({k: v.to(torch.bfloat16) for k, v in load_file(str(path)).items()}, str(path))
    source = _source(run_dir, steps=[0, 1, 2])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows_by_step = _run(source)
    assert _has(rows_by_step[0], "loss_change")  # 0 -> 1: both fp32
    assert not _has(rows_by_step[1], "loss_change")  # 1 -> 2: successor rounded
    assert not _has(rows_by_step[2], "loss_change")  # 2 -> 3: this one rounded
    assert sum("SKIPPED" in str(w.message) for w in caught) == 2


# --------------------------------------------------------------- the update batch


def test_the_update_batch_is_found_by_its_tensors_not_by_its_name(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    run_dir, record = adamw_run
    source = _source(run_dir, steps=[2])
    rows = _run(
        source,
        eval_batches={"held_out": held_out_batch(), "whatever": source.batch_a_spec()},
    )[2]
    assert _value(rows, "held_out", "loss_change_first_order", batch_a="whatever") == pytest.approx(
        _reference(record, 2, "val"), rel=RTOL
    )
    assert _value(rows, "whatever", "loss_change_first_order", batch_a="whatever") == pytest.approx(
        _reference(record, 2, "train"), rel=RTOL
    )


def test_missing_update_batch_is_refused(adamw_run: tuple[Path, TrainingRecord]) -> None:
    run_dir, _record = adamw_run
    source = _source(run_dir, steps=[2])
    with pytest.raises(ValueError, match="none of the eval batches .* is the batch this step"):
        _run(source, eval_batches={"val": held_out_batch()})
    # The update batch of ANOTHER step is not this step's update batch.
    other = source.checkpoint_for(2).batch_a
    source3 = _source(run_dir, steps=[3])
    with pytest.raises(ValueError, match="none of the eval batches"):
        _run(source3, eval_batches={"train": other, "val": held_out_batch()})


# ------------------------------------------------- what the state cannot show


def test_clipping_recorded_with_its_algorithm_is_applied(tmp_path: Path) -> None:
    run_dir, record = _write(tmp_path, clip_grad_norm=0.5)
    clipped = [float(s.grad_seen.norm() / s.grad_update.norm()) < 0.999 for s in record.steps]
    assert any(clipped), "the fixture's clip never engaged"
    rows_by_step = _run(_source(run_dir))
    _assert_matches_the_run(record, rows_by_step)


def test_clipping_must_be_stated_never_assumed(tmp_path: Path) -> None:
    run_dir, record = _write(tmp_path, clip_grad_norm=0.5)
    metadata = run_dir / "metadata.json"

    # Recorded value, no algorithm: not guessed.
    edit_json(metadata, gradient_clip_algorithm=...)
    with pytest.raises(BundleSchemaError, match="does not say HOW"):
        _source(run_dir).update_response()
    stated = _source(run_dir)
    _assert_matches_the_run(
        record, _run(stated, update_response=stated.update_response(clip_grad_norm=0.5))
    )
    with pytest.raises(ValueError, match="contradicts the recorded"):
        _source(run_dir).update_response(clip_grad_norm=1.0)
    with pytest.raises(ValueError, match="says the run did not clip"):
        _source(run_dir).update_response(clip_grad_norm=None)

    # Clipping by value is not implemented.
    edit_json(metadata, gradient_clip_algorithm="value")
    with pytest.raises(BundleSchemaError, match="by value"):
        _source(run_dir).update_response()
    with pytest.raises(ValueError, match="'value'"):
        _source(run_dir).update_response(clip_grad_norm=0.5)

    # Nothing recorded at all: refused, until the caller says which it was.
    edit_json(metadata, gradient_clip_algorithm=..., gradient_clip_val=...)
    with pytest.raises(BundleSchemaError, match="does not say\\s+whether the gradient was clipped"):
        _source(run_dir).update_response()
    stated = _source(run_dir)
    _assert_matches_the_run(
        record, _run(stated, update_response=stated.update_response(clip_grad_norm=0.5))
    )

    # Recorded as "not clipped": a stated clip contradicts it.
    edit_json(metadata, gradient_clip_val=None)
    with pytest.raises(ValueError, match="says the run clipped"):
        _source(run_dir).update_response(clip_grad_norm=0.5)

    for bad in ("norm", True, -1.0, float("inf")):
        with pytest.raises(ValueError):
            _source(run_dir).update_response(clip_grad_norm=bad)  # type: ignore[arg-type]
    for bad_value in ("1.0", -1.0, True):
        edit_json(metadata, gradient_clip_val=bad_value)
        with pytest.raises(BundleSchemaError, match="gradient_clip_val"):
            _source(run_dir).update_response()
    edit_json(metadata, gradient_clip_val=0.5, gradient_clip_algorithm="adaptive")
    with pytest.raises(BundleSchemaError, match="gradient_clip_algorithm"):
        _source(run_dir).update_response()


def test_a_clip_the_bundle_denies_shows_up_against_the_stored_weights(tmp_path: Path) -> None:
    """Metadata that wrongly says "not clipped": the rebuilt update is wrong, and the
    stored weight difference — which needs no formula — says so."""
    run_dir, record = _write(tmp_path, clip_grad_norm=0.5)
    edit_json(run_dir / "metadata.json", gradient_clip_algorithm=..., gradient_clip_val=None)
    rows = _run(_source(run_dir, steps=[1]))[1]
    from_weights = _value(rows, "val", "loss_change_first_order_weights")
    assert from_weights == pytest.approx(_reference(record, 1, "val"), rel=RTOL)
    assert abs(_value(rows, "val", "loss_change_first_order") / from_weights - 1.0) > 1e-2


def test_zero_clip_value_means_not_clipped(tmp_path: Path) -> None:
    run_dir, record = _write(tmp_path, metadata_extra={"gradient_clip_val": 0})
    _assert_matches_the_run(record, _run(_source(run_dir, steps=[0, 1, 6])))


def test_scheduled_learning_rate_through_the_bundle(tmp_path: Path) -> None:
    run_dir, record = _write(
        tmp_path,
        make_scheduler=lambda opt: torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.05, end_factor=1.0, total_iters=6
        ),
    )
    assert len({record.steps[s].lr for s in BUNDLE_STEPS}) >= 5
    _assert_matches_the_run(record, _run(_source(run_dir)))

    # meta.json and the saved state must name the same rate.
    edit_json(run_dir / "checkpoints" / "step_00000002" / "meta.json", lr=3e-3)
    with pytest.raises(BundleSchemaError, match="meta.json records lr"):
        _run(_source(run_dir, steps=[2]))
    # Without a recorded rate there is nothing to cross-check; the saved one is used.
    edit_json(run_dir / "checkpoints" / "step_00000002" / "meta.json", lr=...)
    _assert_matches_the_run(record, _run(_source(run_dir, steps=[2])))


def test_step_mismatch_is_refused_or_warned(tmp_path: Path) -> None:
    run_dir, record = _write(tmp_path, bundle_steps=(0, 1, 2, 3))
    checkpoints = run_dir / "checkpoints"
    (checkpoints / "step_00000003").rename(checkpoints / "step_00000005")
    edit_json(checkpoints / "step_00000005" / "meta.json", step=5)
    with pytest.raises(BundleSchemaError, match="has applied 3 update"):
        _run(_source(run_dir, steps=[5]))
    source = _source(run_dir, steps=[5])
    with pytest.warns(UserWarning, match="used as it is"):
        rows = _run(source, update_response=source.update_response(on_step_mismatch="warn"))[5]
    assert _value(rows, "val", "loss_change_first_order") == pytest.approx(
        _reference(record, 3, "val"), rel=RTOL
    )
    with pytest.raises(ValueError, match="on_step_mismatch"):
        source.update_response(on_step_mismatch="ignore")


# ------------------------------------------------------------ other update rules


def test_no_decay_groups_with_their_own_learning_rate(tmp_path: Path) -> None:
    """Per-group weight decay AND learning rate: exact here, though the P source refuses."""
    run_dir, record = _write(
        tmp_path,
        lambda model: torch.optim.AdamW(
            decay_groups(model, weight_decay=0.5, lr_no_decay=1e-3), lr=3e-3, betas=(0.9, 0.95)
        ),
    )
    # param_names.json follows the optimizer's order (group after group), which
    # here is not named_parameters() order.
    assert record.names != [n for n in sorted(record.names)]
    source = _source(run_dir)
    _assert_matches_the_run(record, _run(source))
    with pytest.raises(ValueError, match="different learning rates"):
        _run(source, preconditioner=source.preconditioner())


def test_sgd_with_momentum_through_the_bundle(tmp_path: Path) -> None:
    run_dir, record = _write(
        tmp_path,
        lambda model: torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01),
        optimizer_name="sgd",
    )
    source = _source(run_dir)
    rows_by_step = _run(
        source, preconditioner=source.preconditioner(), cross_pairs=[("train", "val")]
    )
    _assert_matches_the_run(record, rows_by_step, descriptor="sgd_update")
    assert _value(rows_by_step[0], "val", "loss_change_first_order_hist") == 0.0
    assert _value(rows_by_step[3], "val", "loss_change_first_order_hist") != 0.0
    # SGD's gradient part is -lr * delta_loss (dampening 0), next to the cross row.
    for step in (0, 3):
        rows = rows_by_step[step]
        assert _value(rows, "val", "loss_change_first_order_grad") == pytest.approx(
            -0.05 * _value(rows, "val", "delta_loss"), rel=1e-9
        )


def test_adam_with_coupled_decay_through_the_bundle(tmp_path: Path) -> None:
    run_dir, record = _write(
        tmp_path,
        lambda model: torch.optim.Adam(
            model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1
        ),
        optimizer_name="adam",
    )
    _assert_matches_the_run(
        record, _run(_source(run_dir, steps=[0, 1, 6])), descriptor="adam_update"
    )


def test_pad_masked_training_loss(tmp_path: Path) -> None:
    """A run whose loss ignored a target id: the update batch is scored the same way."""
    run_dir, record = _write(tmp_path, ignore_token=7)
    source = _source(run_dir, steps=[0, 2, 6])
    assert source.loss_ignore_index == 7
    rows_by_step = _run(source)
    counted = []
    for step, rows in rows_by_step.items():
        reference = float(record.steps[step].grad_update @ record.steps[step].delta)
        assert _value(rows, "train", "loss_change_first_order") == pytest.approx(
            reference, rel=RTOL
        )
        counted.append(next(r for r in rows if r.observable == "loss_change_first_order").n_valid_a)
    # The convention really removed targets (a batch without token 7 keeps all 60).
    assert min(counted) < 60


def test_realized_change_is_a_token_weighted_mean_over_micro_batches(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    """Micro-batches with different numbers of counted targets.

    The loss of the batch is the mean over its counted targets, so the
    micro-batch means have to be weighted by their counts. With equal counts
    (every other batch in this file) a plain average of the micro-batch means
    gives the same number.
    """
    run_dir, record = adamw_run
    batch = held_out_batch()
    labels = batch["labels"].clone()
    labels[0, 3:] = -100  # 3 counted targets in row 0, 15 in the others
    labels[5, 9:] = -100
    uneven = {"input_ids": batch["input_ids"], "labels": labels}
    source = _source(run_dir, steps=[1])
    rows = _run(
        source,
        eval_batches={"train": source.batch_a_spec(), "uneven": uneven},
        micro_batch_size=1,
    )[1]
    recorded = record.steps[1]
    reference = _loss_fp64(record, recorded.params_after, uneven) - _loss_fp64(
        record, recorded.params_before, uneven
    )
    assert _value(rows, "uneven", "loss_change") == pytest.approx(reference, rel=1e-4, abs=1e-7)
    _loss, grad = record.loss_and_grad(recorded.params_before, uneven)
    assert _value(rows, "uneven", "loss_change_first_order") == pytest.approx(
        float(grad @ recorded.delta), rel=RTOL
    )


def test_other_optimizers_are_refused_through_the_bundle(tmp_path: Path) -> None:
    run_dir, _record = _write(
        tmp_path,
        lambda model: torch.optim.RAdam(model.parameters(), lr=1e-3),
        optimizer_name=None,
    )
    with pytest.raises(BundleSchemaError, match="Adam, AdamW\\s+and SGD"):
        _run(_source(run_dir, steps=[1]))


# ------------------------------------------------------------- analyzer plumbing


def test_response_observables_need_an_update_response() -> None:
    with pytest.raises(ValueError, match="no update_response was given"):
        Analyzer(eval_batches={"a": held_out_batch()}, observables=["loss_change_first_order"])
    with pytest.raises(TypeError, match="update_response"):
        Analyzer(eval_batches={"a": held_out_batch()}, update_response="adamw")  # type: ignore[arg-type]


def test_observables_can_be_selected(adamw_run: tuple[Path, TrainingRecord]) -> None:
    run_dir, _record = adamw_run
    source = _source(run_dir, steps=[1])
    rows = _run(source, observables=["delta_loss", "loss_change_first_order", "loss_change"])[1]
    assert {r.observable for r in rows} == {"delta_loss", "loss_change_first_order", "loss_change"}


def test_default_observables_without_an_update_response_are_unchanged(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    run_dir, _record = adamw_run
    rows = _run(_source(run_dir, steps=[1]), update_response=None)[1]
    assert {r.observable for r in rows} == set(ALL_OBSERVABLES)


def test_factory_returning_none_skips_the_rows(adamw_run: tuple[Path, TrainingRecord]) -> None:
    run_dir, _record = adamw_run
    source = _source(run_dir, steps=[0, 1])
    build = source.update_response()
    rows_by_step = _run(
        source,
        update_response=lambda ckpt_id: None if ckpt_id.endswith("step0") else build(ckpt_id),
    )
    assert not _has(rows_by_step[0], "loss_change_first_order")
    assert _has(rows_by_step[1], "loss_change_first_order")
    with pytest.raises(TypeError, match="expected an OptimizerUpdate or None"):
        _run(source, update_response=lambda ckpt_id: "adamw")


def test_rows_reach_the_parquet_sink(
    adamw_run: tuple[Path, TrainingRecord], tmp_path: Path
) -> None:
    run_dir, record = adamw_run
    source = _source(run_dir, steps=[1, 2])
    out = tmp_path / "results.parquet"
    _run(source, sink=str(out))
    table = pq.read_table(out).to_pylist()
    picked = {
        (row["revision"], row["batch_b"]): row["value"]
        for row in table
        if row["observable"] == "loss_change_first_order"
    }
    assert set(picked) == {
        ("step1", "train"),
        ("step1", "val"),
        ("step2", "train"),
        ("step2", "val"),
    }
    assert picked[("step2", "val")] == pytest.approx(_reference(record, 2, "val"), rel=RTOL)
    descriptors = {row["observable"]: row["preconditioner"] for row in table}
    assert descriptors["loss_change_first_order_hist"] == "adamw_update"
    assert descriptors["loss_change"] == "none"
    assert descriptors["delta_loss"] == "none"


def test_micro_batch_size_does_not_change_the_rows(adamw_run: tuple[Path, TrainingRecord]) -> None:
    run_dir, _record = adamw_run
    by_micro = {
        micro: _run(_source(run_dir, steps=[1]), micro_batch_size=micro)[1] for micro in (1, 4)
    }
    for observable in UPDATE_RESPONSE_OBSERVABLES + REALIZED_CHANGE_OBSERVABLES:
        a = _value(by_micro[1], "val", observable)
        b = _value(by_micro[4], "val", observable)
        assert a == pytest.approx(b, rel=1e-4, abs=1e-8), observable


def test_an_explicit_rule_works_on_any_model_bundle() -> None:
    """No bundle: a hand-built rule and a ``ModelBundle`` through ``analyze()``."""
    from tests.fixtures.update_response_toys import make_model

    torch.manual_seed(0)
    model = make_model(dtype=torch.float32).eval()
    params = list(model.parameters())
    batch = held_out_batch(seed=905, n_rows=4)
    other = held_out_batch(seed=906, n_rows=4)
    lr = 1e-2
    rule = OptimizerUpdate(
        family="adam",
        hypers=[AdamHyper(lr=lr, beta1=0.9, beta2=0.99, eps=1e-8)] * len(params),
        step=0,
        first_moment=None,
        second_moment=None,
        descriptor="adamw_update",
        update_batch=batch,
    )
    results = analyze(
        model=ModelBundle(
            model=model,
            params=params,
            forward_fn=plain_lm_forward,
            loss_fn=causal_lm_loss,
            valid_mask_fn=causal_lm_valid_mask,
            identifier="toy",
        ),
        eval_batches={"A": batch, "B": other},
        update_response=lambda ckpt_id: rule,
        chi_net_method="hutchinson",
        n_hutchinson=2,
        dtype="fp32",
    )
    rows = results[0].rows
    # First Adam step from an empty state: -lr * g / (|g| + eps), i.e. -lr * ||g||_1
    # on the update batch itself (up to eps).
    loss = lm_loss(model(batch["input_ids"]), batch)
    grads = torch.autograd.grad(loss, params)
    l1 = float(sum(g.double().abs().sum() for g in grads))
    assert _value(rows, "A", "loss_change_first_order", batch_a="A") == pytest.approx(
        -lr * l1, rel=1e-4
    )
    assert _value(rows, "A", "loss_change_first_order_hist", batch_a="A") == 0.0
    assert not _has(rows, "loss_change")

    # A rule that does not say which batch it consumed cannot be placed.
    import dataclasses

    anonymous = dataclasses.replace(rule, update_batch=None)
    with pytest.raises(ValueError, match="does not carry the batch"):
        analyze(
            model=ModelBundle(
                model=model,
                params=params,
                forward_fn=plain_lm_forward,
                loss_fn=causal_lm_loss,
                valid_mask_fn=causal_lm_valid_mask,
                identifier="toy",
            ),
            eval_batches={"A": batch},
            update_response=lambda ckpt_id: anonymous,
            chi_net_method="hutchinson",
            n_hutchinson=2,
            dtype="fp32",
        )


# ------------------------------------------- rounded in memory, not only on disk


def test_a_source_that_rounds_the_weights_on_load_gives_no_realized_rows(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    """fp32 weight files opened with ``dtype="bf16"``: the two thetas the analyzer
    would compare are rounded in memory, exactly as if the files were bf16."""
    run_dir, record = adamw_run
    source = _source(run_dir, dtype="bf16", steps=[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = _run(source)[1]
    skipped = [str(w.message) for w in caught if "SKIPPED" in str(w.message)]
    assert len(skipped) == 1
    assert 'dtype="bf16"' in skipped[0] and "weight files are stored" not in skipped[0]
    assert not any(_has(rows, name) for name in REALIZED_CHANGE_OBSERVABLES)
    # The formula rows are reported, from theta_t and a gradient at bf16 precision.
    for batch in ("train", "val"):
        assert _value(rows, batch, "loss_change_first_order") == pytest.approx(
            _reference(record, 1, batch), rel=0.1
        )

    # fp16 likewise (the factory alone: no forward pass is needed to decide).
    half = _source(run_dir, dtype="fp16", steps=[1])
    with pytest.warns(UserWarning, match='dtype="fp16"'):
        rule = half.update_response()(half.checkpoints[0].identifier)
    assert rule is not None and rule.next_weights is None
    # Both reasons are named when both apply; fp32 loaded as fp32 stays silent.
    exact = _source(run_dir, steps=[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rule = exact.update_response()(exact.checkpoints[0].identifier)
    assert rule is not None and rule.next_weights is not None


def test_rounded_files_and_a_rounding_loader_are_both_named(tmp_path: Path) -> None:
    run_dir, _record = _write(tmp_path, weight_dtype="bf16", bundle_steps=(1, 2))
    source = _source(run_dir, dtype="bf16", steps=[1])
    with pytest.warns(UserWarning, match="stored as .*BF16.* and the source was opened with"):
        rule = source.update_response()(source.checkpoints[0].identifier)
    assert rule is not None and rule.next_weights is None


# ------------------------------------------------- the precision of loss_change


def test_loss_change_is_the_fp64_loss_of_the_fp32_logits(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    """The cross entropy is evaluated in fp64 on the logits the fp32 model produced.

    With each batch in ONE micro-batch the analyzer's forward pass is the
    reference's forward pass, bit for bit, so the only thing left between the
    two is the arithmetic of the loss. Evaluated in fp64 they agree to fp64
    round-off; an fp32 cross entropy would differ by its own round-off, about
    ``1e-7`` on a loss of 4 — which is the size of a small residual.
    """
    from tests.utils.bundle_fixture import batch_from_tokens

    run_dir, record = adamw_run
    val = held_out_batch()
    rows_by_step = _run(_source(run_dir, steps=list(WITH_SUCCESSOR)), micro_batch_size=8)
    worst = 0.0
    for step, rows in rows_by_step.items():
        recorded = record.steps[step]
        for name, batch in (("train", batch_from_tokens(recorded.tokens)), ("val", val)):
            reference = _loss_fp64(record, recorded.params_after, batch) - _loss_fp64(
                record, recorded.params_before, batch
            )
            worst = max(worst, abs(_value(rows, name, "loss_change") - reference))
    assert worst < 1e-11, f"loss_change is {worst:.2e} away from the fp64 loss of the same logits"


# ------------------------------------------------------------ failing early


def _no_forward(model: nn.Module, batch: Any) -> torch.Tensor:
    raise AssertionError("a forward pass ran")


def test_a_rule_that_cannot_be_placed_fails_before_any_forward_pass(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    """Both refusals come before the first forward pass of the checkpoint."""
    import dataclasses

    run_dir, _record = adamw_run
    source = _source(run_dir, steps=[2], forward_fn=_no_forward)
    with pytest.raises(ValueError, match="none of the eval batches"):
        _run(source, eval_batches={"val": held_out_batch()})

    build = source.update_response()

    def anonymous(ckpt_id: str) -> OptimizerUpdate | None:
        rule = build(ckpt_id)
        assert rule is not None
        return dataclasses.replace(rule, update_batch=None)

    with pytest.raises(ValueError, match="does not carry the batch"):
        _run(source, update_response=anonymous)
    # The same source does reach the model once the rule can be placed.
    with pytest.raises(AssertionError, match="a forward pass ran"):
        _run(source)


def test_every_eval_batch_is_fetched_once_per_checkpoint(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    """A resampling spec hands out a new batch per call; the rows of one
    checkpoint must all be about the batch its self pair measured."""
    from vatis.data.batches import EvalBatchSpec

    run_dir, record = adamw_run
    drawn: list[dict[str, torch.Tensor]] = []

    class Loader:
        def __iter__(self) -> Any:
            seed = 910
            while True:
                batch = held_out_batch(seed=seed)
                drawn.append(batch)
                seed += 1
                yield batch

    source = _source(run_dir, steps=[1, 2])
    rows_by_step = _run(
        source,
        eval_batches={"train": source.batch_a_spec(), "val": EvalBatchSpec.resample(Loader())},
    )
    assert len(drawn) == 2
    assert not torch.equal(drawn[0]["input_ids"], drawn[1]["input_ids"])
    for (step, rows), batch in zip(rows_by_step.items(), drawn, strict=True):
        recorded = record.steps[step]
        _loss, grad = record.loss_and_grad(recorded.params_before, batch)
        assert _value(rows, "val", "loss_change_first_order") == pytest.approx(
            float(grad @ recorded.delta), rel=RTOL
        )
        assert _value(rows, "val", "delta_loss", batch_a="val") == pytest.approx(
            float(grad @ grad), rel=RTOL
        )
        reference = _loss_fp64(record, recorded.params_after, batch) - _loss_fp64(
            record, recorded.params_before, batch
        )
        assert _value(rows, "val", "loss_change") == pytest.approx(reference, rel=1e-4, abs=1e-7)


# --------------------------------------- work only for the rows that were asked for


def test_theta_next_is_loaded_and_evaluated_only_for_the_rows_that_need_it(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    import dataclasses

    from vatis.models.local_bundle import plain_lm_forward as real_forward

    run_dir, _record = adamw_run
    forwards = 0

    def counting_forward(model: nn.Module, batch: Any) -> torch.Tensor:
        nonlocal forwards
        forwards += 1
        return real_forward(model, batch)

    def measure(observables: list[str]) -> tuple[int, int, set[str]]:
        nonlocal forwards
        source = _source(run_dir, steps=[1], forward_fn=counting_forward)
        build = source.update_response()
        loads = 0

        def factory(ckpt_id: str) -> OptimizerUpdate | None:
            rule = build(ckpt_id)
            assert rule is not None and rule.next_weights is not None
            load = rule.next_weights

            def counted() -> Any:
                nonlocal loads
                loads += 1
                return load()

            return dataclasses.replace(rule, next_weights=counted)

        forwards = 0
        rows = _run(source, update_response=factory, observables=observables)[1]
        return loads, forwards, {r.observable for r in rows}

    formula = ["delta_loss", "loss_change_first_order", "loss_change_first_order_hist"]
    loads, base_forwards, names = measure(formula)
    assert loads == 0 and names == set(formula)

    # The stored weight difference needs theta_{t+1} but no forward pass.
    loads, n_forwards, names = measure([*formula, "loss_change_first_order_weights"])
    assert loads == 1 and n_forwards == base_forwards
    assert names == {*formula, "loss_change_first_order_weights"}

    # The realized change needs two forward passes per micro-batch: 2 micro-batches
    # of the 4-row update batch and 4 of the 8-row held-out batch, at both thetas.
    loads, n_forwards, names = measure([*formula, "loss_change"])
    assert loads == 1 and n_forwards == base_forwards + 2 * (2 + 4)
    assert names == {*formula, "loss_change"}

    loads, n_forwards, names = measure([*formula, "loss_change_residual"])
    assert loads == 1 and n_forwards == base_forwards + 2 * (2 + 4)
    assert names == {*formula, "loss_change_residual"}


# ---------------------------------------------- parameters without a gradient


class _WithUnusedParameter(nn.Module):
    """The tiny transformer plus a parameter the forward pass never touches."""

    def __init__(self) -> None:
        super().__init__()
        from tests.fixtures.update_response_toys import make_model

        self.inner = make_model(dtype=torch.float32).eval()
        self.unused = nn.Parameter(torch.ones(5))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.inner(input_ids)
        return out


def _first_step_rows(model: nn.Module, decay_on_unused: float = 0.1) -> list[ResultRow]:
    """The rows of a first AdamW step (empty state), decay 0.1 on every parameter
    except a parameter named ``unused``, which gets ``decay_on_unused``."""
    params = list(model.parameters())
    batch = held_out_batch(seed=905, n_rows=4)
    unused = getattr(model, "unused", None)
    hypers = [
        AdamHyper(
            lr=1e-2,
            beta1=0.9,
            beta2=0.99,
            eps=1e-8,
            weight_decay=decay_on_unused if p is unused else 0.1,
        )
        for p in params
    ]
    rule = OptimizerUpdate(
        family="adam",
        hypers=hypers,
        step=0,
        first_moment=None,
        second_moment=None,
        descriptor="adamw_update",
        update_batch=batch,
    )
    results = analyze(
        model=ModelBundle(
            model=model,
            params=params,
            forward_fn=plain_lm_forward,
            loss_fn=causal_lm_loss,
            valid_mask_fn=causal_lm_valid_mask,
            identifier="toy",
        ),
        eval_batches={"A": batch, "B": held_out_batch(seed=906, n_rows=4)},
        update_response=lambda ckpt_id: rule,
        chi_net_method="hutchinson",
        n_hutchinson=2,
        dtype="fp32",
    )
    return results[0].rows


def test_a_parameter_without_a_gradient_is_refused_when_the_rule_would_move_it() -> None:
    """torch skips a parameter whose gradient is ``None``; the rule would decay it.

    The analyzer knows the parameter got no gradient on the update batch (autograd
    returned ``None``), and whether the run skipped it or decayed it is not in a
    bundle — so the rows are refused, naming the parameter.
    """
    torch.manual_seed(0)
    with pytest.raises(ValueError, match=r"parameter \d+ \(unused\) receives no gradient"):
        _first_step_rows(_WithUnusedParameter(), decay_on_unused=0.1)


def test_a_parameter_without_a_gradient_is_harmless_when_nothing_moves_it() -> None:
    """No decay and no history on it: skipped or not, its update is zero."""
    torch.manual_seed(0)
    model = _WithUnusedParameter()
    with_unused = _first_step_rows(model, decay_on_unused=0.0)
    # The same model without the extra parameter gives the same rows.
    alone = _first_step_rows(model.inner)
    for batch in ("A", "B"):
        for observable in UPDATE_RESPONSE_OBSERVABLES:
            assert _value(with_unused, batch, observable, batch_a="A") == pytest.approx(
                _value(alone, batch, observable, batch_a="A"), rel=1e-12, abs=1e-18
            )


# ------------------------------------------- parameters outside the canonical list


def _frozen_position_embedding(model: nn.Module) -> torch.optim.Optimizer:
    """AdamW over every parameter but the position embedding, which is frozen."""
    frozen = dict(model.named_parameters())["pos_embed.weight"]
    frozen.requires_grad_(False)
    return torch.optim.AdamW(
        [p for p in model.parameters() if p is not frozen],
        lr=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )


def test_unlisted_parameters_must_be_confirmed_frozen(tmp_path: Path) -> None:
    """A model with parameters ``param_names.json`` does not name.

    If they were trained, their part of the step is in ``loss_change`` and in no
    other row; if they had gradients when the norm was clipped, the clip factor
    is wrong. A bundle cannot say, so the caller has to.
    """
    run_dir, record = _write(tmp_path, _frozen_position_embedding, clip_grad_norm=0.5)
    assert list(record.frozen) == ["pos_embed.weight"]
    assert "pos_embed.weight" not in record.names

    source = _source(run_dir, steps=[0, 1], allow_unlisted_params=True)
    with pytest.raises(ValueError, match=r"outside\s+param_names.json \(\['pos_embed.weight'\]"):
        _run(source)

    confirmed = _source(run_dir, steps=[0, 1], allow_unlisted_params=True)
    rows_by_step = _run(
        confirmed, update_response=confirmed.update_response(unlisted_params_frozen=True)
    )
    _assert_matches_the_run(record, rows_by_step)
    for step, rows in rows_by_step.items():
        recorded = record.steps[step]
        val = held_out_batch()
        reference = _loss_fp64(record, recorded.params_after, val) - _loss_fp64(
            record, recorded.params_before, val
        )
        assert _value(rows, "val", "loss_change") == pytest.approx(reference, rel=1e-4, abs=1e-7)


def test_allow_unlisted_params_is_harmless_when_every_parameter_is_listed(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    run_dir, record = adamw_run
    source = _source(run_dir, steps=[1], allow_unlisted_params=True)
    _assert_matches_the_run(record, _run(source))


# ------------------------------------- the step's learning rate and clip factor


def test_the_learning_rate_row_is_the_rate_the_step_applied(tmp_path: Path) -> None:
    """One rate for every parameter: ``update_lr``, at every checkpoint, step 0 included."""
    run_dir, record = _write(
        tmp_path,
        make_scheduler=lambda opt: torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.05, end_factor=1.0, total_iters=6
        ),
    )
    rows_by_step = _run(_source(run_dir))
    assert 0 in rows_by_step
    seen = set()
    for step, rows in rows_by_step.items():
        rate = _value(rows, "train", "update_lr")
        assert rate == record.steps[step].lr
        seen.add(rate)
        assert not _has(rows, "update_lr_min") and not _has(rows, "update_lr_max")
        # Only on the update batch's own pair, tagged like the formula rows.
        (row,) = [r for r in rows if r.observable == "update_lr"]
        assert (row.batch_a, row.batch_b, row.preconditioner) == ("train", "train", "adamw_update")
        assert row.hutchinson_seed is None and row.wallclock_s is None
    assert len(seen) >= 5, "the schedule did not move the rate"


def test_different_rates_per_group_give_a_range_not_one_number(tmp_path: Path) -> None:
    """No ``update_lr`` when parameter groups have different rates: the smallest and
    the largest are recorded instead, and nothing that looks like *the* rate."""
    run_dir, _record = _write(
        tmp_path,
        lambda model: torch.optim.AdamW(
            decay_groups(model, weight_decay=0.5, lr_no_decay=1e-3), lr=3e-3, betas=(0.9, 0.95)
        ),
    )
    for rows in _run(_source(run_dir, steps=[0, 2])).values():
        assert not _has(rows, "update_lr")
        assert _value(rows, "train", "update_lr_min") == 1e-3
        assert _value(rows, "train", "update_lr_max") == 3e-3


def test_the_clip_factor_row_is_the_factor_torch_applied(tmp_path: Path) -> None:
    run_dir, record = _write(tmp_path, clip_grad_norm=0.5)
    rows_by_step = _run(_source(run_dir))
    factors = {}
    for step, rows in rows_by_step.items():
        recorded = record.steps[step]
        applied = float(recorded.grad_seen.norm() / recorded.grad_update.norm())
        factors[step] = _value(rows, "train", "update_clip_factor")
        assert factors[step] == pytest.approx(applied, rel=RTOL)
        assert 0.0 < factors[step] <= 1.0
        (row,) = [r for r in rows if r.observable == "update_clip_factor"]
        assert (row.batch_a, row.batch_b, row.preconditioner) == ("train", "train", "adamw_update")
        # It is recoverable from the update batch's own delta_loss row (= ||g||^2).
        norm = _value(rows, "train", "delta_loss") ** 0.5
        assert factors[step] == pytest.approx(min(1.0, 0.5 / (norm + 1e-6)), rel=1e-9)
    assert 0 in factors and factors[0] < 1.0, "the clip did not engage at the first step"


def test_the_clip_factor_of_a_run_that_did_not_clip_is_exactly_one(
    adamw_run: tuple[Path, TrainingRecord],
) -> None:
    run_dir, _record = adamw_run
    for rows in _run(_source(run_dir)).values():
        assert _value(rows, "train", "update_clip_factor") == 1.0


def test_step_rows_for_sgd_and_on_request_only(tmp_path: Path) -> None:
    run_dir, _record = _write(
        tmp_path,
        lambda model: torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9),
        optimizer_name="sgd",
        bundle_steps=(0, 1),
    )
    source = _source(run_dir)
    rows = _run(source)[1]
    (row,) = [r for r in rows if r.observable == "update_lr"]
    assert (row.value, row.preconditioner) == (0.05, "sgd_update")
    # Not without an update rule, and selectable like any other row.
    assert not any(
        r.observable in UPDATE_STEP_OBSERVABLES for r in _run(source, update_response=None)[1]
    )
    picked = _run(source, observables=["delta_loss", "update_clip_factor"])[1]
    assert {r.observable for r in picked} == {"delta_loss", "update_clip_factor"}
    with pytest.raises(ValueError, match="no update_response was given"):
        Analyzer(eval_batches={"a": held_out_batch()}, observables=["update_lr"])


# -------------------------------------------------------------------- on a GPU


@pytest.mark.gpu
@pytest.mark.parametrize("cross_grad_storage", ["gpu", "cpu"])
def test_rows_on_a_gpu_match_the_cpu_rows(tmp_path: Path, cross_grad_storage: str) -> None:
    """The device-dependent paths of the new rows, against the same analysis on the CPU.

    On a CUDA source the weights, the successor model and (by default) the
    gradients live on the GPU while the saved optimizer state stays on the CPU,
    so the fp64 reductions move state chunks to the gradients' device; with
    ``cross_grad_storage="cpu"`` the gradients are offloaded to the host and the
    weights of both models are moved to them instead. Both arrangements have to
    give the CPU rows: the formula rows, the step rows (the clip engages, so the
    factor is formed from the gradient norm on the device) and the realized rows
    (fp64 loss on GPU logits, weight difference of two GPU models).
    """
    run_dir, record = _write(tmp_path, clip_grad_norm=0.5, bundle_steps=(0, 1, 2, 3))
    on_cpu = _run(_source(run_dir, steps=[0, 2]))
    gpu_source = _source(run_dir, steps=[0, 2], device="cuda")
    on_gpu = _run(gpu_source, cross_grad_storage=cross_grad_storage)
    names = UPDATE_RESPONSE_OBSERVABLES + UPDATE_STEP_OBSERVABLES + REALIZED_CHANGE_OBSERVABLES
    compared = 0
    for step in (0, 2):
        cpu_rows = {
            (r.batch_a, r.batch_b, r.observable): r.value
            for r in on_cpu[step]
            if r.observable in names
        }
        gpu_rows = {
            (r.batch_a, r.batch_b, r.observable): r.value
            for r in on_gpu[step]
            if r.observable in names
        }
        assert set(gpu_rows) == set(cpu_rows) and len(cpu_rows) == 2 * 7 + 2
        for key, expected in cpu_rows.items():
            # The residual is a difference of two numbers of the first-order
            # term's size; its tolerance is set by that size.
            scale = abs(cpu_rows[(key[0], key[1], "loss_change_first_order_weights")])
            tolerance = RTOL * (scale if key[2] == "loss_change_residual" else abs(expected))
            assert abs(gpu_rows[key] - expected) <= tolerance, (step, key, gpu_rows[key], expected)
            compared += 1
        # ... and against the run itself, not only against the CPU analysis.
        assert gpu_rows[("train", "val", "loss_change_first_order")] == pytest.approx(
            _reference(record, step, "val"), rel=RTOL
        )
        assert gpu_rows[("train", "train", "update_clip_factor")] < 1.0
    assert compared == 32


# ------------------------------------------ the provenance columns on these rows

STEP_TAKEN_ROWS = (
    UPDATE_RESPONSE_OBSERVABLES + UPDATE_STEP_OBSERVABLES + REALIZED_CHANGE_OBSERVABLES
)
SAMPLED_ROWS = (
    "chi_net",
    "chi_net_normalized",
    "chi_pos",
    "chi_net_P",
    "chi_pos_P",
    "chi_net_P_capped",
)
ADAMW_P_ROWS = ("delta_loss_P", "chi_net_P", "chi_pos_P", "delta_loss_P_capped", "chi_net_P_capped")


@pytest.mark.parametrize(
    ("method", "distribution", "micro"),
    [("hutchinson", "rademacher", 2), ("per_sequence_cv", "gaussian", 1)],
)
def test_the_rows_of_the_step_taken_follow_the_column_rule_through_the_parquet_file(
    adamw_run: tuple[Path, TrainingRecord],
    tmp_path: Path,
    method: str,
    distribution: str,
    micro: int,
) -> None:
    """What the eight provenance columns hold on the ``loss_change*`` and step rows.

    - the run settings (``chi_net_method``, ``n_ranks``, ``micro_batch_size``,
      ``hutchinson_distribution``, and ``n_hutchinson``) are on them as on every
      other row of the file;
    - the three ``preconditioner_*`` columns are null: an ``adamw_update`` row
      is not a row of an AdamW ``P``;
    - ``value_probe_stderr`` is null: nothing in them is sampled.

    Read back from the parquet file, next to the rows of an AdamW ``P`` of the
    same run, so that the rule for the whole file can be checked row by row.
    """
    from vatis.sinks import read_results

    run_dir, _record = adamw_run
    source = _source(run_dir, steps=[1, 2])
    out = tmp_path / "results.parquet"
    analyzer = Analyzer(
        eval_batches={"train": source.batch_a_spec(), "val": held_out_batch()},
        update_response=source.update_response(),
        preconditioner=source.preconditioner(),
        cross_pairs=[("train", "val")],
        chi_net_method=method,
        hutchinson_distribution=distribution,
        n_hutchinson=3,
        micro_batch_size=micro,
        sink=str(out),
        dtype="fp32",
    )
    analyzer.run(source.model_specs())
    rows = read_results(out).to_pylist()
    mine = [r for r in rows if r["observable"] in STEP_TAKEN_ROWS]
    assert {r["observable"] for r in mine} == {
        name for name in STEP_TAKEN_ROWS if name not in ("update_lr_min", "update_lr_max")
    }

    # --- on the rows of the step taken -------------------------------------
    for r in mine:
        assert r["chi_net_method"] == method, r
        assert r["n_ranks"] == 1 and r["micro_batch_size"] == micro, r
        assert r["hutchinson_distribution"] == distribution, r
        assert r["n_hutchinson"] == 3, r
        assert r["hutchinson_seed"] is None and r["wallclock_s"] is None, r
        assert r["preconditioner_eps"] is None, r
        assert r["preconditioner_beta2"] is None, r
        assert r["preconditioner_capped_fraction"] is None, r
        assert r["value_probe_stderr"] is None, r
        tagged = r["observable"] in UPDATE_RESPONSE_OBSERVABLES + UPDATE_STEP_OBSERVABLES
        assert r["preconditioner"] == ("adamw_update" if tagged else "none"), r

    # --- and the rule for the whole file still holds, row by row -----------
    for r in rows:
        assert r["chi_net_method"] == method and r["n_ranks"] == 1, r
        assert r["micro_batch_size"] == micro and r["n_hutchinson"] == 3, r
        assert r["hutchinson_distribution"] == distribution, r
        is_p_row = r["observable"] in ADAMW_P_ROWS
        assert (r["preconditioner_eps"] is not None) == is_p_row, r
        assert (r["preconditioner_beta2"] is not None) == is_p_row, r
        assert (r["preconditioner_capped_fraction"] is not None) == is_p_row, r
        assert (r["value_probe_stderr"] is not None) == (r["observable"] in SAMPLED_ROWS), r

    # --- no reader takes an update-rule row for a row of the AdamW P --------
    by_name = {id(r) for r in rows if r["observable"] in ADAMW_P_ROWS}
    by_descriptor = {id(r) for r in rows if r["preconditioner"] == "adamw"}
    by_eps = {id(r) for r in rows if r["preconditioner_eps"] is not None}
    assert by_name == by_descriptor == by_eps and by_name
    assert not by_name & {id(r) for r in mine}
    # ``preconditioner != "none"`` alone is NOT such a selection: it also holds
    # for the rows tagged with an update rule.
    not_raw = {r["preconditioner"] for r in rows if r["preconditioner"] != "none"}
    assert not_raw == {"adamw", "adamw_update"}


def test_a_pair_measured_with_two_estimators_names_both_on_these_rows_too(
    adamw_run: tuple[Path, TrainingRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The estimator is resolved per batch; the rows on (update batch, B) record
    the two batches' estimators the way a cross pair does."""
    import vatis.analyzer as analyzer_mod

    monkeypatch.setattr(
        analyzer_mod,
        "select_chi_net_method",
        lambda b_total, requested: "per_sequence_cv" if b_total <= 4 else "hutchinson",
    )
    run_dir, _record = adamw_run
    source = _source(run_dir, steps=[1])
    analyzer = Analyzer(
        eval_batches={"train": source.batch_a_spec(), "val": held_out_batch()},
        update_response=source.update_response(),
        n_hutchinson=2,
        micro_batch_size=2,
        dtype="fp32",
    )
    rows = analyzer.run(source.model_specs())[0].rows
    methods: dict[tuple[str, str], set[str | None]] = {}
    for r in rows:
        if r.observable in STEP_TAKEN_ROWS:
            methods.setdefault((r.batch_a, r.batch_b), set()).add(r.chi_net_method)
    assert methods == {
        ("train", "train"): {"per_sequence_cv"},
        ("train", "val"): {"per_sequence_cv+hutchinson"},
    }


def test_step_rows_record_the_world_size(
    adamw_run: tuple[Path, TrainingRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``n_ranks`` is the world size the run saw (emulated: rank 0 of 3, no all-reduce)."""
    import vatis.analyzer as analyzer_mod

    monkeypatch.setattr(analyzer_mod, "get_world_size", lambda: 3)
    run_dir, _record = adamw_run
    source = _source(run_dir, steps=[1])
    rows = _run(source, update_response=source.update_response(realized=False))[1]
    mine = [r for r in rows if r.observable in STEP_TAKEN_ROWS]
    assert mine and {r.n_ranks for r in mine} == {3}
