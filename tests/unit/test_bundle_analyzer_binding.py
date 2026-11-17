"""What the review of the preconditioned-observables PR showed the suite could not see.

Each test here corresponds to a mutant that survived the unit and golden tiers,
or to an input the bundle path mishandled:

- **Step binding.** Loading every checkpoint's weights from the first (or last)
  checkpoint, building every ``P`` from the last checkpoint's optimizer state,
  or handing every checkpoint the first checkpoint's update batch all passed,
  because the older fixtures write the same theta at every step and the golden
  gate asserts presence and finiteness only. Here theta, ``v`` and the batch all
  differ per step and the analyzer's rows are compared to values recomputed from
  what was written.
- **The cross-pair ``chi_pos_P`` row** was asserted by no test.
- Loader inputs: a directory step that contradicts the optimizer's counter, a
  persistent buffer the file does not hold, a 0-dim parameter, a parameter that
  never received a gradient, per-group ``eps``, a non-Adam optimizer, and an
  ``eps`` / ``beta2`` override that used to leave the descriptor at ``"adamw"``.

A second review added three things. The fixture has **three** bundled
checkpoints: with two, a loader that hands the third and later checkpoints the
previous checkpoint's optimizer state changed no asserted value. RAdam and NAdam
states are refused **whether or not** ``metadata.json`` names the optimizer —
their param groups have ``betas`` like Adam's, and without a name they used to
come back labelled ``"adamw"``. And the optimizer state is read on the CPU also
when the source places its model on a GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import vatis.analyzer as analyzer_mod
from tests.fixtures.synthetic_bundle import (
    BundleTinyLM,
    bundle_metadata,
    handwritten_optim_state,
    make_batch_a,
    tiny_lm_factory,
    write_checkpoint,
    write_run_files,
)
from tests.fixtures.tiny_transformer import (
    TinyTransformer,
    causal_lm_loss,
    causal_lm_valid_mask,
    make_tiny_lm_batch,
)
from vatis import Analyzer, analyze, load_local_bundle
from vatis.core.param_layout import ParameterLayout
from vatis.core.preconditioner import Preconditioner, weighted_dot_fp64, weighted_sum_sq_fp64
from vatis.data.collate import shifted_lm_targets
from vatis.models.hf import ModelBundle
from vatis.models.local_bundle import (
    BundleMetadata,
    BundleSchemaError,
    OptimizerStateView,
    _optimizer_family,
)

BETAS = (0.9, 0.95)
EPS = 1e-8
LR = 1e-2
# Three, not two: a defect that only shows from the third checkpoint on (e.g. a
# state that lags one checkpoint behind) needs a third checkpoint to be seen.
BUNDLED_STEPS = (2, 5, 9)


def _lm_loss(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    logits = model(ids)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shifted_lm_targets(ids).reshape(-1),
        ignore_index=-100,
    )


def _flat_grad(model: nn.Module, order: list[str], ids: torch.Tensor) -> torch.Tensor:
    named = dict(model.named_parameters())
    grads = torch.autograd.grad(_lm_loss(model, ids), [named[n] for n in order])
    return torch.cat([g.reshape(-1).to(torch.float64) for g in grads])


def _as_loaded(model: BundleTinyLM) -> BundleTinyLM:
    """The model as the loader will rebuild it: weights rounded through bf16."""
    twin = BundleTinyLM(seed=0)
    twin.load_state_dict(
        {k: v.to(torch.bfloat16).to(torch.float32) for k, v in model.state_dict().items()}
    )
    return twin


@pytest.fixture
def trained_bundle(tmp_path: Path) -> dict[str, Any]:
    """A bundle whose theta, second moment and update batch all change per step."""
    torch.manual_seed(0)
    model = BundleTinyLM(seed=0)
    run_dir = tmp_path / "run"
    order = write_run_files(
        run_dir, model, metadata=bundle_metadata(model, betas=BETAS, eps=EPS, lr=LR)
    )
    named = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [named[n] for n in order], lr=LR, betas=BETAS, eps=EPS, weight_decay=0.1
    )
    reference: dict[int, dict[str, torch.Tensor]] = {}
    for step in range(max(BUNDLED_STEPS) + 1):
        ids_np = make_batch_a(seed=100 + step)
        ids = torch.from_numpy(ids_np).long()
        if step in BUNDLED_STEPS:
            # pre-step: theta_t, the state after t updates, the batch step t consumes
            write_checkpoint(
                run_dir, step, model=model, optim_state=optimizer.state_dict(), batch_a=ids_np
            )
            state = optimizer.state_dict()["state"]
            v = torch.cat([state[i]["exp_avg_sq"].reshape(-1) for i in range(len(order))])
            p_diag = 1.0 / ((v.double() / (1.0 - BETAS[1] ** step)).sqrt() + EPS)
            reference[step] = {
                "ids": ids,
                "p": p_diag,
                "g": _flat_grad(_as_loaded(model), order, ids),
            }
        optimizer.zero_grad(set_to_none=True)
        _lm_loss(model, ids).backward()
        optimizer.step()
    val = torch.from_numpy(make_batch_a(seed=999)).long()
    return {"run_dir": run_dir, "order": order, "reference": reference, "val": val}


def _rows(results: list[Any]) -> dict[tuple[str, str, str, str], float]:
    return {
        (r.revision, r.batch_a, r.batch_b, r.observable): r.value
        for result in results
        for r in result.rows
    }


def _run(bundle: dict[str, Any], **kwargs: Any) -> dict[tuple[str, str, str, str], float]:
    source = load_local_bundle(bundle["run_dir"], tiny_lm_factory, dtype="fp32")
    val_batch = {"input_ids": bundle["val"], "labels": shifted_lm_targets(bundle["val"])}
    analyzer = Analyzer(
        eval_batches={"train": source.batch_a_spec(), "val": val_batch},
        preconditioner=source.adamw_preconditioner(),
        cross_pairs=[("train", "val")],
        chi_net_method="hutchinson",
        n_hutchinson=4,
        micro_batch_size=2,
        dtype="fp32",
        sink=None,
        **kwargs,
    )
    return _rows(analyzer.run(source.model_specs()))


# ------------------------------------------------------------- step binding


def test_each_checkpoint_is_measured_with_its_own_theta_state_and_batch(
    trained_bundle: dict[str, Any],
) -> None:
    rows = _run(trained_bundle)
    ref = trained_bundle["reference"]
    expected = {
        step: (float(r["g"] @ r["g"]), float((r["p"] * r["g"] * r["g"]).sum()))
        for step, r in ref.items()
    }
    # vacuity guard: the two checkpoints must really differ in both quantities
    (raw_a, p_a), (raw_b, p_b) = expected[BUNDLED_STEPS[0]], expected[BUNDLED_STEPS[1]]
    assert abs(raw_a - raw_b) > 0.02 * max(raw_a, raw_b)
    assert abs(p_a - p_b) > 0.02 * max(p_a, p_b)
    # ... and so must every pair of the three
    assert len(expected) == len(BUNDLED_STEPS) == 3
    for first in BUNDLED_STEPS:
        for second in BUNDLED_STEPS:
            if first < second:
                (raw_a, p_a), (raw_b, p_b) = expected[first], expected[second]
                assert abs(raw_a - raw_b) > 0.02 * max(raw_a, raw_b), (first, second)
                assert abs(p_a - p_b) > 0.02 * max(p_a, p_b), (first, second)

    for step, (raw, weighted) in expected.items():
        rev = f"step{step}"
        assert rows[(rev, "train", "train", "delta_loss")] == pytest.approx(raw, rel=1e-5)
        assert rows[(rev, "train", "train", "delta_loss_P")] == pytest.approx(weighted, rel=1e-5)


def test_a_p_built_from_another_checkpoints_state_would_be_seen(
    trained_bundle: dict[str, Any],
) -> None:
    """The discriminating power of the test above, stated as its own assertion."""
    ref = trained_bundle["reference"]
    first, last = ref[BUNDLED_STEPS[0]], ref[BUNDLED_STEPS[1]]
    own = float((first["p"] * first["g"] * first["g"]).sum())
    swapped = float((last["p"] * first["g"] * first["g"]).sum())
    assert abs(own - swapped) > 0.02 * own


def test_a_p_built_from_the_previous_checkpoints_state_would_be_seen_at_every_checkpoint(
    trained_bundle: dict[str, Any],
) -> None:
    """Each later checkpoint, measured with the state of the one before it, gives
    a ``delta_loss_P`` the value assertion above rejects (it asserts ``rel=1e-5``).

    This is what a state that lags one checkpoint behind produces. With two
    bundled checkpoints a loader doing that from the third checkpoint on is not
    exercised at all.
    """
    ref = trained_bundle["reference"]
    assert len(BUNDLED_STEPS) >= 3
    for previous, current in zip(BUNDLED_STEPS, BUNDLED_STEPS[1:], strict=False):
        g = ref[current]["g"]
        own = float((ref[current]["p"] * g * g).sum())
        lagging = float((ref[previous]["p"] * g * g).sum())
        assert abs(own - lagging) > 0.02 * own, (previous, current)


def test_the_third_checkpoints_p_is_the_one_from_its_own_state(
    trained_bundle: dict[str, Any],
) -> None:
    """The same binding, asserted on ``P`` itself rather than through a row."""
    source = load_local_bundle(trained_bundle["run_dir"], tiny_lm_factory, dtype="fp32")
    factory = source.adamw_preconditioner()
    ref = trained_bundle["reference"]
    for step in BUNDLED_STEPS:
        precond = factory(source.checkpoint_for(step).identifier)
        assert precond is not None
        assert torch.allclose(precond.diag.double(), ref[step]["p"], rtol=1e-6, atol=0.0), step
    second, third = ref[BUNDLED_STEPS[1]]["p"], ref[BUNDLED_STEPS[2]]["p"]
    assert not torch.allclose(second, third, rtol=1e-3, atol=0.0)


# ------------------------------------------------------- cross-pair P rows


def test_cross_pair_p_rows_are_the_p_weighted_quantities(trained_bundle: dict[str, Any]) -> None:
    rows = _run(trained_bundle)
    order, val = trained_bundle["order"], trained_bundle["val"]
    source = load_local_bundle(trained_bundle["run_dir"], tiny_lm_factory, dtype="fp32")
    for step, ref in trained_bundle["reference"].items():
        rev = f"step{step}"
        g_val = _flat_grad(source.checkpoint_for(step).model_bundle.model, order, val)
        cross = lambda name, rev=rev: rows[(rev, "train", "val", name)]  # noqa: E731
        self_ = lambda batch, name, rev=rev: rows[(rev, batch, batch, name)]  # noqa: E731

        # delta_loss_P(A, B) against the hand-built P
        assert cross("delta_loss_P") == pytest.approx(
            float((ref["p"] * ref["g"] * g_val).sum()), rel=1e-5
        )
        # the cross chi_net_P is the geometric mean of the two self values ...
        geo_p = math.sqrt(self_("train", "chi_net_P") * self_("val", "chi_net_P"))
        geo_raw = math.sqrt(self_("train", "chi_net") * self_("val", "chi_net"))
        assert cross("chi_net_P") == pytest.approx(geo_p, rel=1e-6)
        assert cross("chi_net") == pytest.approx(geo_raw, rel=1e-6)
        # ... and the published chi_pos rows divide by their own kernel's chi_net.
        assert cross("chi_pos_P") == pytest.approx(
            cross("delta_loss_P") / (cross("chi_loss") * geo_p), rel=1e-6
        )
        assert cross("chi_pos") == pytest.approx(
            cross("delta_loss") / (cross("chi_loss") * geo_raw), rel=1e-6
        )
        # vacuity guard: dividing the P numerator by the raw chi_net is a different number
        wrong = cross("delta_loss_P") / (cross("chi_loss") * geo_raw)
        assert abs(wrong - cross("chi_pos_P")) > 0.5 * abs(cross("chi_pos_P"))


# ------------------------------------------------------------ loader inputs


def test_optimizer_state_is_released_once_p_is_built(trained_bundle: dict[str, Any]) -> None:
    source = load_local_bundle(trained_bundle["run_dir"], tiny_lm_factory, dtype="fp32")
    checkpoint = source.checkpoint_for(BUNDLED_STEPS[1])
    precond = source.adamw_preconditioner()(checkpoint.identifier)
    assert precond is not None
    assert checkpoint._optimizer_state is None
    expected = trained_bundle["reference"][BUNDLED_STEPS[1]]["p"]
    assert torch.allclose(precond.diag.double(), expected, rtol=1e-6, atol=0.0)


def test_inplace_adamw_build_is_bit_identical_and_only_then_overwrites_its_input() -> None:
    gen = torch.Generator().manual_seed(4)
    v = torch.rand(257, generator=gen) * 1e-3
    v[::7] = 0.0
    reference = 1.0 / ((v / (1.0 - 0.95**9)).sqrt() + 1e-8)

    kept = v.clone()
    out_of_place = Preconditioner.adamw(kept, step=9, beta2=0.95, eps=1e-8)
    assert torch.equal(kept, v), "the default must not modify the caller's tensor"
    assert torch.equal(out_of_place.diag, reference)

    owned = v.clone()
    in_place = Preconditioner.adamw(owned, step=9, beta2=0.95, eps=1e-8, inplace=True)
    assert torch.equal(in_place.diag, reference)
    assert in_place.diag.data_ptr() == owned.data_ptr()


def test_directory_step_that_contradicts_the_optimizer_counter_warns(tmp_path: Path) -> None:
    model = BundleTinyLM(seed=0)
    run_dir = tmp_path / "run"
    order = write_run_files(run_dir, model, metadata=bundle_metadata(model))
    write_checkpoint(
        run_dir,
        5,
        model=model,
        optim_state=handwritten_optim_state(model, order, step=500),
        batch_a=make_batch_a(seed=5),
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    # ``"warn"`` is the behaviour that used to be the default: a warning, and P
    # built from the saved state's own counter (500), not from the directory's.
    with pytest.warns(UserWarning, match="directory says step 5"):
        precond = source.adamw_preconditioner(on_step_mismatch="warn")(
            source.checkpoints[0].identifier
        )
    assert precond is not None
    v = source.checkpoints[0].optimizer_state.flat_state("exp_avg_sq")
    assert torch.equal(precond.diag, 1.0 / ((v / (1.0 - 0.95**500)).sqrt() + 1e-8))
    assert not torch.equal(precond.diag, 1.0 / ((v / (1.0 - 0.95**5)).sqrt() + 1e-8))
    with pytest.warns(UserWarning, match="directory says step 5"):
        unified = source.preconditioner(on_step_mismatch="warn")(source.checkpoints[0].identifier)
    assert unified is not None and torch.equal(unified.diag, precond.diag)


def test_directory_step_that_contradicts_the_optimizer_counter_is_refused_by_default(
    tmp_path: Path,
) -> None:
    """Under pre-step semantics the two numbers are the same; when they are not,
    the optimizer state and the weights may not belong together. A warning let a
    loader that hands a checkpoint another checkpoint's state pass every tier."""
    model = BundleTinyLM(seed=0)
    run_dir = tmp_path / "run"
    order = write_run_files(run_dir, model, metadata=bundle_metadata(model))
    write_checkpoint(
        run_dir,
        5,
        model=model,
        optim_state=handwritten_optim_state(model, order, step=500),
        batch_a=make_batch_a(seed=5),
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    ident = source.checkpoints[0].identifier
    message = "applied 500 update.*directory says step 5.*may not belong together"
    for factory in (
        source.adamw_preconditioner(),
        source.adamw_preconditioner(on_step_mismatch="raise"),
        source.preconditioner(),
        source.adamw_preconditioner(beta2=0.999, eps=1e-6),  # an override does not lift it
    ):
        with pytest.raises(BundleSchemaError, match=message):
            factory(ident)
    # a consistent checkpoint is unaffected by the option
    write_checkpoint(
        run_dir,
        7,
        model=model,
        optim_state=handwritten_optim_state(model, order, step=7),
        batch_a=make_batch_a(seed=7),
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    consistent = source.checkpoint_for(7).identifier
    strict = source.adamw_preconditioner()(consistent)
    lenient = source.adamw_preconditioner(on_step_mismatch="warn")(consistent)
    assert strict is not None and lenient is not None
    assert torch.equal(strict.diag, lenient.diag)


@pytest.mark.parametrize("value", ["ignore", "error", "", "Raise", None, True])
def test_on_step_mismatch_accepts_raise_and_warn_only(
    trained_bundle: dict[str, Any], value: Any
) -> None:
    source = load_local_bundle(trained_bundle["run_dir"], tiny_lm_factory)
    with pytest.raises(ValueError, match="on_step_mismatch must be one of"):
        source.adamw_preconditioner(on_step_mismatch=value)
    with pytest.raises(ValueError, match="on_step_mismatch must be one of"):
        source.preconditioner(on_step_mismatch=value)


class _BufferLM(BundleTinyLM):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.register_buffer("logit_scale", torch.tensor(1.0))  # persistent

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.logit_scale


def test_persistent_buffer_missing_from_the_file_is_reported(tmp_path: Path) -> None:
    trained = _BufferLM(seed=0)
    trained.logit_scale.fill_(7.0)
    run_dir = tmp_path / "run"
    order = write_run_files(run_dir, trained, metadata=bundle_metadata(trained))
    write_checkpoint(
        run_dir,
        2,
        model=trained,
        optim_state=handwritten_optim_state(trained, order, step=2),
        batch_a=make_batch_a(seed=2),
    )
    source = load_local_bundle(run_dir, lambda metadata: _BufferLM(seed=1))
    with pytest.warns(UserWarning, match="persistent buffer.*logit_scale"):
        source.checkpoints[0].model_bundle  # noqa: B018


class _ExtraParamLM(BundleTinyLM):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.extra = nn.Parameter(torch.ones(3))


def test_weight_file_with_a_tensor_the_model_lacks_is_refused(tmp_path: Path) -> None:
    base = BundleTinyLM(seed=0)
    run_dir = tmp_path / "run"
    order = write_run_files(run_dir, base, metadata=bundle_metadata(base))
    write_checkpoint(
        run_dir,
        2,
        model=_ExtraParamLM(seed=0),
        optim_state=handwritten_optim_state(base, order, step=2),
        batch_a=make_batch_a(seed=2),
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    with pytest.raises(BundleSchemaError, match="that the model does not have"):
        source.checkpoints[0].model_bundle  # noqa: B018


class _ScalarLM(BundleTinyLM):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.temperature


class _UnusedParamLM(BundleTinyLM):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.never_used = nn.Parameter(torch.zeros(3))


def _real_adamw_bundle(
    tmp_path: Path,
    model: BundleTinyLM,
    *,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
    metadata_optimizer: str = "adamw",
    param_groups: Any = None,
    n_steps: int = 3,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    order = write_run_files(
        run_dir, model, metadata=bundle_metadata(model, optimizer=metadata_optimizer)
    )
    named = dict(model.named_parameters())
    params = param_groups(named, order) if param_groups else [named[n] for n in order]
    optimizer = optimizer_cls(params, lr=1e-3, **(optimizer_kwargs or {}))
    ids = torch.from_numpy(make_batch_a(seed=0)).long()
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        _lm_loss(model, ids).backward()
        optimizer.step()
    write_checkpoint(
        run_dir,
        n_steps,
        model=model,
        optim_state=optimizer.state_dict(),
        batch_a=make_batch_a(seed=n_steps),
    )
    return run_dir


def test_zero_dim_parameter_has_a_p_entry(tmp_path: Path) -> None:
    model = _ScalarLM(seed=0)
    run_dir = _real_adamw_bundle(tmp_path, model)
    source = load_local_bundle(run_dir, lambda metadata: _ScalarLM(seed=0))
    precond = source.adamw_preconditioner()(source.checkpoints[0].identifier)
    assert precond is not None
    assert len(precond) == sum(p.numel() for p in model.parameters())
    assert torch.isfinite(precond.diag).all()


def test_parameter_that_never_received_a_gradient_is_named_as_such(tmp_path: Path) -> None:
    run_dir = _real_adamw_bundle(tmp_path, _UnusedParamLM(seed=0))
    source = load_local_bundle(run_dir, lambda metadata: _UnusedParamLM(seed=0))
    with pytest.raises(BundleSchemaError, match="never received a\\s+gradient|never updated"):
        source.adamw_preconditioner()(source.checkpoints[0].identifier)


def test_param_groups_that_disagree_on_eps_are_refused_with_the_reason(tmp_path: Path) -> None:
    def groups(named: dict[str, nn.Parameter], order: list[str]) -> list[dict[str, Any]]:
        half = len(order) // 2
        return [
            {"params": [named[n] for n in order[:half]], "eps": 1e-8},
            {"params": [named[n] for n in order[half:]], "eps": 1e-6},
        ]

    run_dir = _real_adamw_bundle(tmp_path, BundleTinyLM(seed=0), param_groups=groups)
    source = load_local_bundle(run_dir, tiny_lm_factory)
    with pytest.raises(ValueError, match="disagree on \\['eps'\\]"):
        source.adamw_preconditioner()(source.checkpoints[0].identifier)
    # an explicit pair for all parameters is accepted, and is visible in the descriptor
    precond = source.adamw_preconditioner(beta2=0.999, eps=1e-8)(source.checkpoints[0].identifier)
    assert precond is not None
    assert precond.descriptor == "adamw(beta2=0.999,eps=1e-08)"


@pytest.mark.parametrize("amsgrad_everywhere", [True, False])
@pytest.mark.parametrize("explicit_hparams", [False, True])
def test_amsgrad_is_refused_in_any_group_and_under_an_override(
    tmp_path: Path, amsgrad_everywhere: bool, explicit_hparams: bool
) -> None:
    def groups(named: dict[str, nn.Parameter], order: list[str]) -> list[dict[str, Any]]:
        half = len(order) // 2
        return [
            {"params": [named[n] for n in order[:half]], "amsgrad": True},
            {"params": [named[n] for n in order[half:]], "amsgrad": amsgrad_everywhere},
        ]

    run_dir = _real_adamw_bundle(tmp_path, BundleTinyLM(seed=0), param_groups=groups)
    source = load_local_bundle(run_dir, tiny_lm_factory)
    overrides = {"beta2": 0.999, "eps": 1e-8} if explicit_hparams else {}
    with pytest.raises(ValueError, match="amsgrad=True"):
        source.adamw_preconditioner(**overrides)(source.checkpoints[0].identifier)


@pytest.mark.parametrize("optimizer_cls", [torch.optim.RAdam, torch.optim.NAdam])
def test_non_adam_optimizer_state_is_not_labelled_adamw(
    tmp_path: Path, optimizer_cls: type[torch.optim.Optimizer]
) -> None:
    run_dir = _real_adamw_bundle(
        tmp_path,
        BundleTinyLM(seed=0),
        optimizer_cls=optimizer_cls,
        metadata_optimizer=optimizer_cls.__name__.lower(),
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    with pytest.raises(BundleSchemaError, match="names the optimizer"):
        source.adamw_preconditioner()(source.checkpoints[0].identifier)


# ------------------------------- which optimizer wrote the state: groups, not names

NAME_VARIANTS = ("no name", "optimizer_class", "optimizer")
ADAM_BETAS, ADAM_EPS = (0.9, 0.95), 1e-8  # what ``bundle_metadata`` records by default


def _bundle_written_by(
    tmp_path: Path,
    optimizer_cls: type[torch.optim.Optimizer],
    *,
    name_variant: str,
    n_steps: int,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> Path:
    """A bundle holding a real ``optimizer_cls`` state; ``metadata.json`` names the
    optimizer under ``optimizer`` (lower case), under ``optimizer_class`` only
    (the class name, as the project's bundle writer records it), or not at all."""
    assert name_variant in NAME_VARIANTS
    run_dir = _real_adamw_bundle(
        tmp_path,
        BundleTinyLM(seed=0),
        optimizer_cls=optimizer_cls,
        metadata_optimizer=optimizer_cls.__name__.lower(),
        n_steps=n_steps,
        optimizer_kwargs=optimizer_kwargs,
    )
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["optimizer"] == optimizer_cls.__name__.lower()
    if name_variant != "optimizer":
        del metadata["optimizer"]
    if name_variant == "optimizer_class":
        metadata["optimizer_class"] = optimizer_cls.__name__
    metadata_path.write_text(json.dumps(metadata))
    return run_dir


@pytest.mark.parametrize(
    ("optimizer_cls", "expected"),
    [
        (torch.optim.Adam, "adam"),
        (torch.optim.AdamW, "adam"),
        (torch.optim.SGD, "sgd"),
        (torch.optim.RAdam, None),
        (torch.optim.NAdam, None),
        (torch.optim.Adamax, None),
        (torch.optim.RMSprop, None),
        (torch.optim.Adagrad, None),
    ],
)
@pytest.mark.parametrize("n_steps", [0, 2])
def test_the_optimizer_family_is_read_off_real_param_groups(
    optimizer_cls: type[torch.optim.Optimizer], expected: str | None, n_steps: int
) -> None:
    """``betas`` does not identify Adam: RAdam, NAdam and Adamax groups carry it too."""
    torch.manual_seed(0)
    model = BundleTinyLM(seed=0)
    optimizer = optimizer_cls(model.parameters(), lr=1e-3)
    ids = torch.from_numpy(make_batch_a(seed=0)).long()
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        _lm_loss(model, ids).backward()
        optimizer.step()
    state = OptimizerStateView(
        optimizer.state_dict(), ParameterLayout.from_named_parameters(model), source="<live>"
    )
    assert _optimizer_family(state) == expected
    if optimizer_cls in (torch.optim.RAdam, torch.optim.NAdam, torch.optim.Adamax):
        # the case this test exists for: a non-Adam group that has 'betas'
        assert "betas" in optimizer.state_dict()["param_groups"][0]


@pytest.mark.parametrize(
    "optimizer_cls",
    [torch.optim.RAdam, torch.optim.NAdam, torch.optim.Adamax, torch.optim.RMSprop],
)
@pytest.mark.parametrize("name_variant", NAME_VARIANTS)
@pytest.mark.parametrize("n_steps", [0, 3])
def test_a_non_adam_state_gets_no_adamw_rows_whatever_the_metadata_names(
    tmp_path: Path,
    optimizer_cls: type[torch.optim.Optimizer],
    name_variant: str,
    n_steps: int,
) -> None:
    """Refused by both P sources, at step 0 too, with or without a recorded name."""
    run_dir = _bundle_written_by(
        tmp_path, optimizer_cls, name_variant=name_variant, n_steps=n_steps
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    ident = source.checkpoints[0].identifier
    with pytest.raises(BundleSchemaError):
        source.adamw_preconditioner()(ident)
    with pytest.raises(BundleSchemaError, match="cannot tell which geometry factor"):
        source.preconditioner()(ident)


@pytest.mark.parametrize("optimizer_cls", [torch.optim.RAdam, torch.optim.NAdam])
def test_an_unnamed_radam_or_nadam_state_is_refused_because_of_its_param_groups(
    tmp_path: Path, optimizer_cls: type[torch.optim.Optimizer]
) -> None:
    """The reproduced defect: nothing in ``metadata.json`` names the optimizer, the
    state has ``betas``, ``eps``, ``exp_avg_sq`` and ``step`` — and is not AdamW's."""
    run_dir = _bundle_written_by(tmp_path, optimizer_cls, name_variant="no name", n_steps=3)
    source = load_local_bundle(run_dir, tiny_lm_factory)
    checkpoint = source.checkpoints[0]
    assert source.metadata.optimizer_name is None
    state = checkpoint.optimizer_state
    assert state.has_state("exp_avg_sq") and state.step_count == 3
    assert {"betas", "eps"} <= set(state.hparams)
    with pytest.raises(BundleSchemaError, match="not those of torch.optim.Adam / AdamW"):
        source.adamw_preconditioner()(checkpoint.identifier)


@pytest.mark.parametrize("optimizer_cls", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("name_variant", NAME_VARIANTS)
def test_adam_and_adamw_states_are_accepted_whatever_the_metadata_names(
    tmp_path: Path, optimizer_cls: type[torch.optim.Optimizer], name_variant: str
) -> None:
    n_steps = 3
    run_dir = _bundle_written_by(
        tmp_path,
        optimizer_cls,
        name_variant=name_variant,
        n_steps=n_steps,
        optimizer_kwargs={"betas": ADAM_BETAS, "eps": ADAM_EPS},
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    checkpoint = source.checkpoints[0]
    saved = torch.load(checkpoint.path / "optim.pt", weights_only=True)["state"]
    v = torch.cat([saved[i]["exp_avg_sq"].reshape(-1) for i in range(len(source.layout))])
    by_hand = 1.0 / ((v / (1.0 - ADAM_BETAS[1] ** n_steps)).sqrt() + ADAM_EPS)
    for factory in (source.adamw_preconditioner(), source.preconditioner()):
        precond = factory(checkpoint.identifier)
        assert precond is not None
        assert precond.descriptor == "adamw"
        assert torch.equal(precond.diag, by_hand)


@pytest.mark.parametrize("optimizer_cls", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("name_variant", NAME_VARIANTS)
def test_an_adam_or_adamw_step_zero_checkpoint_is_still_skipped_with_a_warning(
    tmp_path: Path, optimizer_cls: type[torch.optim.Optimizer], name_variant: str
) -> None:
    run_dir = _bundle_written_by(tmp_path, optimizer_cls, name_variant=name_variant, n_steps=0)
    source = load_local_bundle(run_dir, tiny_lm_factory)
    for factory in (source.adamw_preconditioner(), source.preconditioner()):
        with pytest.warns(UserWarning, match="SKIPPED"):
            assert factory(source.checkpoints[0].identifier) is None


@pytest.mark.parametrize("name_variant", NAME_VARIANTS)
@pytest.mark.parametrize("n_steps", [0, 3])
def test_an_sgd_state_is_recognised_whatever_the_metadata_names(
    tmp_path: Path, name_variant: str, n_steps: int
) -> None:
    run_dir = _bundle_written_by(
        tmp_path,
        torch.optim.SGD,
        name_variant=name_variant,
        n_steps=n_steps,
        optimizer_kwargs={"momentum": 0.9},
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    ident = source.checkpoints[0].identifier
    precond = source.preconditioner()(ident)
    assert precond is not None
    assert precond.descriptor == "sgd"
    assert torch.equal(precond.diag, torch.ones(source.layout.total_numel))
    with pytest.raises(BundleSchemaError, match="SGD optimizer"):
        source.adamw_preconditioner()(ident)


def test_optimizer_name_also_reads_optimizer_class() -> None:
    """``optimizer_class`` is the key the project's own bundle writer records."""
    name = lambda raw: BundleMetadata(raw=raw).optimizer_name  # noqa: E731
    assert name({"optimizer_class": "AdamW"}) == "AdamW"
    assert name({"optimizer_class": "torch.optim.radam.RAdam"}) == "RAdam"
    assert name({}) is None
    # the keys read before keep their precedence and their values
    assert name({"optimizer": "adamw", "optimizer_class": "RAdam"}) == "adamw"
    assert name({"optimizer_name": "sgd", "optimizer_class": "AdamW"}) == "sgd"
    assert name({"optimizer_type": "adam", "optimizer": "sgd"}) == "adam"
    assert name({"optimizer": {"name": "nadam", "lr": 1e-3}}) == "nadam"
    assert name({"optimizer": {"type": "adamax"}}) == "adamax"
    # a dict without a name falls through to the class
    assert name({"optimizer": {"lr": 1e-3}, "optimizer_class": "AdamW"}) == "AdamW"
    assert name({"optimizer": {"lr": 1e-3}}) is None


def test_a_recorded_optimizer_class_other_than_adam_is_refused_by_name(tmp_path: Path) -> None:
    """With ``optimizer_class`` read, the metadata cross-check sees the writer's own
    record: an AdamW state under metadata that says ``"RAdam"`` do not belong together."""
    run_dir = _bundle_written_by(tmp_path, torch.optim.AdamW, name_variant="no name", n_steps=3)
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["optimizer_class"] = "RAdam"
    metadata_path.write_text(json.dumps(metadata))
    source = load_local_bundle(run_dir, tiny_lm_factory)
    with pytest.raises(BundleSchemaError, match="names the optimizer 'RAdam'"):
        source.adamw_preconditioner()(source.checkpoints[0].identifier)


# ----------------------------------------------------- per-group learning rates


def _two_group_bundle(
    tmp_path: Path,
    optimizer_cls: type[torch.optim.Optimizer],
    rates: tuple[float, float],
    *,
    bundled_steps: tuple[int, ...] = (3,),
    scheduled: bool = False,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> Path:
    """A real optimizer with two param groups (first / second half of the
    parameters) at the given base rates. With ``scheduled`` one ``LambdaLR``
    steps both groups, so the saved ``lr`` changes from checkpoint to checkpoint.
    Checkpoints are written pre-step."""
    torch.manual_seed(0)
    model = BundleTinyLM(seed=0)
    run_dir = tmp_path / "run"
    name = "sgd" if optimizer_cls is torch.optim.SGD else "adamw"
    order = write_run_files(run_dir, model, metadata=bundle_metadata(model, optimizer=name))
    named = dict(model.named_parameters())
    half = len(order) // 2
    optimizer = optimizer_cls(
        [
            {"params": [named[n] for n in order[:half]], "lr": rates[0]},
            {"params": [named[n] for n in order[half:]], "lr": rates[1]},
        ],
        **(optimizer_kwargs or {}),
    )
    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5**step) if scheduled else None
    )
    ids = torch.from_numpy(make_batch_a(seed=0)).long()
    for step in range(max(bundled_steps) + 1):
        if step in bundled_steps:
            write_checkpoint(
                run_dir,
                step,
                model=model,
                optim_state=optimizer.state_dict(),
                batch_a=make_batch_a(seed=step),
            )
        optimizer.zero_grad(set_to_none=True)
        _lm_loss(model, ids).backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return run_dir


def _saved_rates(source: Any, step: int) -> list[float]:
    return [group["lr"] for group in source.checkpoint_for(step).optimizer_state.param_groups]


ADAMW_KW = {"betas": ADAM_BETAS, "eps": ADAM_EPS}
PER_GROUP_LR = "different learning rates.*Preconditioner.from_diagonal"


@pytest.mark.parametrize("step", [0, 3])
def test_adamw_groups_with_unequal_rates_are_refused(tmp_path: Path, step: int) -> None:
    """``P`` leaves the learning rate out; that is only sound when there is one."""
    run_dir = _two_group_bundle(
        tmp_path, torch.optim.AdamW, (1e-3, 1e-4), bundled_steps=(step,), optimizer_kwargs=ADAMW_KW
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    assert _saved_rates(source, step) == [1e-3, 1e-4]
    ident = source.checkpoint_for(step).identifier
    for factory in (
        source.adamw_preconditioner(),
        source.preconditioner(),
        # explicit beta2 / eps must not be a way around the check
        source.adamw_preconditioner(beta2=0.95, eps=1e-8),
        source.preconditioner(beta2=0.95, eps=1e-8),
        source.adamw_preconditioner(on_step_mismatch="warn"),
    ):
        with pytest.raises(ValueError, match=PER_GROUP_LR):
            factory(ident)


def test_adamw_groups_with_equal_rates_are_accepted(tmp_path: Path) -> None:
    """Two groups as such are fine (e.g. to exempt biases from weight decay)."""
    torch.manual_seed(0)
    run_dir = _two_group_bundle(
        tmp_path, torch.optim.AdamW, (1e-3, 1e-3), bundled_steps=(0, 3), optimizer_kwargs=ADAMW_KW
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    assert len(source.checkpoint_for(3).optimizer_state.param_groups) == 2
    with pytest.warns(UserWarning, match="SKIPPED"):
        assert source.adamw_preconditioner()(source.checkpoint_for(0).identifier) is None
    saved = torch.load(source.checkpoint_for(3).path / "optim.pt", weights_only=True)["state"]
    v = torch.cat([saved[i]["exp_avg_sq"].reshape(-1) for i in range(len(source.layout))])
    by_hand = 1.0 / ((v / (1.0 - ADAM_BETAS[1] ** 3)).sqrt() + ADAM_EPS)
    for factory in (source.adamw_preconditioner(), source.preconditioner()):
        precond = factory(source.checkpoint_for(3).identifier)
        assert precond is not None and precond.descriptor == "adamw"
        assert torch.equal(precond.diag, by_hand)


@pytest.mark.parametrize("momentum", [0.0, 0.9])
@pytest.mark.parametrize("step", [0, 3])
def test_sgd_groups_with_unequal_rates_are_refused(
    tmp_path: Path, momentum: float, step: int
) -> None:
    """With per-group rates ``P = I`` is not SGD's geometry either."""
    run_dir = _two_group_bundle(
        tmp_path,
        torch.optim.SGD,
        (1e-2, 1e-3),
        bundled_steps=(step,),
        optimizer_kwargs={"momentum": momentum},
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    assert _saved_rates(source, step) == [1e-2, 1e-3]
    with pytest.raises(ValueError, match=PER_GROUP_LR):
        source.preconditioner()(source.checkpoint_for(step).identifier)


@pytest.mark.parametrize("step", [0, 3])
def test_sgd_groups_with_equal_rates_get_the_identity(tmp_path: Path, step: int) -> None:
    run_dir = _two_group_bundle(
        tmp_path,
        torch.optim.SGD,
        (1e-2, 1e-2),
        bundled_steps=(step,),
        optimizer_kwargs={"momentum": 0.9},
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    assert len(source.checkpoint_for(step).optimizer_state.param_groups) == 2
    precond = source.preconditioner()(source.checkpoint_for(step).identifier)
    assert precond is not None and precond.descriptor == "sgd"
    assert torch.equal(precond.diag, torch.ones(source.layout.total_numel))


@pytest.mark.parametrize("optimizer_cls", [torch.optim.AdamW, torch.optim.SGD])
def test_one_schedule_stepping_both_groups_keeps_their_saved_rates_equal(
    tmp_path: Path, optimizer_cls: type[torch.optim.Optimizer]
) -> None:
    """The saved ``lr`` is the scheduled value at that step. Groups that share a
    schedule and a base rate agree on it at every checkpoint, whatever it is."""
    kwargs = ADAMW_KW if optimizer_cls is torch.optim.AdamW else {"momentum": 0.9}
    steps = (1, 2, 4)
    run_dir = _two_group_bundle(
        tmp_path,
        optimizer_cls,
        (1e-2, 1e-2),
        bundled_steps=steps,
        scheduled=True,
        optimizer_kwargs=kwargs,
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    seen = []
    for step in steps:
        rates = _saved_rates(source, step)
        assert rates[0] == rates[1] == pytest.approx(1e-2 * 0.5**step)
        seen.append(rates[0])
        precond = source.preconditioner()(source.checkpoint_for(step).identifier)
        assert precond is not None
        assert precond.descriptor == ("adamw" if optimizer_cls is torch.optim.AdamW else "sgd")
    assert len(set(seen)) == len(steps)  # the schedule really moved the rate


@pytest.mark.parametrize("optimizer_cls", [torch.optim.AdamW, torch.optim.SGD])
def test_a_schedule_over_groups_with_different_base_rates_is_refused_at_every_checkpoint(
    tmp_path: Path, optimizer_cls: type[torch.optim.Optimizer]
) -> None:
    kwargs = ADAMW_KW if optimizer_cls is torch.optim.AdamW else {"momentum": 0.9}
    steps = (1, 2, 4)
    run_dir = _two_group_bundle(
        tmp_path,
        optimizer_cls,
        (1e-2, 1e-3),
        bundled_steps=steps,
        scheduled=True,
        optimizer_kwargs=kwargs,
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    for step in steps:
        rates = _saved_rates(source, step)
        assert rates[0] == pytest.approx(10.0 * rates[1])
        with pytest.raises(ValueError, match=PER_GROUP_LR):
            source.preconditioner()(source.checkpoint_for(step).identifier)


# ------------------------------------------------ where the optimizer state lives


def test_optim_pt_is_read_onto_the_cpu_whatever_the_source_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GPU source must not put ``exp_avg`` / ``exp_avg_sq`` next to the model.

    Runs without a GPU: the source is only *told* its device is CUDA, and the
    ``map_location`` handed to ``torch.load`` is recorded. (Reading with
    ``map_location="cuda"`` fails outright on a machine without CUDA.)
    """
    run_dir = _real_adamw_bundle(tmp_path, BundleTinyLM(seed=0))
    source = load_local_bundle(run_dir, tiny_lm_factory, device="cuda")
    assert source.device.type == "cuda"
    seen: list[Any] = []
    real_load = torch.load

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("map_location"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    state = source.checkpoints[0].optimizer_state
    assert len(seen) == 1
    assert torch.device(seen[0]).type == "cpu"
    assert state.flat_state("exp_avg_sq").device.type == "cpu"
    assert state.flat_state("exp_avg").device.type == "cpu"


@pytest.mark.gpu
def test_a_gpu_source_keeps_the_state_on_the_cpu_and_returns_p_on_the_gpu(
    tmp_path: Path,
) -> None:
    run_dir = _real_adamw_bundle(tmp_path, BundleTinyLM(seed=0))
    on_cpu = load_local_bundle(run_dir, tiny_lm_factory, device="cpu")
    on_gpu = load_local_bundle(run_dir, tiny_lm_factory, device="cuda")
    ident = on_gpu.checkpoints[0].identifier

    state = on_gpu.checkpoints[0].optimizer_state
    assert state.flat_state("exp_avg_sq").device.type == "cpu"
    # ``flat_state(device=...)`` still places the vector where it is asked to
    assert state.flat_state("exp_avg_sq", device="cuda").device.type == "cuda"

    precond = on_gpu.adamw_preconditioner()(ident)
    assert precond is not None
    assert precond.device.type == "cuda"
    reference = on_cpu.adamw_preconditioner()(ident)
    assert reference is not None and reference.device.type == "cpu"
    assert torch.equal(precond.diag.cpu(), reference.diag)
    unified = on_gpu.preconditioner()(ident)
    assert unified is not None and unified.device.type == "cuda"


# ------------------------------------------------- one P source for both optimizers


def _sgd_bundle(tmp_path: Path, *, momentum: float, n_steps: int) -> Path:
    return _real_adamw_bundle(
        tmp_path,
        BundleTinyLM(seed=0),
        optimizer_cls=torch.optim.SGD,
        metadata_optimizer="sgd",
        n_steps=n_steps,
        optimizer_kwargs={"momentum": momentum, "weight_decay": 0.1},
    )


@pytest.mark.parametrize("momentum", [0.0, 0.9])
@pytest.mark.parametrize("n_steps", [0, 3])
def test_sgd_bundle_gets_identity_rows_under_the_sgd_descriptor(
    tmp_path: Path, momentum: float, n_steps: int
) -> None:
    """SGD's geometry factor is ``P = I``: its P rows repeat the raw ones exactly.

    Step 0 included -- unlike AdamW, SGD needs no saved state to have a ``P``.
    """
    source = load_local_bundle(
        _sgd_bundle(tmp_path, momentum=momentum, n_steps=n_steps), tiny_lm_factory, dtype="fp32"
    )
    val = torch.from_numpy(make_batch_a(seed=999)).long()
    analyzer = Analyzer(
        eval_batches={
            "train": source.batch_a_spec(),
            "val": {"input_ids": val, "labels": shifted_lm_targets(val)},
        },
        preconditioner=source.preconditioner(),
        cross_pairs=[("train", "val")],
        chi_net_method="hutchinson",
        n_hutchinson=4,
        micro_batch_size=2,
        dtype="fp32",
        sink=None,
    )
    rows = [r for result in analyzer.run(source.model_specs()) for r in result.rows]
    values = {(r.batch_a, r.batch_b, r.observable): r.value for r in rows}
    descriptors = {r.observable: r.preconditioner for r in rows}
    for name in ("delta_loss_P", "chi_net_P", "chi_pos_P"):
        assert descriptors[name] == "sgd"
        assert descriptors[name.removesuffix("_P")] == "none"
    for pair in (("train", "train"), ("val", "val"), ("train", "val")):
        for name in ("delta_loss", "chi_net", "chi_pos"):
            assert values[(*pair, f"{name}_P")] == values[(*pair, name)], (pair, name)
    assert values[("train", "train", "delta_loss")] > 0.0


@pytest.mark.parametrize("momentum", [0.0, 0.9])
@pytest.mark.parametrize("n_steps", [0, 3])
def test_the_adamw_source_names_an_sgd_bundle_as_such(
    tmp_path: Path, momentum: float, n_steps: int
) -> None:
    """Not the step-0 "empty state" skip: plain SGD has an empty state at every step."""
    source = load_local_bundle(
        _sgd_bundle(tmp_path, momentum=momentum, n_steps=n_steps), tiny_lm_factory
    )
    with pytest.raises(BundleSchemaError, match="SGD optimizer"):
        source.adamw_preconditioner()(source.checkpoints[0].identifier)


def test_the_unified_source_returns_the_adamw_p_for_an_adamw_bundle(
    trained_bundle: dict[str, Any],
) -> None:
    source = load_local_bundle(trained_bundle["run_dir"], tiny_lm_factory, dtype="fp32")
    ident = source.checkpoint_for(BUNDLED_STEPS[0]).identifier
    direct = source.adamw_preconditioner()(ident)
    unified = source.preconditioner()(ident)
    assert direct is not None and unified is not None
    assert unified.descriptor == "adamw"
    assert torch.equal(unified.diag, direct.diag)
    assert not torch.equal(unified.diag, torch.ones_like(unified.diag))


def test_the_unified_source_refuses_what_it_cannot_name(tmp_path: Path) -> None:
    # RMSprop also keeps a second moment and has a ``momentum`` key, but it is
    # neither Adam nor SGD: it must not come back as "adamw" or as "sgd".
    run_dir = _real_adamw_bundle(
        tmp_path,
        BundleTinyLM(seed=0),
        optimizer_cls=torch.optim.RMSprop,
        metadata_optimizer="rmsprop",
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    with pytest.raises(BundleSchemaError, match="cannot tell which geometry factor"):
        source.preconditioner()(source.checkpoints[0].identifier)


def test_sgd_state_under_metadata_that_names_adamw_is_refused(tmp_path: Path) -> None:
    run_dir = _real_adamw_bundle(
        tmp_path, BundleTinyLM(seed=0), optimizer_cls=torch.optim.SGD, metadata_optimizer="adamw"
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    with pytest.raises(BundleSchemaError, match="do not belong together"):
        source.preconditioner()(source.checkpoints[0].identifier)


def test_adamw_overrides_are_refused_for_an_sgd_bundle(tmp_path: Path) -> None:
    source = load_local_bundle(_sgd_bundle(tmp_path, momentum=0.9, n_steps=3), tiny_lm_factory)
    with pytest.raises(ValueError, match="AdamW overrides"):
        source.preconditioner(eps=1e-6)(source.checkpoints[0].identifier)


def test_an_override_is_visible_in_the_descriptor(trained_bundle: dict[str, Any]) -> None:
    source = load_local_bundle(trained_bundle["run_dir"], tiny_lm_factory, dtype="fp32")
    ident = source.checkpoint_for(BUNDLED_STEPS[0]).identifier
    default = source.adamw_preconditioner()(ident)
    overridden = source.adamw_preconditioner(eps=1e-3)(ident)
    assert default is not None and overridden is not None
    assert default.descriptor == "adamw"
    assert overridden.descriptor == "adamw(eps=0.001)"
    assert not torch.equal(default.diag, overridden.diag)


# ---------------------------------------------------------------- reductions


def test_weighted_reductions_agree_with_the_explicit_sum_across_chunk_boundaries() -> None:
    """Above ``2**25`` parameters only the multi-chunk path runs; pin it at a small chunk."""
    gen = torch.Generator().manual_seed(11)
    a = torch.randn(1000, generator=gen)
    b = torch.randn(1000, generator=gen)
    w = torch.rand(1000, generator=gen) * 5.0
    dot = (w.double() * a.double() * b.double()).sum()
    sq = (w.double() * a.double() * a.double()).sum()
    for chunk in (1, 97, 1000, 4096):
        assert float(weighted_dot_fp64(a, b, w, chunk=chunk)) == pytest.approx(
            float(dot), rel=1e-12
        )
        assert float(weighted_sum_sq_fp64(a, w, chunk=chunk)) == pytest.approx(float(sq), rel=1e-12)


# ------------------------------------------------------- the range backstop


def _toy_bundle(loss_fn: Any) -> ModelBundle:
    torch.manual_seed(0)
    model = TinyTransformer(
        vocab_size=16, seq_len=8, d_model=16, n_layers=1, n_heads=2, d_ff=32
    ).eval()
    return ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=lambda m, b: m(b["input_ids"]),
        loss_fn=loss_fn,
        valid_mask_fn=causal_lm_valid_mask,
        identifier="toy@step0",
    )


def _sum_reduced_ce(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch["labels"].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


@pytest.mark.parametrize("micro_batch_size", [1, 2, 4, 8])
def test_a_really_sum_reduced_loss_inflates_by_the_per_micro_batch_count_squared(
    micro_batch_size: int,
) -> None:
    """Not by the global ``N**2``: the analyzer reassembles the gradient from
    micro-batch losses, so a ``reduction="sum"`` loss scales each micro-batch's
    contribution by *its own* valid-token count. The backstop therefore sees a
    factor that depends on ``micro_batch_size`` — 49 at the default 1 on this
    batch, 3136 only when the whole batch is one micro-batch.
    """
    batch = make_tiny_lm_batch(batch_size=8, seq_len=8, vocab_size=16, seed=9)
    per_sequence = 7  # seq_len 8, last position has no target

    def chi_pos(loss_fn: Any) -> float:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = analyze(
                model=_toy_bundle(loss_fn),
                revisions=["step0"],
                eval_batches={"lm": batch},
                chi_net_method="hutchinson",
                n_hutchinson=4,
                micro_batch_size=micro_batch_size,
                dtype="fp32",
                sink=None,
            )[0].rows
        return next(r.value for r in rows if r.observable == "chi_pos")

    ratio = chi_pos(_sum_reduced_ce) / chi_pos(causal_lm_loss)
    assert ratio == pytest.approx((per_sequence * micro_batch_size) ** 2, rel=1e-4)


def test_the_range_warning_also_covers_chi_pos_p() -> None:
    batch = make_tiny_lm_batch(batch_size=8, seq_len=8, vocab_size=16, seed=9)
    with pytest.warns(UserWarning) as caught:
        analyze(
            model=_toy_bundle(_sum_reduced_ce),
            revisions=["step0"],
            eval_batches={"lm": batch},
            preconditioner="identity",
            chi_net_method="hutchinson",
            n_hutchinson=8,
            micro_batch_size=8,
            dtype="fp32",
            sink=None,
        )
    messages = [str(w.message) for w in caught]
    assert any(m.startswith("chi_pos = ") for m in messages)
    assert any(m.startswith("chi_pos_P = ") for m in messages)


# ------------------------------------------------- cross-gradient storage


@pytest.mark.gpu
def test_auto_cross_grad_storage_follows_the_models_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bundle source places its own model; ``Analyzer.device`` may still be the
    ``"cpu"`` default. The offload heuristic has to look at the model."""
    consulted: list[bool] = []

    def always_offload(**kwargs: Any) -> bool:
        consulted.append(True)
        return True

    monkeypatch.setattr(analyzer_mod, "_should_offload_cross_grads", always_offload)
    torch.manual_seed(0)
    model = TinyTransformer(
        vocab_size=16, seq_len=8, d_model=16, n_layers=1, n_heads=2, d_ff=32
    ).to("cuda")
    bundle = ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=lambda m, b: m(b["input_ids"]),
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        identifier="toy@step0",
    )
    batch = make_tiny_lm_batch(batch_size=2, seq_len=8, vocab_size=16, seed=1)
    analyzer = Analyzer(eval_batches={"a": batch, "b": batch}, cross_pairs=[("a", "b")], sink=None)
    assert analyzer.device.type == "cpu"
    assert analyzer._resolve_cross_grad_storage(bundle) == "cpu"
    assert consulted
