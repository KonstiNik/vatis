"""Unit tests for the native checkpoint-bundle loader (``vatis.models.local_bundle``).

The four tests the WP0 spec (``02_vatis_loader.md``) declares:

- **T-roundtrip** — :func:`test_roundtrip_theta_optimizer_and_metadata`
- **T-alignment** — :func:`test_alignment_flat_state_matches_gradient_flattening`
  and :func:`test_alignment_against_real_adamw_state`
- **T-lazy** — :func:`test_lazy_iteration_holds_one_checkpoint_at_a_time`
- **T-factory** — :func:`test_import_string_factory_matches_callable_factory`

plus the loud-failure tests for the mis-alignment modes the loader must reject,
and one end-to-end check that the source drives the existing ``Analyzer``.

All bundles are written by hand (``tests/fixtures/synthetic_bundle.py``), which
imports none of the loader's code — so an ordering bug in the loader cannot
cancel against the same bug in the fixture.
"""

from __future__ import annotations

import gc
import json
import math
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tests.fixtures.synthetic_bundle import (
    DEFAULT_D_MODEL,
    DEFAULT_SEQ_LEN,
    DEFAULT_VOCAB_SIZE,
    BundleTinyLM,
    bf16_reference_weights,
    bundle_metadata,
    handwritten_optim_state,
    make_batch_a,
    named_parameter_order,
    permuted_parameter_order,
    run_adamw_steps,
    tiny_lm_factory,
    write_checkpoint,
    write_run_files,
    write_simple_bundle,
    wrong_factory,
)
from vatis.analyzer import Analyzer, _add_grads_into_flat, _zero_param_vector
from vatis.core.observables import parameter_grad_vector
from vatis.core.param_layout import LayoutMismatchError, ParameterLayout
from vatis.models.local_bundle import (
    BundleMetadata,
    BundleSchemaError,
    load_local_bundle,
    resolve_model_factory,
)

# Exact types only: an ``isinstance`` check would walk ``obj.__class__`` on
# every object the gc knows about, which trips deprecation shims in unrelated
# libraries.
_TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)


def _live_tensor_bytes(min_numel: int = 4096) -> int:
    """Total bytes of the "large" torch tensors currently reachable.

    The T-lazy weak check: scan the garbage collector for live tensors and sum
    their footprint. Tensors below ``min_numel`` are ignored so unrelated small
    tensors in the process don't add noise; the bundle's weight / ``exp_avg`` /
    ``exp_avg_sq`` tensors are far above the threshold.
    """
    gc.collect()
    total = 0
    for obj in gc.get_objects():
        if type(obj) in _TENSOR_TYPES and obj.numel() >= min_numel:
            total += obj.numel() * obj.element_size()
    return total


# --------------------------------------------------------------- T-roundtrip


def test_roundtrip_theta_optimizer_and_metadata(tmp_path: Path) -> None:
    """T-roundtrip: theta exact, optimizer tensors exact, metadata present."""
    run_dir = tmp_path / "roundtrip_run"
    probe = BundleTinyLM()
    order = named_parameter_order(probe)
    constants = {name: 3.0 + 2.0 * i for i, name in enumerate(order)}
    model, written_order = write_simple_bundle(
        run_dir, steps=(0, 10), param_order=order, exp_avg_sq_constants=constants
    )

    source = load_local_bundle(run_dir, tiny_lm_factory)

    # The bundle-level pieces.
    assert source.layout.names == tuple(written_order)
    assert source.layout.total_numel == sum(p.numel() for p in model.parameters())
    assert source.steps == (0, 10)
    assert len(source) == 2

    checkpoint = source.checkpoints[1]
    assert checkpoint.step == 10
    assert checkpoint.name == "step_00000010"
    assert checkpoint.revision == "step10"
    assert checkpoint.identifier == "roundtrip_run@step10"

    # --- theta: bf16 on disk, fp32 in memory, bit-identical values ---
    bundle = checkpoint.model_bundle
    reference = bf16_reference_weights(model)
    loaded = dict(bundle.model.named_parameters())
    assert set(loaded) == set(reference)
    for name, expected in reference.items():
        assert loaded[name].dtype is torch.float32
        assert torch.equal(loaded[name], expected), f"theta mismatch for {name}"
    # Weights really came from the file (not the factory's own init).
    fresh = tiny_lm_factory(source.metadata)
    fresh_named = dict(fresh.named_parameters())
    assert not torch.equal(loaded["fc1.weight"], fresh_named["fc1.weight"])
    assert all(p.requires_grad for p in bundle.params)
    assert not bundle.model.training

    # --- optimizer tensors: fp32 on disk, exact after the round trip ---
    state = checkpoint.optimizer_state
    assert state.state_keys() == ("exp_avg", "exp_avg_sq", "step")
    assert state.step_count == 10
    assert not state.is_empty
    flat_sq = state.flat_state("exp_avg_sq")
    flat_avg = state.flat_state("exp_avg")
    assert flat_sq.dtype is torch.float32
    assert flat_sq.numel() == source.layout.total_numel
    for name, constant in constants.items():
        segment = source.layout.segment(flat_sq, name)
        assert torch.equal(segment, torch.full_like(segment, constant))
    assert torch.equal(flat_avg, torch.zeros_like(flat_avg))
    assert state.scalar_state("step") == [10.0] * len(source.layout)
    assert state.hparams["eps"] == pytest.approx(1e-8)
    assert state.hparams["betas"] == (0.9, 0.95)
    assert state.hparams["weight_decay"] == pytest.approx(0.1)
    assert "params" not in state.hparams
    assert state.param_groups[0]["params"] == list(range(len(source.layout)))

    # --- metadata.json fields ---
    metadata = source.metadata
    assert metadata.optimizer_name == "adamw"
    hparams = metadata.optimizer_hparams
    assert hparams is not None
    assert hparams["betas"] == [0.9, 0.95]
    assert hparams["eps"] == pytest.approx(1e-8)
    assert metadata.model_config == {
        "vocab_size": DEFAULT_VOCAB_SIZE,
        "d_model": DEFAULT_D_MODEL,
        "seq_len": DEFAULT_SEQ_LEN,
    }
    assert metadata.seed == 0
    assert metadata.batch_size_train == 3
    assert metadata.seq_len == DEFAULT_SEQ_LEN
    assert metadata.dataset == "synthetic-random-tokens"
    assert metadata.git_hashes == {"antonio_code": "0" * 40, "vatis": "1" * 40}
    assert metadata.lr_schedule is not None
    assert metadata["model_class"] == "BundleTinyLM"

    # --- meta.json fields ---
    meta = checkpoint.meta
    assert meta.step == 10
    assert meta.tokens_seen == 10 * 3 * DEFAULT_SEQ_LEN
    assert meta.lr == pytest.approx(1e-3)
    assert meta.train_loss == pytest.approx(2.4)
    assert meta.wall_time == pytest.approx(5.0)

    # --- batch_a: int32 on disk, int64 ids + pre-shifted labels in memory ---
    on_disk = np.load(checkpoint.path / "batch_a.npy")
    assert on_disk.dtype == np.int32
    tokens = checkpoint.batch_a_tokens
    assert tokens.dtype is torch.int64
    assert torch.equal(tokens, torch.from_numpy(on_disk).to(torch.int64))
    batch = checkpoint.batch_a
    assert set(batch) == {"input_ids", "labels"}
    assert torch.equal(batch["labels"][:, :-1], tokens[:, 1:])
    assert torch.all(batch["labels"][:, -1] == -100)


def test_pre_step_semantics_are_preserved_per_checkpoint(tmp_path: Path) -> None:
    """theta_t, the *pre-step* optimizer state and batch_a stay together.

    Each step gets a distinguishable optimizer state (``exp_avg_sq`` = the step
    number) and a different ``batch_a``; the loader must hand back the triple
    that belongs to the step it was asked for, never a neighbour's.
    """
    run_dir = tmp_path / "prestep_run"
    model = BundleTinyLM(seed=3)
    order = write_run_files(run_dir, model, metadata=bundle_metadata(model))
    steps = (0, 4, 9)
    for step in steps:
        write_checkpoint(
            run_dir,
            step,
            model=model,
            optim_state=handwritten_optim_state(
                model,
                order,
                step=step,
                exp_avg_sq_constants={name: float(step) + 1.0 for name in order},
            ),
            batch_a=make_batch_a(seed=100 + step),
        )

    source = load_local_bundle(run_dir, tiny_lm_factory)
    for checkpoint in source:
        state = checkpoint.optimizer_state
        # The saved counter is the number of updates already applied to
        # theta_t (pre-step), i.e. the bias-correction exponent for the saved v.
        assert state.step_count == checkpoint.step
        flat = state.flat_state("exp_avg_sq")
        assert torch.equal(flat, torch.full_like(flat, float(checkpoint.step) + 1.0))
        expected_batch = make_batch_a(seed=100 + checkpoint.step)
        assert torch.equal(
            checkpoint.batch_a_tokens, torch.from_numpy(expected_batch).to(torch.int64)
        )


# --------------------------------------------------------------- T-alignment


def test_alignment_flat_state_matches_gradient_flattening(tmp_path: Path) -> None:
    """T-alignment: optimizer state indices <-> names <-> gradient segments.

    ``param_names.json`` is written as a **permutation** of the model's own
    ``named_parameters()`` order, and every parameter's ``exp_avg_sq`` is a
    distinct constant. So:

    - a loader that flattened in ``named_parameters()`` order instead of file
      order produces a vector with the constants in the wrong segments;
    - a loader that mis-mapped state index ``i`` to a name puts the wrong
      constant in a segment;
    - a loader that built ``ModelBundle.params`` in model order makes vatis's
      gradient flattening disagree with ``flat_state``.

    All three are permutations of a same-length vector, so a length check
    cannot catch them — hence the per-name, per-segment assertions. The last
    block asserts the test is actually discriminating (the wrong vectors exist
    and differ).
    """
    run_dir = tmp_path / "alignment_run"
    probe = BundleTinyLM()
    model_order = named_parameter_order(probe)
    canonical = permuted_parameter_order(probe)
    assert canonical != model_order, "the fixture must exercise a real permutation"

    shapes = {name: tuple(p.shape) for name, p in probe.named_parameters()}
    # Same-shape pairs whose relative order the permutation flips: a shape
    # check alone cannot detect swapping them.
    assert shapes["fc1.weight"] == shapes["fc2.weight"]
    assert shapes["fc1.bias"] == shapes["fc2.bias"]
    assert canonical.index("fc1.weight") > canonical.index("fc2.weight")
    assert model_order.index("fc1.weight") < model_order.index("fc2.weight")

    constants = {name: 1000.0 + 7.0 * i for i, name in enumerate(canonical)}
    assert len(set(constants.values())) == len(constants), "constants must be distinct"

    _model, written_order = write_simple_bundle(
        run_dir, steps=(7,), param_order=canonical, exp_avg_sq_constants=constants
    )
    assert written_order == canonical
    assert json.loads((run_dir / "param_names.json").read_text())[0][0] == canonical[0]

    source = load_local_bundle(run_dir, tiny_lm_factory)
    layout = source.layout
    assert layout.names == tuple(canonical)

    checkpoint = source.checkpoints[0]
    bundle = checkpoint.model_bundle
    named = dict(bundle.model.named_parameters())

    # (1) ModelBundle.params — the list vatis flattens — follows the FILE order.
    assert [id(p) for p in bundle.params] == [id(named[name]) for name in canonical]
    layout.validate_params(bundle.params)

    # (2) flat_state segments carry each parameter's own constant.
    flat_state = checkpoint.optimizer_state.flat_state("exp_avg_sq")
    for name in canonical:
        segment = layout.segment(flat_state, name)
        assert torch.equal(segment, torch.full_like(segment, constants[name])), (
            f"exp_avg_sq segment for {name} holds {segment.unique().tolist()}, "
            f"expected {constants[name]}"
        )

    # (3) vatis's gradient flattening uses the same segments — both the
    #     observables helper and the analyzer's accumulator.
    batch = checkpoint.batch_a
    logits = bundle.forward_fn(bundle.model, batch)
    loss = bundle.loss_fn(logits, batch)
    grads = torch.autograd.grad(loss, bundle.params)
    analyzer_flat = _zero_param_vector(bundle.params, torch.device("cpu"))
    _add_grads_into_flat(analyzer_flat, grads, bundle.params)
    for param, grad in zip(bundle.params, grads, strict=True):
        param.grad = grad
    observables_flat = parameter_grad_vector(bundle.params)
    assert torch.equal(observables_flat, analyzer_flat)
    assert observables_flat.abs().sum() > 0
    for name in canonical:
        grad = named[name].grad
        assert grad is not None
        assert grad.abs().sum() > 0, f"{name} has an all-zero gradient; test is vacuous"
        assert torch.equal(layout.segment(observables_flat, name), grad.reshape(-1))

    # (4) The test discriminates: the model-order vectors are same-length but
    #     different, so any of the permutation bugs above would fail (2)/(3).
    wrong_state = torch.cat(
        [torch.full((named[name].numel(),), constants[name]) for name in model_order]
    )
    assert wrong_state.numel() == flat_state.numel()
    assert not torch.equal(wrong_state, flat_state)
    wrong_grad = torch.cat(
        [named[name].grad.reshape(-1) for name in model_order if named[name].grad is not None]
    )
    assert wrong_grad.numel() == observables_flat.numel()
    assert not torch.equal(wrong_grad, observables_flat)


def test_alignment_against_real_adamw_state(tmp_path: Path) -> None:
    """T-alignment, second half: a *real* ``torch.optim.AdamW`` state_dict.

    The index -> name mapping rests on how torch assigns ``state_dict``
    indices (enumerating ``param_groups`` in order). This test doesn't assume
    it: it runs real AdamW steps with the parameters handed over in the
    canonical (permuted) order, saves the optimizer's own ``state_dict``, and
    asserts ``flat_state`` segments equal ``optimizer.state[param][...]``
    parameter by parameter.
    """
    run_dir = tmp_path / "adamw_run"
    model = BundleTinyLM(seed=1)
    canonical = permuted_parameter_order(model)
    named = dict(model.named_parameters())
    batch = torch.from_numpy(make_batch_a(seed=11)).to(torch.int64)

    optimizer = run_adamw_steps(model, [named[name] for name in canonical], n_steps=5, batch=batch)
    write_run_files(run_dir, model, param_order=canonical, metadata=bundle_metadata(model))
    write_checkpoint(
        run_dir,
        5,
        model=model,
        optim_state=optimizer.state_dict(),
        batch_a=make_batch_a(seed=12),
    )

    source = load_local_bundle(run_dir, tiny_lm_factory)
    state = source.checkpoints[0].optimizer_state
    assert state.step_count == 5
    assert state.hparams["betas"] == (0.9, 0.95)
    assert state.hparams["eps"] == pytest.approx(1e-8)

    # Distinct per-parameter values, so a permutation would change the result.
    means = [float(optimizer.state[named[name]]["exp_avg_sq"].mean()) for name in canonical]
    assert len(set(means)) == len(means)

    for key in ("exp_avg_sq", "exp_avg"):
        flat = state.flat_state(key)
        for name in canonical:
            expected = optimizer.state[named[name]][key].detach().reshape(-1)
            assert torch.equal(source.layout.segment(flat, name), expected), (
                f"{key} segment for {name} does not match the optimizer's own tensor"
            )


def test_permuted_optimizer_param_indices_are_rejected(tmp_path: Path) -> None:
    """A ``param_groups`` index list that isn't ``range(n)`` fails loudly.

    ``param_names.json`` is the ground truth for what state index ``i`` means,
    so the loader cannot detect a writer that permuted the *contents*. What it
    can and must detect is the observable symptom: index lists that are not
    ``[0 .. n-1]`` in order (a permutation, a subset, a hole) — at which point
    ``state[i]`` is provably not ``param_names[i]``.
    """
    run_dir = tmp_path / "bad_indices_run"
    model = BundleTinyLM()
    order = write_run_files(run_dir, model, metadata=bundle_metadata(model))
    write_checkpoint(
        run_dir,
        0,
        model=model,
        optim_state=handwritten_optim_state(
            model, order, step=0, param_indices=list(reversed(range(len(order))))
        ),
        batch_a=make_batch_a(),
    )
    source = load_local_bundle(run_dir, tiny_lm_factory)
    with pytest.raises(LayoutMismatchError, match="index-aligned"):
        _ = source.checkpoints[0].optimizer_state


def test_optimizer_state_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    """A state tensor whose shape contradicts the canonical order fails loudly."""
    run_dir = tmp_path / "bad_shape_run"
    model = BundleTinyLM()
    order = write_run_files(run_dir, model, metadata=bundle_metadata(model))
    state = handwritten_optim_state(model, order, step=1)
    state["state"][0]["exp_avg_sq"] = torch.zeros(3, 3)
    write_checkpoint(run_dir, 0, model=model, optim_state=state, batch_a=make_batch_a())
    source = load_local_bundle(run_dir, tiny_lm_factory)
    with pytest.raises(LayoutMismatchError, match="canonical order says"):
        _ = source.checkpoints[0].optimizer_state


def test_flat_state_refuses_per_parameter_scalars(tmp_path: Path) -> None:
    """``flat_state("step")`` is a mistake and says so (it is not per-element)."""
    run_dir = tmp_path / "scalar_run"
    write_simple_bundle(run_dir, steps=(3,))
    state = load_local_bundle(run_dir, tiny_lm_factory).checkpoints[0].optimizer_state
    with pytest.raises(BundleSchemaError, match="scalar_state"):
        state.flat_state("step")
    with pytest.raises(BundleSchemaError, match="momentum_buffer"):
        state.flat_state("momentum_buffer")


def test_empty_optimizer_state_is_reported_not_faked(tmp_path: Path) -> None:
    """A step-0 checkpoint has no optimizer state; the loader says so."""
    run_dir = tmp_path / "empty_state_run"
    model = BundleTinyLM()
    order = write_run_files(run_dir, model, metadata=bundle_metadata(model))
    write_checkpoint(
        run_dir,
        0,
        model=model,
        optim_state={
            "state": {},
            "param_groups": [{"lr": 1e-3, "eps": 1e-8, "params": list(range(len(order)))}],
        },
        batch_a=make_batch_a(),
    )
    state = load_local_bundle(run_dir, tiny_lm_factory).checkpoints[0].optimizer_state
    assert state.is_empty
    assert state.state_keys() == ()
    assert state.has_state("exp_avg_sq") is False
    assert state.step_count is None
    with pytest.raises(BundleSchemaError, match="empty"):
        state.flat_state("exp_avg_sq")


def test_model_factory_mismatch_is_rejected(tmp_path: Path) -> None:
    """A factory that doesn't match ``param_names.json`` fails loudly."""
    run_dir = tmp_path / "mismatch_run"
    write_simple_bundle(run_dir, steps=(0,))
    source = load_local_bundle(run_dir, wrong_factory)
    with pytest.raises(LayoutMismatchError, match="has no parameter"):
        _ = source.checkpoints[0].model_bundle


def test_unlisted_model_parameters_are_rejected_unless_allowed(tmp_path: Path) -> None:
    """Parameters missing from ``param_names.json`` are an error by default.

    They would be silently absent from every parameter-space vector, so the
    default is to refuse; ``allow_unlisted_params=True`` is the documented
    opt-out and then simply does not measure them.
    """
    run_dir = tmp_path / "unlisted_run"
    probe = BundleTinyLM()
    partial = [name for name in named_parameter_order(probe) if name != "fc2.bias"]
    write_simple_bundle(run_dir, steps=(0,), param_order=partial)

    strict = load_local_bundle(run_dir, tiny_lm_factory)
    with pytest.raises(LayoutMismatchError, match="does not list"):
        _ = strict.checkpoints[0].model_bundle

    lenient = load_local_bundle(run_dir, tiny_lm_factory, allow_unlisted_params=True)
    bundle = lenient.checkpoints[0].model_bundle
    assert [name for name, _ in lenient.layout] == partial
    assert len(bundle.params) == len(partial)
    # theta is still fully loaded — the unlisted parameter is measured by
    # nothing, but it is not left at the factory's initialization.
    reference = bf16_reference_weights(BundleTinyLM())
    loaded = dict(bundle.model.named_parameters())
    assert torch.equal(loaded["fc2.bias"], reference["fc2.bias"])


def test_incomplete_checkpoint_directory_is_rejected(tmp_path: Path) -> None:
    """A checkpoint missing a schema file fails at construction, not mid-sweep."""
    run_dir = tmp_path / "incomplete_run"
    write_simple_bundle(run_dir, steps=(0, 10))
    (run_dir / "checkpoints" / "step_00000010" / "meta.json").unlink()
    with pytest.raises(BundleSchemaError, match="missing"):
        load_local_bundle(run_dir, tiny_lm_factory)


def test_meta_step_must_match_directory_name(tmp_path: Path) -> None:
    """meta.json disagreeing with the directory name is a writer bug."""
    run_dir = tmp_path / "bad_meta_run"
    model = BundleTinyLM()
    order = write_run_files(run_dir, model, metadata=bundle_metadata(model))
    ckpt_dir = write_checkpoint(
        run_dir,
        10,
        model=model,
        optim_state=handwritten_optim_state(model, order, step=10),
        batch_a=make_batch_a(),
        meta={"step": 11},
    )
    assert ckpt_dir.name == "step_00000010"
    source = load_local_bundle(run_dir, tiny_lm_factory)
    with pytest.raises(BundleSchemaError, match="directory is named"):
        _ = source.checkpoints[0].meta


# --------------------------------------------------------------------- T-lazy


def test_lazy_iteration_holds_one_checkpoint_at_a_time(tmp_path: Path) -> None:
    """T-lazy: iterating N checkpoints keeps at most one in memory.

    Three complementary weak checks (a strict guarantee is impossible — Python
    frees by reference counting, and a caller is free to retain whatever it
    likes):

    1. **Nothing is read until asked.** No checkpoint reports ``is_loaded``
       right after opening the bundle, and exactly one does during iteration.
    2. **Release is real.** ``weakref``\\ s to the previous checkpoint's model
       and to one of its parameter tensors are dead after the iterator advances.
    3. **Allocation.** The total footprint of large live torch tensors
       (:func:`_live_tensor_bytes`) never grows beyond one checkpoint's worth
       (weights + ``exp_avg`` + ``exp_avg_sq``) over five checkpoints. The
       bound below is 2x that, i.e. it fails if two checkpoints are retained;
       the measured value is ~1.0x.
    """
    run_dir = tmp_path / "lazy_run"
    steps = (0, 10, 20, 30, 40)
    model, _order = write_simple_bundle(
        run_dir, steps=steps, vocab_size=512, d_model=96, seq_len=8, batch_size=2
    )
    n_params = sum(p.numel() for p in model.parameters())
    # weights + exp_avg + exp_avg_sq, all fp32 in memory.
    per_checkpoint_bytes = 3 * 4 * n_params
    assert per_checkpoint_bytes > 1_000_000, "fixture too small for the allocation check"
    del model
    gc.collect()

    source = load_local_bundle(run_dir, tiny_lm_factory)
    assert not any(c.is_loaded for c in source.checkpoints)
    baseline = _live_tensor_bytes()

    peak = baseline
    previous: list[tuple[int, weakref.ReferenceType[Any], weakref.ReferenceType[Any]]] = []
    seen_steps: list[int] = []
    for checkpoint in source:
        seen_steps.append(checkpoint.step)
        bundle = checkpoint.model_bundle
        flat = checkpoint.optimizer_state.flat_state("exp_avg_sq")
        assert flat.numel() == source.layout.total_numel
        del flat

        # (1) exactly one checkpoint holds tensors.
        assert [c.step for c in source.checkpoints if c.is_loaded] == [checkpoint.step]
        # (2) everything yielded earlier is gone.
        for step, model_ref, param_ref in previous:
            assert model_ref() is None, f"model of step {step} is still alive"
            assert param_ref() is None, f"a parameter of step {step} is still alive"
        previous.append((checkpoint.step, weakref.ref(bundle.model), weakref.ref(bundle.params[0])))
        del bundle
        # (3) allocation never exceeds one checkpoint's worth.
        peak = max(peak, _live_tensor_bytes())

    assert seen_steps == list(steps)
    assert peak - baseline <= 2 * per_checkpoint_bytes, (
        f"live tensor bytes grew by {peak - baseline}, more than two checkpoints "
        f"({per_checkpoint_bytes} each)"
    )
    # Releasing the last checkpoint returns the process to the baseline.
    source.checkpoints[-1].release()
    assert not any(c.is_loaded for c in source.checkpoints)
    assert _live_tensor_bytes() - baseline <= per_checkpoint_bytes // 2


def test_release_and_reload_returns_equal_tensors(tmp_path: Path) -> None:
    """``release()`` is a cache drop, not a one-shot: re-access re-reads."""
    run_dir = tmp_path / "reload_run"
    write_simple_bundle(run_dir, steps=(0,), exp_avg_sq_constants=None)
    source = load_local_bundle(run_dir, tiny_lm_factory)
    checkpoint = source.checkpoints[0]

    first_bundle = checkpoint.model_bundle
    first_theta = torch.cat([p.detach().reshape(-1).clone() for p in first_bundle.params])
    first_state = checkpoint.optimizer_state.flat_state("exp_avg_sq").clone()
    assert checkpoint.model_bundle is first_bundle  # cached, not re-read

    checkpoint.release()
    assert not checkpoint.is_loaded
    second_bundle = checkpoint.model_bundle
    assert second_bundle is not first_bundle
    second_theta = torch.cat([p.detach().reshape(-1) for p in second_bundle.params])
    assert torch.equal(first_theta, second_theta)
    assert torch.equal(first_state, checkpoint.optimizer_state.flat_state("exp_avg_sq"))


# ------------------------------------------------------------------ T-factory


def test_import_string_factory_matches_callable_factory(tmp_path: Path) -> None:
    """T-factory: ``"module:function"`` resolves and loads identical weights."""
    run_dir = tmp_path / "factory_run"
    write_simple_bundle(run_dir, steps=(0,))

    via_callable = load_local_bundle(run_dir, tiny_lm_factory).checkpoints[0].model_bundle
    via_string = (
        load_local_bundle(run_dir, "tests.fixtures.synthetic_bundle:tiny_lm_factory")
        .checkpoints[0]
        .model_bundle
    )

    assert type(via_string.model) is type(via_callable.model)
    named_string = dict(via_string.model.named_parameters())
    for name, param in via_callable.model.named_parameters():
        assert torch.equal(named_string[name], param)
    assert [p.shape for p in via_string.params] == [p.shape for p in via_callable.params]


def test_factory_receives_bundle_metadata(tmp_path: Path) -> None:
    """The factory is handed the parsed metadata (dict access also works)."""
    run_dir = tmp_path / "factory_metadata_run"
    write_simple_bundle(run_dir, steps=(0,))
    seen: list[Any] = []

    def spy(metadata: Any) -> torch.nn.Module:
        seen.append(metadata)
        return tiny_lm_factory(metadata)

    _ = load_local_bundle(run_dir, spy).checkpoints[0].model_bundle
    assert len(seen) == 1
    metadata = seen[0]
    assert isinstance(metadata, BundleMetadata)
    assert metadata["model_config"]["d_model"] == DEFAULT_D_MODEL
    assert metadata.get("nope", "fallback") == "fallback"
    assert "optimizer" in metadata


@pytest.mark.parametrize(
    ("spec", "exc"),
    [
        ("tests.fixtures.synthetic_bundle", ValueError),
        ("no.such.module:factory", ImportError),
        ("tests.fixtures.synthetic_bundle:not_a_function", AttributeError),
        ("tests.fixtures.synthetic_bundle:DEFAULT_VOCAB_SIZE", TypeError),
        (42, TypeError),
    ],
)
def test_bad_factory_specs_raise(spec: Any, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        resolve_model_factory(spec)


def test_factory_returning_non_module_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_factory_run"
    write_simple_bundle(run_dir, steps=(0,))
    source = load_local_bundle(run_dir, lambda metadata: "not a module")  # type: ignore[arg-type,return-value]
    with pytest.raises(TypeError, match="expected an nn.Module"):
        _ = source.checkpoints[0].model_bundle


# -------------------------------------------------- source / analyzer plumbing


def test_step_selection_and_lookup(tmp_path: Path) -> None:
    run_dir = tmp_path / "select_run"
    write_simple_bundle(run_dir, steps=(0, 5, 10))

    subset = load_local_bundle(run_dir, tiny_lm_factory, steps=[10, 0])
    assert subset.steps == (0, 10)
    assert subset.checkpoint_for(10).step == 10
    assert subset.checkpoint_for("step0").step == 0
    assert subset.checkpoint_for("select_run@step10").step == 10
    assert subset.checkpoint_for("step_00000000").step == 0
    with pytest.raises(BundleSchemaError, match="no checkpoint"):
        subset.checkpoint_for(5)
    with pytest.raises(BundleSchemaError, match="not in the bundle"):
        load_local_bundle(run_dir, tiny_lm_factory, steps=[7])


def test_source_drives_the_analyzer_over_batch_a(tmp_path: Path) -> None:
    """The source plugs into the existing Analyzer loop, batch_a included."""
    run_dir = tmp_path / "analyzer_run"
    write_simple_bundle(run_dir, steps=(0, 10))
    source = load_local_bundle(run_dir, tiny_lm_factory)

    analyzer = Analyzer(
        eval_batches={"batch_a": source.batch_a_spec()},
        n_hutchinson=8,
        micro_batch_size=1,
        seed=0,
        sink=None,
    )
    results = analyzer.run(source.model_specs())

    assert [r.revision for r in results] == ["step0", "step10"]
    assert [r.checkpoint_id for r in results] == ["analyzer_run@step0", "analyzer_run@step10"]
    for result in results:
        observables = {row.observable: row.value for row in result.rows}
        assert observables.keys() >= {"chi_loss", "chi_net", "delta_loss", "chi_pos"}
        for name, value in observables.items():
            assert math.isfinite(value), f"{name} is not finite at {result.checkpoint_id}"
        assert observables["chi_loss"] > 0
        assert observables["chi_net"] > 0


# ------------------------------------------------------------ ParameterLayout


def test_layout_from_named_parameters_matches_writer_convention() -> None:
    model = BundleTinyLM()
    layout = ParameterLayout.from_named_parameters(model)
    assert layout.names == tuple(named_parameter_order(model))
    assert layout.total_numel == sum(p.numel() for p in model.parameters())
    assert layout.to_records() == [[n, list(p.shape)] for n, p in model.named_parameters()]
    offset = 0
    for name, shape in layout:
        assert layout.offset_of(name) == offset
        assert layout.slice_of(name) == slice(offset, offset + layout.numel_of(name))
        offset += math.prod(shape) if shape else 1
    assert offset == layout.total_numel


@pytest.mark.parametrize(
    "records",
    [
        {"embed.weight": [16, 6]},  # not a list
        [["embed.weight"]],  # not a pair
        [[1, [16, 6]]],  # non-string name
        [["embed.weight", "16x6"]],  # non-list shape
        [["embed.weight", [16, "6"]]],  # non-integer dimension
        [["a", [2]], ["a", [2]]],  # duplicate name
        [],  # empty
    ],
)
def test_layout_rejects_malformed_param_names(records: Any) -> None:
    with pytest.raises(LayoutMismatchError):
        ParameterLayout.from_records(records, source="param_names.json")


def test_layout_segment_rejects_foreign_vectors() -> None:
    layout = ParameterLayout.from_named_parameters(BundleTinyLM())
    with pytest.raises(LayoutMismatchError, match="expected a 1-D vector"):
        layout.segment(torch.zeros(layout.total_numel + 1), "fc1.bias")
    with pytest.raises(LayoutMismatchError, match="expected a 1-D vector"):
        layout.segment(torch.zeros(2, layout.total_numel // 2), "fc1.bias")
    with pytest.raises(LayoutMismatchError, match="unknown parameter"):
        layout.segment(torch.zeros(layout.total_numel), "nope")


def test_layout_flatten_named_validates_shapes() -> None:
    model = BundleTinyLM()
    layout = ParameterLayout.from_named_parameters(model)
    tensors = {name: p.detach() for name, p in model.named_parameters()}
    flat = layout.flatten_named(tensors, dtype=torch.float64)
    assert flat.dtype is torch.float64
    assert flat.numel() == layout.total_numel
    assert torch.equal(
        layout.segment(flat, "fc2.weight"),
        tensors["fc2.weight"].reshape(-1).to(torch.float64),
    )

    tensors["fc2.weight"] = torch.zeros(3, 3)
    with pytest.raises(LayoutMismatchError, match="canonical order says"):
        layout.flatten_named(tensors)

    del tensors["fc2.weight"]
    with pytest.raises(LayoutMismatchError, match="no tensor supplied"):
        layout.flatten_named(tensors)


def test_layout_validate_params_catches_count_and_shape() -> None:
    model = BundleTinyLM()
    layout = ParameterLayout.from_named_parameters(model)
    params = list(model.parameters())
    layout.validate_params(params)
    with pytest.raises(LayoutMismatchError, match="parameters but the canonical order"):
        layout.validate_params(params[:-1])
    swapped = [torch.nn.Parameter(torch.zeros(3, 3)), *params[1:]]
    with pytest.raises(LayoutMismatchError, match="canonical order says"):
        layout.validate_params(swapped)
