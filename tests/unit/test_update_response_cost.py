"""What the rows of the step that was taken cost, in countable terms.

Wall-clock and resident memory on a real model are measured by
``examples/benchmark/update_response/`` and depend on the machine. This file
pins the part that does not: how many more forward passes, backward passes,
parameter-sized buffers and models the feature needs, per checkpoint and per
evaluation batch, on a toy bundle. Nothing here is a timing.

Three configurations, as in the benchmark:

    baseline   the analysis without an update rule (the existing raw rows)
    formula    with the rule: the ``loss_change_first_order*`` rows + step rows
    realized   with the rule and the three realized rows

each for an SGD bundle (with and without momentum) and an AdamW bundle.

**How it is counted.** Forward passes: calls of the bundle's ``forward_fn``.
Backward passes: calls of ``torch.autograd.grad`` (the analyzer and the
``chi_net`` estimator use nothing else). Models: calls of the model factory, and
the models alive at a given moment. Memory: at fixed moments of the analysis the
garbage collector is scanned for live tensors, and their storages are summed
once each. The moments are

- *the start of the last evaluation batch's self pair* — the same point in all
  three configurations, with nothing transient alive, so the difference to the
  baseline is exactly what the feature keeps while later batches are processed;
- *the first loss evaluation at* ``theta_{t+1}`` (realized rows only).

The unit is ``P``: the bytes of one fp32 parameter-sized vector.

**What is pinned** (``E`` evaluation batches, ``M_k`` micro-batches of batch
``k``, ``S`` saved state vectors of the optimizer: 2 for AdamW, 1 for SGD with
momentum, 0 for SGD without):

    =========================================  ========  ==========  ================
    per checkpoint                             baseline  formula     realized
    =========================================  ========  ==========  ================
    backward passes                            n         n           n
    forward passes                             f         f           f + 2 * sum M_k
    models built                               1         1           2 [1]
    models alive at once                       1         1           2
    buffers kept while later batches run       0         E - 1 + S   E - 1 + S
    buffers kept during the realized pass      -         -           E + S, + 1 model
    =========================================  ========  ==========  ================

    [1] 1 when the successor is itself among the analyzed checkpoints: its model
        is then built once and reused.

So the feature adds no backward pass at all, no forward pass for the formula
rows, and **memory that grows by one parameter-sized vector per evaluation
batch**: the gradient of every evaluation batch is kept until the end of the
checkpoint, where the baseline keeps none (without cross pairs).
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from tests.fixtures.update_response_toys import held_out_batch, write_bundle
from tests.utils.bundle_fixture import build_default_model
from vatis.analyzer import (
    REALIZED_CHANGE_OBSERVABLES,
    UPDATE_RESPONSE_OBSERVABLES,
    Analyzer,
)
from vatis.core.update_response import AdamHyper, OptimizerUpdate, SGDHyper
from vatis.models.local_bundle import load_local_bundle, plain_lm_forward

N_HUTCHINSON = 2
MICRO_BATCH = 2
UPDATE_ROWS = 4  # rows of the update batch: 2 micro-batches
HELD_OUT_ROWS = 8  # rows of each held-out batch: 4 micro-batches
# Tolerance of the byte accounting, as a fraction of P. The differences are
# whole multiples of P up to a few small tensors that do not scale with the
# model (an Adam state's per-parameter step counters, the update batch's token
# ids fetched early): about 1 kB against P = 77 kB on this toy.
SMALL = 0.05


def _sgd(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01)


def _sgd_without_momentum(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=0.01)


def _adamw(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)


# name -> (optimizer factory, name recorded in metadata.json, state vectors S)
OPTIMIZERS: dict[str, tuple[Callable[[nn.Module], torch.optim.Optimizer], str, int]] = {
    "sgd": (_sgd, "sgd", 1),
    "sgd-without-momentum": (_sgd_without_momentum, "sgd", 0),
    "adamw": (_adamw, "adamw", 2),
}


# --------------------------------------------------------------- the counters

_TENSOR_TYPES = (torch.Tensor, nn.Parameter)
# One measurement per (bundle, configuration): several tests read the same one.
_MEASURED: dict[tuple[Any, ...], Any] = {}


def _live_tensor_bytes() -> tuple[int, int]:
    """``(bytes, n_flat)`` of the tensors alive right now.

    Every storage is counted once, however many tensors view it. ``n_flat`` is
    the number of distinct 1-D floating-point storages that are at least 1000
    elements long — on these toys, the flat parameter-space vectors (a gradient
    the analyzer keeps), since no parameter or batch tensor is 1-D and that long.
    """
    gc.collect()
    storages: dict[int, int] = {}
    flat: set[int] = set()
    for obj in gc.get_objects():
        # ``type(...) in``, not ``isinstance``: the latter touches lazy module
        # attributes of unrelated objects the collector tracks.
        if type(obj) not in _TENSOR_TYPES:
            continue
        storage = obj.untyped_storage()
        pointer = storage.data_ptr()
        if pointer == 0:
            continue
        storages[pointer] = storage.nbytes()
        if obj.ndim == 1 and obj.is_floating_point() and obj.numel() >= 1000:
            flat.add(pointer)
    return sum(storages.values()), len(flat)


@dataclass
class Snapshot:
    tensor_bytes: int
    flat_vectors: int
    models_alive: int


@dataclass
class Cost:
    """Everything one analysis of one checkpoint was seen to do."""

    forwards: int = 0
    backwards: int = 0
    models_built: int = 0
    max_models_alive: int = 0
    n_params: int = 0
    self_pair_start: dict[str, Snapshot] = field(default_factory=dict)
    realized_start: Snapshot | None = None
    observables: set[str] = field(default_factory=set)

    @property
    def vector_bytes(self) -> int:
        """``P``: one fp32 parameter-sized vector."""
        return 4 * self.n_params


def _measure(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    *,
    n_held_out: int = 1,
    steps: tuple[int, ...] = (1,),
) -> Cost:
    """Analyze ``steps`` of the bundle in one configuration and count."""
    key = (str(run_dir), mode, n_held_out, steps)
    if key in _MEASURED:
        cached: Cost = _MEASURED[key]
        return cached
    cost = Cost()
    alive: weakref.WeakSet[nn.Module] = weakref.WeakSet()

    def factory(metadata: Any) -> nn.Module:
        model = build_default_model(metadata)
        cost.models_built += 1
        alive.add(model)
        return model

    def forward(model: nn.Module, batch: Any) -> torch.Tensor:
        cost.forwards += 1
        cost.max_models_alive = max(cost.max_models_alive, len(alive))
        return plain_lm_forward(model, batch)

    real_grad = torch.autograd.grad

    def counting_grad(*args: Any, **kwargs: Any) -> Any:
        cost.backwards += 1
        return real_grad(*args, **kwargs)

    def snapshot() -> Snapshot:
        tensor_bytes, flat_vectors = _live_tensor_bytes()
        return Snapshot(tensor_bytes, flat_vectors, len(alive))

    real_self_pair = Analyzer._compute_self_pair
    real_mean_loss = Analyzer._mean_loss

    def self_pair(
        self: Analyzer, bundle: Any, ckpt_id: str, revision: str, name: str, *a: Any, **k: Any
    ) -> Any:
        cost.self_pair_start[name] = snapshot()
        return real_self_pair(self, bundle, ckpt_id, revision, name, *a, **k)

    def mean_loss(self: Analyzer, bundle: Any, raw: Any) -> float:
        if cost.realized_start is None:
            cost.realized_start = snapshot()
        return real_mean_loss(self, bundle, raw)

    source = load_local_bundle(run_dir, factory, steps=list(steps), forward_fn=forward)
    cost.n_params = source.layout.total_numel
    eval_batches: dict[str, Any] = {"train": source.batch_a_spec()}
    for index in range(n_held_out):
        eval_batches[f"val{index}"] = held_out_batch(seed=903 + index, n_rows=HELD_OUT_ROWS)
    update_response = (
        None if mode == "baseline" else source.update_response(realized=mode == "realized")
    )
    analyzer = Analyzer(
        eval_batches=eval_batches,
        update_response=update_response,
        chi_net_method="hutchinson",
        n_hutchinson=N_HUTCHINSON,
        micro_batch_size=MICRO_BATCH,
        dtype="fp32",
    )
    with monkeypatch.context() as patch:
        patch.setattr(torch.autograd, "grad", counting_grad)
        patch.setattr(Analyzer, "_compute_self_pair", self_pair)
        patch.setattr(Analyzer, "_mean_loss", mean_loss)
        results = analyzer.run(source.model_specs())
    cost.observables = {row.observable for result in results for row in result.rows}
    _MEASURED[key] = cost
    return cost


def _micro_batches(n_held_out: int) -> int:
    """``sum_k M_k``: micro-batches of all evaluation batches of one checkpoint."""
    return (UPDATE_ROWS + n_held_out * HELD_OUT_ROWS) // MICRO_BATCH


@pytest.fixture(scope="module", params=sorted(OPTIMIZERS))
def bundle(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, int]:
    make_optimizer, name, state_vectors = OPTIMIZERS[request.param]
    run_dir = tmp_path_factory.mktemp(f"cost_{request.param}") / "run"
    write_bundle(run_dir, make_optimizer, n_steps=3, bundle_steps=(1, 2, 3), optimizer_name=name)
    return run_dir, state_vectors


# ------------------------------------------------------------ passes and models


@pytest.mark.parametrize("n_held_out", [1, 2], ids=["2-eval-batches", "3-eval-batches"])
def test_the_formula_rows_cost_no_forward_and_no_backward_pass(
    bundle: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, n_held_out: int
) -> None:
    run_dir, _state_vectors = bundle
    baseline = _measure(run_dir, monkeypatch, "baseline", n_held_out=n_held_out)
    formula = _measure(run_dir, monkeypatch, "formula", n_held_out=n_held_out)

    assert not baseline.observables & set(UPDATE_RESPONSE_OBSERVABLES)
    assert set(UPDATE_RESPONSE_OBSERVABLES) <= formula.observables
    assert not formula.observables & set(REALIZED_CHANGE_OBSERVABLES)

    # The baseline itself: one backward per micro-batch for the gradient and one
    # per probe and micro-batch for chi_net.
    assert baseline.backwards == (1 + N_HUTCHINSON) * _micro_batches(n_held_out)
    assert formula.backwards == baseline.backwards
    assert formula.forwards == baseline.forwards
    assert (formula.models_built, formula.max_models_alive) == (1, 1)
    assert (baseline.models_built, baseline.max_models_alive) == (1, 1)


@pytest.mark.parametrize("n_held_out", [1, 2], ids=["2-eval-batches", "3-eval-batches"])
def test_the_realized_rows_cost_two_forward_passes_per_micro_batch_and_one_model(
    bundle: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, n_held_out: int
) -> None:
    run_dir, _state_vectors = bundle
    baseline = _measure(run_dir, monkeypatch, "baseline", n_held_out=n_held_out)
    realized = _measure(run_dir, monkeypatch, "realized", n_held_out=n_held_out)

    assert set(REALIZED_CHANGE_OBSERVABLES) <= realized.observables
    assert realized.backwards == baseline.backwards
    assert realized.forwards == baseline.forwards + 2 * _micro_batches(n_held_out)
    # theta_{t+1} is a second model: built for this checkpoint (step 2 is not
    # among the analyzed ones here) and alive next to the first.
    assert (realized.models_built, realized.max_models_alive) == (2, 2)


def test_a_successor_that_is_analyzed_too_is_built_once(
    bundle: tuple[Path, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consecutive checkpoints 1, 2: the model of step 2 serves as ``theta_{t+1}`` of
    step 1 and as ``theta_t`` of step 2. Step 3 is loaded for step 2's realized rows."""
    run_dir, _state_vectors = bundle
    baseline = _measure(run_dir, monkeypatch, "baseline", steps=(1, 2))
    realized = _measure(run_dir, monkeypatch, "realized", steps=(1, 2))
    assert baseline.models_built == 2
    assert realized.models_built == 3
    assert realized.max_models_alive == 2
    assert realized.forwards == baseline.forwards + 2 * 2 * _micro_batches(1)
    assert realized.backwards == baseline.backwards


# ----------------------------------------------------------------------- memory


@pytest.mark.parametrize("n_held_out", [1, 2, 3], ids=["E=2", "E=3", "E=4"])
def test_buffers_kept_while_later_batches_are_processed(
    bundle: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, n_held_out: int
) -> None:
    """``E - 1`` gradients and ``S`` optimizer-state vectors, each one ``P``.

    Measured where the last evaluation batch's self pair starts, in the baseline
    and with the rule: the same moment, nothing transient alive, so the
    difference is what the feature keeps — and it grows with ``E``.
    """
    run_dir, state_vectors = bundle
    n_eval = 1 + n_held_out
    last = f"val{n_held_out - 1}"
    baseline = _measure(run_dir, monkeypatch, "baseline", n_held_out=n_held_out)
    formula = _measure(run_dir, monkeypatch, "formula", n_held_out=n_held_out)
    vector = baseline.vector_bytes

    # The baseline keeps no gradient across batches (no cross pairs here) ...
    assert baseline.self_pair_start[last].flat_vectors == 0
    # ... the rule keeps the gradient of every batch already measured.
    assert formula.self_pair_start[last].flat_vectors == n_eval - 1

    extra = formula.self_pair_start[last].tensor_bytes - baseline.self_pair_start[last].tensor_bytes
    expected = (n_eval - 1 + state_vectors) * vector
    assert abs(extra - expected) < SMALL * vector, (
        f"{extra / vector:.3f} P kept, expected {n_eval - 1} gradients + {state_vectors} "
        "state vectors"
    )
    # At the first batch nothing is cached yet: only the state is extra.
    first = (
        formula.self_pair_start["train"].tensor_bytes
        - baseline.self_pair_start["train"].tensor_bytes
    )
    assert abs(first - state_vectors * vector) < SMALL * vector


@pytest.mark.parametrize("n_held_out", [1, 2], ids=["E=2", "E=3"])
def test_buffers_kept_during_the_realized_pass(
    bundle: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, n_held_out: int
) -> None:
    """All ``E`` gradients, the state, and a second model."""
    run_dir, state_vectors = bundle
    n_eval = 1 + n_held_out
    last = f"val{n_held_out - 1}"
    baseline = _measure(run_dir, monkeypatch, "baseline", n_held_out=n_held_out)
    realized = _measure(run_dir, monkeypatch, "realized", n_held_out=n_held_out)
    vector = baseline.vector_bytes
    assert realized.realized_start is not None and baseline.realized_start is None

    assert realized.realized_start.flat_vectors == n_eval
    assert realized.realized_start.models_alive == 2
    # Against the baseline's quiet moment: E gradients, S state vectors and the
    # weights of the second model (one more P).
    extra = realized.realized_start.tensor_bytes - baseline.self_pair_start[last].tensor_bytes
    expected = (n_eval + state_vectors + 1) * vector
    assert abs(extra - expected) < SMALL * vector, f"{extra / vector:.3f} P"


def test_the_reduction_never_holds_a_parameter_sized_fp64_vector() -> None:
    """The three parts are formed chunk by chunk: at most seven fp64 tensors of
    one chunk's size are alive at a time, whatever the size of the model."""
    torch.manual_seed(0)
    n_params, chunk = 5000, 64
    params = [torch.randn(3000), torch.randn(2000)]
    grad = torch.randn(n_params)
    rules = {
        "adamw": OptimizerUpdate(
            family="adam",
            hypers=[AdamHyper(lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.1)] * 2,
            step=3,
            first_moment=[torch.randn(3000), torch.randn(2000)],
            second_moment=[torch.rand(3000), torch.rand(2000)],
        ),
        "sgd": OptimizerUpdate(
            family="sgd",
            hypers=[SGDHyper(lr=1e-2, momentum=0.9, weight_decay=0.1)] * 2,
            step=0,
            first_moment=[torch.randn(3000), torch.randn(2000)],
            second_moment=None,
        ),
    }

    def live_fp64() -> list[int]:
        gc.collect()
        return [
            obj.numel()
            for obj in gc.get_objects()
            if type(obj) in _TENSOR_TYPES and obj.dtype == torch.float64
        ]

    assert live_fp64() == []  # every input above is fp32
    most: dict[str, int] = {}
    for name, rule in rules.items():
        parts: Iterator[Any] = rule.iter_parts(grad, params, chunk=chunk)
        most[name] = 0
        for _start, _stop, _d_grad, _d_hist, _d_wd in parts:
            sizes = live_fp64()
            assert max(sizes) <= chunk
            most[name] = max(most[name], len(sizes))
        del parts
    # g, theta, m, v and the three parts for AdamW; no v for SGD.
    assert most == {"adamw": 7, "sgd": 6}


# ------------------------------------- allocations of the calls the analyzer makes


def _allocations(call: Callable[[], Any]) -> tuple[int, int]:
    """What ``call`` allocates on the CPU, in bytes, as torch's profiler records it.

    Returns ``(largest, peak)``: the largest amount a single operator allocated,
    and the highest net amount allocated at any moment. An allocation is seen
    whether or not the tensor is kept, so a temporary that is dropped at once
    counts like one that is held.
    """
    with torch.autograd.profiler.profile(profile_memory=True) as prof:
        call()
    events = sorted(prof.function_events, key=lambda event: event.time_range.start)
    largest = net = peak = 0
    for event in events:
        largest = max(largest, event.self_cpu_memory_usage)
        net += event.self_cpu_memory_usage
        peak = max(peak, net)
    return largest, peak


@pytest.mark.parametrize("rule_name", ["adamw", "sgd"])
def test_the_reductions_the_analyzer_calls_allocate_per_chunk_not_per_model(
    monkeypatch: pytest.MonkeyPatch, rule_name: str
) -> None:
    """``response``, ``clip_coefficient`` and ``weight_delta_dot_fp64``, called as
    the analyzer calls them (without a ``chunk`` argument), on a model of 64
    chunks.

    The module's chunk size is lowered for the test, so that the calls
    themselves are watched and not a re-implementation of their loops. Pinned:

    - no operator allocates as much as two fp64 chunks (8 bytes per element;
      measured: one chunk, plus a scalar). A copy of the gradient in fp64, kept
      or not, would be 64 chunks; one in fp32, 32;
    - the net amount allocated never exceeds 20 fp64 chunks (measured: 14 in
      ``response`` for AdamW and 10 for SGD, 2 for the clip factor, 4 for the
      weight difference), under a third of one parameter-sized fp64 vector.
    """
    import vatis.core.update_response as update_response_module
    from vatis.core.update_response import weight_delta_dot_fp64

    chunk, sizes = 1000, (40_000, 24_000)
    n_params = sum(sizes)
    fp64_chunk = 8 * chunk
    monkeypatch.setattr(update_response_module, "_CHUNK", chunk)
    torch.manual_seed(0)
    params = [torch.randn(n) for n in sizes]
    params_after = [p + 1e-3 * torch.randn_like(p) for p in params]
    grad = torch.randn(n_params)
    grads_eval = {"update": grad, "held_out": torch.randn(n_params)}
    if rule_name == "adamw":
        rule = OptimizerUpdate(
            family="adam",
            hypers=[AdamHyper(lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.1)] * 2,
            step=3,
            first_moment=[torch.randn(n) for n in sizes],
            second_moment=[torch.rand(n) for n in sizes],
            clip_grad_norm=1.0,
        )
    else:
        rule = OptimizerUpdate(
            family="sgd",
            hypers=[SGDHyper(lr=1e-2, momentum=0.9, weight_decay=0.1)] * 2,
            step=3,
            first_moment=[torch.randn(n) for n in sizes],
            second_moment=None,
            clip_grad_norm=1.0,
        )
    assert rule.clip_coefficient(grad) < 1.0  # the clip engages: its branch runs too

    # The instrument sees a temporary that is not kept.
    largest, _ = _allocations(lambda: grad.to(torch.float64))
    assert largest == 8 * n_params

    calls: dict[str, Callable[[], Any]] = {
        "response": lambda: rule.response(grad, grads_eval, params),
        "clip_coefficient": lambda: rule.clip_coefficient(grad),
        "weight_delta_dot_fp64": lambda: weight_delta_dot_fp64(grad, params_after, params),
    }
    measured = {name: _allocations(call) for name, call in calls.items()}
    for name, (largest, peak) in measured.items():
        assert largest < 2 * fp64_chunk, (name, largest / fp64_chunk)
        assert peak <= 20 * fp64_chunk, (name, peak / fp64_chunk)
        # ... and the chunks were really cut: something of a chunk's size was allocated.
        assert largest >= fp64_chunk, (name, largest / fp64_chunk)
