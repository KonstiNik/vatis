"""Analyzer — the orchestrator that loops over (checkpoint, eval-batch) pairs.

The analyzer is intentionally a single thin class. It does NOT know about
HF (it consumes a :class:`vatis.models.hf.ModelBundle` opaque to its source)
and does NOT know about parallelism beyond the standard ``vatis.distributed``
helpers (sharding, all-reduce, rank-aware sink writes).

The flow per ``(checkpoint, eval_batch)`` is:

    1. Move the (sub-shard of the) batch to the model device.
    2. Compute ``chi_loss`` (closed form for CE; non-CE losses are deferred
       to v1.2 — see SESSION_SUMMARY.md).
    3. Compute ``delta_loss`` self via one backward of the total loss.
    4. Compute ``chi_net`` via the chosen estimator.
    5. (For each cross pair) compute ``delta_loss`` cross via two backwards.
    6. All-reduce the accumulators across ranks. (No-op single-GPU.)
    7. Compute ``chi_pos`` from the all-reduced totals.
    8. (Rank 0) write rows to the sink.

Both unnormalized and normalized observables are emitted.

Optimizer-aware measurement
---------------------------

When a ``preconditioner`` is configured the analyzer also measures the
preconditioned observables in the same pass: ``delta_loss_P``, ``chi_net_P`` and
``chi_pos_P``, the LNP triple of the effective kernel ``Theta_P = J P J^T`` of a preconditioned
update ``-eta P grad L`` (``docs/SPEC.md`` §"Optimizer-aware measurement"; for AdamW
this is the geometry factor only, not the step that was taken — see §"What
``delta_loss_P`` is not" there). It costs no extra
backward — ``delta_loss_P`` re-reduces the gradient vector step 3 already built
with the elementwise ``P`` weight, and ``chi_net_P`` comes out of the same
estimator call and the same probes as ``chi_net``. ``chi_loss`` is
kernel-independent and is not recomputed. Raw numerics are unchanged, bit for
bit: the P accumulators are additive, never in the raw path.

The rows land in the existing long format as new ``observable`` values, plus one
``preconditioner`` descriptor column that carries ``"none"`` on every raw row.

The update that was taken
-------------------------

When an ``update_response`` is configured the analyzer additionally measures,
for every eval batch ``B``, the first-order loss change under the update the
optimizer applied at that checkpoint, ``grad L^B . (theta_{t+1} - theta_t)``,
split into its gradient, history and weight-decay parts
(:mod:`vatis.core.update_response`) — and, where ``theta_{t+1}`` is available,
the realized loss change and what the first-order term leaves over. These need
no extra backward either (two extra forwards per batch for the realized change):
they are dot products of the gradients the self pairs already built.

Provenance and sampling error
-----------------------------

Every row also says how it was obtained: the ``chi_net`` estimator that ran
(``chi_net_method``), the number of ranks (``n_ranks``), the micro-batch size
and the probe distribution (``micro_batch_size``, ``hutchinson_distribution``),
and on the rows of an AdamW ``P`` its ``eps`` / ``beta2`` and the fraction of
its coordinates at the ``1/eps`` ceiling. The rows whose value depends on the Hutchinson probes carry
``value_probe_stderr``, the standard error of the value under the probe draw: the
estimators report the variance of their own estimate
(``ChiNetResult.chi_net_var``, a sum over micro-batches), the ranks' variances
are all-reduce-summed like the estimates themselves (the ranks draw independent
streams), and ``chi_pos`` and the cross pair inherit it to first order
(``_emit_rows``, ``_geometric_mean_stderr``). ``docs/SPEC.md`` -> "Standard
error of the sampled values" says what the number is and is not.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from vatis.core.chi_net import (
    build_estimator,
    select_chi_net_method,
)
from vatis.core.chi_net.base import ChiNetResult
from vatis.core.normalization import (
    DEFAULT_IGNORE_INDEX,
    normalization_factor,
    valid_token_mask,
)
from vatis.core.observables import (
    chi_loss_cross_entropy_unnormalized,
    chi_pos,
    chi_pos_exceeds_cauchy_schwarz,
)
from vatis.core.preconditioner import (
    RAW_DESCRIPTOR,
    Preconditioner,
    weighted_dot_fp64,
    weighted_sum_sq_fp64,
)
from vatis.core.update_response import OptimizerUpdate, weight_delta_dot_fp64
from vatis.data.batches import EvalBatchSpec, normalize_eval_batches
from vatis.data.collate import iter_micro_batches, move_batch
from vatis.distributed.ddp import (
    all_reduce_sum_scalar,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_rank,
)
from vatis.distributed.sharding import shard_batch
from vatis.models.hf import LossFn, ModelBundle, ValidMaskFn, load_hf_model
from vatis.sinks.base import ResultRow, ResultSink
from vatis.sinks.parquet import ParquetSink

# A "model spec" the analyzer accepts: either an HF (name, revision) string,
# or a fully-built ModelBundle for the user-driven path.
ModelSpec = str | ModelBundle

# All known observable names. Order matters only for the default sink rows.
ALL_OBSERVABLES: tuple[str, ...] = (
    "chi_loss",
    "chi_loss_normalized",
    "chi_net",
    "chi_net_normalized",
    "delta_loss",
    "chi_pos",
)

# The preconditioned observables. Emitted only when a ``preconditioner``
# is configured, and always tagged with that preconditioner's descriptor in the
# ``preconditioner`` column.
#
# There is deliberately no ``chi_net_P_normalized``: the result format names three
# P observables and extends the schema *minimally* (``docs/SPEC.md``). The sigma-normalized
# companion is a pure function of what is already written —
# ``chi_net_P_normalized = chi_net_P / sqrt(n_valid_a * n_valid_b)`` — using the
# ``n_valid_a`` / ``n_valid_b`` columns of the same row, so nothing is lost.
# ``chi_loss`` has no P variant at all (it is kernel-independent), and
# ``chi_pos_P`` is normalization-invariant just like ``chi_pos``.
P_OBSERVABLES: tuple[str, ...] = (
    "delta_loss_P",
    "chi_net_P",
    "chi_pos_P",
)

# Diagnostic companions of the P observables, emitted only for a preconditioner
# that has a ceiling (``Preconditioner.ceiling`` — ``1/eps`` for AdamW, reached
# where the second moment is zero): the same quantity computed with ``P`` zeroed
# everywhere except at the entries sitting at that ceiling. For
# ``delta_loss_P`` and for a self-pair ``chi_net_P`` (both linear in ``P``) this
# is the *part* of the value those entries contribute, so
# ``delta_loss_P_capped / delta_loss_P`` is their share. They exist because a
# handful of such entries can carry almost all of a P observable — see
# ``docs/SPEC.md`` §"Coordinates at the ceiling".
P_CAPPED_OBSERVABLES: tuple[str, ...] = (
    "delta_loss_P_capped",
    "chi_net_P_capped",
)

# The first-order loss change under the update the optimizer applied at the
# checkpoint, ``grad L^B . dtheta`` with ``dtheta`` rebuilt from the optimizer
# state saved before the step: its three parts and their sum
# (``docs/SPEC.md`` §"The response to the update that was taken"). Emitted only
# when an ``update_response`` is configured, on the pair (update batch, B) for
# every eval batch B, tagged with the update rule's descriptor
# (``"adamw_update"``, ...). Unlike ``delta_loss`` these are loss changes: they
# carry the learning rate and the sign (negative = the loss on B goes down).
UPDATE_RESPONSE_OBSERVABLES: tuple[str, ...] = (
    "loss_change_first_order_grad",
    "loss_change_first_order_hist",
    "loss_change_first_order_wd",
    "loss_change_first_order",
)

# Two facts about the step itself, recorded next to the rows they scale so that
# a result file can be read without the bundle: the learning rate the step
# applied and the factor its gradient was clipped by. Emitted once per
# checkpoint, on the pair (update batch, update batch), tagged with the update
# rule's descriptor like the formula rows.
#
# - ``update_lr`` exists only when every parameter was updated with the same
#   rate. When parameter groups have different rates there is no single number,
#   and ``update_lr_min`` / ``update_lr_max`` are written instead (the smallest
#   and the largest rate of any group) — never an average.
# - ``update_clip_factor`` is ``min(1, c / (||g|| + 1e-6))`` for a run that
#   clipped the global gradient norm at ``c``, and exactly 1 for a run that did
#   not clip: either way the factor the gradient reached the optimizer with.
UPDATE_STEP_OBSERVABLES: tuple[str, ...] = (
    "update_lr",
    "update_lr_min",
    "update_lr_max",
    "update_clip_factor",
)

# Measured only where the source can also supply ``theta_{t+1}``: the same
# first-order change from the two stored weight vectors, the loss change the
# step realized on B, and their difference (what first order leaves over).
# No optimizer formula enters them, so they are tagged ``"none"``.
REALIZED_CHANGE_OBSERVABLES: tuple[str, ...] = (
    "loss_change_first_order_weights",
    "loss_change",
    "loss_change_residual",
)

# What an ``update_response`` argument may be: a per-checkpoint factory
# ``fn(checkpoint_id) -> OptimizerUpdate | None`` (``None`` skips the rows for
# that checkpoint). For a checkpoint bundle: ``LocalBundleSource.update_response()``.
UpdateResponseFactory = Callable[[str], OptimizerUpdate | None]

# What a ``preconditioner`` argument may be:
#   - None / "none"                  -> raw observables only (the default, and the
#                                       behaviour before this argument existed)
#   - "identity"                     -> P = I, SGD's geometry measured through the
#                                       preconditioned code path (T-identity)
#   - a Preconditioner               -> the same fixed P at every checkpoint
#   - fn(checkpoint_id) -> P | None  -> per-checkpoint P; returning None skips
#                                       the P rows for that checkpoint (this is
#                                       how the adamw source handles a step-0
#                                       checkpoint, whose optimizer state is
#                                       legitimately empty)
PreconditionerFactory = Callable[[str], Preconditioner | None]
PreconditionerLike = str | Preconditioner | PreconditionerFactory

# The mask closure every accumulator in ``_compute_self_pair`` shares: unlike a
# bundle's ``ValidMaskFn`` it never returns ``None`` (see
# ``_intersected_valid_mask_fn``). Bound here as a module-level alias because
# ``Callable`` itself is deleted at the bottom of this module.
EffectiveMaskFn = Callable[[Any, torch.Tensor], torch.Tensor]


@dataclass
class CheckpointResult:
    """In-memory record of all observables computed at one checkpoint."""

    checkpoint_id: str
    revision: str
    rows: list[ResultRow] = field(default_factory=list)


def _resolve_model(
    model: ModelSpec,
    *,
    revision: str | None,
    dtype: str,
    device: torch.device | str,
) -> ModelBundle:
    if isinstance(model, ModelBundle):
        return model
    if isinstance(model, str):
        return load_hf_model(model, revision=revision, dtype=dtype, device=device)
    raise TypeError(
        f"unsupported model spec type: {type(model).__name__}; expected str or ModelBundle"
    )


class Analyzer:
    """Orchestrate vatis observables across checkpoints and eval batches.

    The class can be used standalone — instantiate, then call :meth:`run` —
    or as part of the one-shot :func:`analyze` convenience function below.

    Args:
        eval_batches: dict mapping eval-batch name to a value that is one of:
            - already a :class:`EvalBatchSpec`,
            - a tensor / dict / tuple (auto-wrapped as ``fixed``),
            - a DataLoader (auto-wrapped as ``resample``),
            - a callable ``fn(checkpoint_id) -> batch`` (auto-wrapped).
        observables: which observables to compute and emit. ``None`` (default)
            means all of ``ALL_OBSERVABLES``, plus all of ``P_OBSERVABLES`` when
            a ``preconditioner`` is configured. Pass an explicit list to select;
            naming a P observable without a ``preconditioner`` is an error.
        preconditioner: the diagonal ``P`` for the preconditioned observables, or
            ``None`` (default) for the raw observables only. See
            :data:`PreconditionerLike`. For a checkpoint bundle use
            ``LocalBundleSource.adamw_preconditioner()``, which reads each
            checkpoint's saved (pre-step) optimizer state.
        update_response: a per-checkpoint factory of the update the optimizer
            applied (:data:`UpdateResponseFactory`), or ``None`` (default). With
            it the analyzer emits the :data:`UPDATE_RESPONSE_OBSERVABLES` rows
            — and the :data:`REALIZED_CHANGE_OBSERVABLES` rows where the factory
            supplies ``theta_{t+1}`` — on the pair (update batch, ``B``) for
            every eval batch ``B``, plus the step's learning rate and clip
            factor (:data:`UPDATE_STEP_OBSERVABLES`) on the update batch's own
            pair. The update batch must be one of the eval
            batches; the analyzer finds it by comparing them with the batch the
            factory reports and refuses when none matches. For a checkpoint
            bundle: ``update_response=source.update_response()`` with
            ``source.batch_a_spec()`` among the eval batches. Independent of
            ``preconditioner``.
        chi_net_method: ``"hutchinson"``, ``"per_sequence_cv"``, ``"opacus"``,
            or ``None`` for the default, ``"hutchinson"``.
        n_hutchinson: number of probe vectors per micro-batch. With at least
            two, the rows whose value depends on the probes carry its standard
            error in ``value_probe_stderr``; with one there is no spread to estimate
            it from and the column is null.
        hutchinson_distribution: ``"rademacher"`` (default) or ``"gaussian"``.
        micro_batch_size: number of sequences per backward. Default 1.
        cross_pairs: optional list of ``(name_a, name_b)`` tuples to compute
            ``delta_loss(A, B)`` (and the corresponding ``chi_pos``) for.
            Self pairs ``(name, name)`` are always computed.
        cross_grad_storage: where a batch's cached full gradient lives while the
            cross-pair dot product is computed — ``"gpu"``, ``"cpu"``, or
            ``"auto"`` (default). The cross dot needs both batches' P-sized
            gradients; for large models keeping them on the GPU alongside a
            later batch's backward can OOM. ``"cpu"`` offloads them to host RAM
            (the dot then runs on the CPU); ``"gpu"`` keeps them on-device
            (fast, no transfer); ``"auto"`` offloads only when an on-GPU cache
            would not leave room for a self-pair backward. Only matters when
            ``cross_pairs`` is non-empty.
        sink: a :class:`ResultSink`, a path str (auto-wraps to
            :class:`ParquetSink`), or a list of sinks.
        seed: base seed used to derive Hutchinson probes; the analyzer mixes
            this with the checkpoint id and batch name to get a per-call seed,
            recorded as ``hutchinson_seed``. A single process draws its probes
            from that seed. Under DDP rank 0 does too, and every other rank
            draws from its own stream, seeded from that seed and its rank — so
            ``seed``, the checkpoint id, the batch name and the world size
            determine a run's probes, and a multi-rank ``chi_net`` depends on
            the number of ranks.
        device: model device for single-GPU mode. In DDP mode this is
            overridden by ``cuda:LOCAL_RANK``.
        dtype: forward dtype for HF models loaded by name. Ignored when the
            user passes a fully built ModelBundle.
        ignore_index: the label value that marks an ignored position. It **must
            be the value** ``loss_fn`` **ignores**: the analyzer uses it to decide
            which positions count (also in the ``chi_loss`` closed form), and
            writes it into the labels ``loss_fn`` sees where a ``valid_mask_fn``
            rejects a position. Defaults to HF's -100.
    """

    def __init__(
        self,
        *,
        eval_batches: dict[str, Any],
        observables: Sequence[str] | None = None,
        preconditioner: PreconditionerLike | None = None,
        update_response: UpdateResponseFactory | None = None,
        chi_net_method: str | None = None,
        n_hutchinson: int = 32,
        hutchinson_distribution: str = "rademacher",
        micro_batch_size: int = 1,
        cross_pairs: list[tuple[str, str]] | None = None,
        cross_grad_storage: str = "auto",
        sink: ResultSink | str | list[ResultSink] | None = None,
        seed: int = 0,
        device: torch.device | str = "cpu",
        dtype: str = "bf16",
        ignore_index: int = DEFAULT_IGNORE_INDEX,
    ) -> None:
        self.eval_batches: dict[str, EvalBatchSpec] = normalize_eval_batches(eval_batches)
        if not self.eval_batches:
            raise ValueError("at least one eval batch is required")

        self.preconditioner = _normalize_preconditioner(preconditioner)
        if update_response is not None and not callable(update_response):
            raise TypeError(
                f"unsupported update_response type: {type(update_response).__name__}; "
                "expected a callable(checkpoint_id) -> OptimizerUpdate | None, e.g. "
                "LocalBundleSource.update_response()"
            )
        self.update_response = update_response
        response_observables = (
            UPDATE_RESPONSE_OBSERVABLES + UPDATE_STEP_OBSERVABLES + REALIZED_CHANGE_OBSERVABLES
        )
        valid_observables = (
            ALL_OBSERVABLES + P_OBSERVABLES + P_CAPPED_OBSERVABLES + response_observables
        )
        if observables is None:
            observables = (
                ALL_OBSERVABLES
                + (P_OBSERVABLES + P_CAPPED_OBSERVABLES if self.preconditioner is not None else ())
                + (response_observables if self.update_response is not None else ())
            )
        unknown = set(observables) - set(valid_observables)
        if unknown:
            raise ValueError(f"unknown observables {sorted(unknown)}; valid: {valid_observables}")
        requested_response = set(observables) & set(response_observables)
        if requested_response and self.update_response is None:
            raise ValueError(
                f"observables {sorted(requested_response)} describe the update the optimizer "
                "applied but no update_response was given; pass update_response=... (for a "
                "checkpoint bundle: LocalBundleSource.update_response())"
            )
        requested_p = set(observables) & set(P_OBSERVABLES + P_CAPPED_OBSERVABLES)
        if requested_p and self.preconditioner is None:
            raise ValueError(
                f"observables {sorted(requested_p)} are preconditioned observables "
                "but no preconditioner was given; pass preconditioner=... (e.g. "
                "'identity', or LocalBundleSource.adamw_preconditioner())"
            )
        self.observables = tuple(observables)
        self.chi_net_method_request = chi_net_method
        self.n_hutchinson = n_hutchinson
        self.hutchinson_distribution = hutchinson_distribution
        self.micro_batch_size = micro_batch_size
        self.cross_pairs = list(cross_pairs or [])
        if cross_grad_storage not in ("auto", "gpu", "cpu"):
            raise ValueError(
                f"cross_grad_storage must be 'auto', 'gpu', or 'cpu'; got {cross_grad_storage!r}"
            )
        self.cross_grad_storage = cross_grad_storage
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self.ignore_index = ignore_index

        # Sink wiring
        self.sinks: list[ResultSink] = self._normalize_sinks(sink)

        # Per-batch self-pair scratch caches. ``_compute_cross_pair`` reads
        # ``chi_loss`` / ``chi_net`` for each named batch from these dicts;
        # they are populated by ``_emit_rows`` during the self loop. Initialize
        # them eagerly so the cross-pair precondition check doesn't need to
        # special-case "cache not yet created".
        self._chi_loss_cache: dict[str, float] = {}
        self._chi_net_cache: dict[str, float] = {}
        # Same, for the preconditioned chi_net^P (the cross pair takes
        # the geometric mean of the two batches' values, exactly as for raw).
        self._chi_net_p_cache: dict[str, float] = {}
        self._chi_net_p_capped_cache: dict[str, float] = {}
        # What the cross pair needs to say how its rows were obtained: the
        # estimator each batch was measured with, and the standard error of each
        # batch's chi_net / chi_net^P / capped part (``None`` where there is none).
        self._chi_net_method_cache: dict[str, str] = {}
        self._hutchinson_distribution_cache: dict[str, str | None] = {}
        self._chi_net_stderr_cache: dict[str, float | None] = {}
        self._chi_net_p_stderr_cache: dict[str, float | None] = {}
        self._chi_net_p_capped_stderr_cache: dict[str, float | None] = {}

    @staticmethod
    def _normalize_sinks(
        sink: ResultSink | str | list[ResultSink] | None,
    ) -> list[ResultSink]:
        if sink is None:
            return []
        if isinstance(sink, ResultSink):
            return [sink]
        if isinstance(sink, str):
            return [ParquetSink(sink)]
        if isinstance(sink, list):
            out: list[ResultSink] = []
            for s in sink:
                if isinstance(s, str):
                    out.append(ParquetSink(s))
                elif isinstance(s, ResultSink):
                    out.append(s)
                else:
                    raise TypeError(f"unsupported sink entry type: {type(s).__name__}")
            return out
        raise TypeError(f"unsupported sink type: {type(sink).__name__}")

    def _resolve_cross_grad_storage(self, bundle: ModelBundle, n_cached: int | None = None) -> str:
        """Decide where cached cross-pair gradients live: ``"gpu"`` or ``"cpu"``.

        ``n_cached`` is the number of batches whose gradient is kept; by default
        the batches named in ``cross_pairs`` (an update response keeps all).

        Explicit ``"gpu"``/``"cpu"`` pass through. ``"auto"`` offloads to host
        RAM only when keeping the cached fp32 gradient(s) on the GPU would not
        leave room for a self-pair's backward. For a 7B model the cached
        gradient (~29 GB) plus one self-pair's footprint (~weights + 8·P)
        overflows an 80–90 GB card, so ``"auto"`` offloads; small models stay
        GPU-resident (no transfer, no numeric change — important for the
        perspic cross-validation tests).

        The estimate is intentionally conservative (a 0.85 fraction over an
        optimistic ``8·P`` self-pair floor); if ``"gpu"`` still OOMs on a
        borderline model, pass ``cross_grad_storage="cpu"`` explicitly.
        """
        if self.cross_grad_storage in ("gpu", "cpu"):
            return self.cross_grad_storage
        # The device that matters is the one the *model* (and hence the cached
        # gradients) lives on. For the HF path that is ``self.device``; a bundle
        # source places its own model (``load_local_bundle(device=...)``) and
        # ``Analyzer.device`` may still be the "cpu" default, which used to
        # switch the heuristic off for exactly the large-model case it exists for.
        device = next((p.device for p in bundle.params), self.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return "gpu"  # no host-offload target that would help
        if n_cached is None:
            n_cached = len({n for pair in self.cross_pairs for n in pair if pair[0] != pair[1]})
        n_params = sum(p.numel() for p in bundle.params)
        weights_bytes = sum(p.numel() * p.element_size() for p in bundle.params)
        total = torch.cuda.get_device_properties(device).total_memory
        offload = _should_offload_cross_grads(
            n_params=n_params,
            weights_bytes=weights_bytes,
            n_cross_batches=n_cached,
            total_device_bytes=total,
        )
        return "cpu" if offload else "gpu"

    # ------------------------------------------------------------------ runs

    def run(
        self,
        model_specs: Iterable[tuple[ModelSpec, str | None]],
    ) -> list[CheckpointResult]:
        """Run analysis over a sequence of ``(model_spec, revision)`` pairs.

        For HF model loading, ``model_spec`` is the hub name and ``revision``
        is the checkpoint tag. For pre-built ModelBundles, ``revision`` is
        used only as the row label.

        Returns the per-checkpoint results in input order. Sink writes happen
        on rank 0; the in-memory result objects are populated on every rank
        but only meaningful on rank 0.
        """
        results: list[CheckpointResult] = []
        try:
            for spec, revision in model_specs:
                bundle = _resolve_model(
                    spec,
                    revision=revision,
                    dtype=self.dtype,
                    device=self.device,
                )
                ckpt_id = bundle.identifier or _label(spec, revision)
                rev_label = revision or ""
                results.append(self._run_one_checkpoint(bundle, ckpt_id, rev_label))
                # If the user gave us a string spec we loaded the model
                # ourselves and should free it.
                if isinstance(spec, str):
                    del bundle
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            for s in self.sinks:
                s.close()
        return results

    def _run_one_checkpoint(
        self,
        bundle: ModelBundle,
        ckpt_id: str,
        revision: str,
    ) -> CheckpointResult:
        bundle.model.eval()
        # The analyzer needs requires_grad for its backwards, but the bundle may
        # be the *caller's* model: a user-built ModelBundle whose params include a
        # deliberately frozen tensor used to come back unfrozen. Save the flags and
        # restore them in the finally below, so measuring a model does not mutate
        # it. (The bundle loaders set requires_grad themselves on models they own;
        # that is their business, and restoring is a no-op there.)
        saved_requires_grad = [p.requires_grad for p in bundle.params]
        for p in bundle.params:
            p.requires_grad_(True)
        try:
            return self._run_one_checkpoint_inner(bundle, ckpt_id, revision)
        finally:
            for p, flag in zip(bundle.params, saved_requires_grad, strict=True):
                p.requires_grad_(flag)

    def _run_one_checkpoint_inner(
        self,
        bundle: ModelBundle,
        ckpt_id: str,
        revision: str,
    ) -> CheckpointResult:
        result = CheckpointResult(checkpoint_id=ckpt_id, revision=revision)

        # The preconditioner is per checkpoint (an Adam P is built from *this*
        # checkpoint's saved optimizer state), resolved once and reused for every
        # eval batch and cross pair below. ``None`` here means "no P rows for
        # this checkpoint" — either no preconditioner was configured at all, or
        # the factory declined (see the step-0 policy in
        # ``LocalBundleSource.adamw_preconditioner``).
        # The update rule is resolved first: it keeps the saved optimizer state,
        # which the P source would otherwise load, release and have reloaded.
        update = self._resolve_update_response(ckpt_id)
        # With an update rule the eval batches are fetched once, here, and the
        # update batch is located among them before anything expensive runs: a
        # rule that cannot be placed fails now, not after every chi_net
        # estimate of the checkpoint. (A resampling spec hands out another batch
        # each time it is asked, so the self pairs below get these very ones.)
        raw_batches: dict[str, Any] | None = None
        update_name: str | None = None
        if update is not None:
            raw_batches = {
                name: spec.get_batch(ckpt_id) for name, spec in self.eval_batches.items()
            }
            update_name = _locate_update_batch(update, raw_batches, ckpt_id)
        precond = self._resolve_preconditioner(bundle, ckpt_id)

        # Self pairs first. We cache a batch's full parameter-space gradient
        # ONLY if some cross pair will consume it via the dot product. Caching it
        # unconditionally keeps a P-sized vector resident (e.g. ~29 GB fp32 for a
        # 7B model) while subsequent batches run their own backward — for large
        # models that retained gradient is enough to OOM even when no cross pair
        # was requested. So we cache selectively and drop the local reference
        # otherwise, freeing it before the next batch's backward.
        cross_names = {n for pair in self.cross_pairs for n in pair if pair[0] != pair[1]}
        # The update response dots EVERY batch's gradient with the update, so
        # with it all of them are kept (one P-sized vector per eval batch).
        cached_names = set(self.eval_batches) if update is not None else cross_names
        grad_storage = (
            self._resolve_cross_grad_storage(bundle, n_cached=len(cached_names))
            if cached_names
            else "gpu"
        )
        per_batch_grad_cache: dict[str, torch.Tensor] = {}
        per_batch_n_valid: dict[str, int] = {}
        no_gradient: tuple[int, ...] = ()

        for name, spec in self.eval_batches.items():
            self_result, g_full = self._compute_self_pair(
                bundle,
                ckpt_id,
                revision,
                name,
                spec,
                precond,
                raw_batch=None if raw_batches is None else raw_batches[name],
                track_no_gradient=name == update_name,
            )
            result.rows.extend(self_result)
            per_batch_n_valid[name] = self._last_n_valid
            if name == update_name:
                no_gradient = self._last_no_gradient
            if g_full is not None and name in cached_names:
                # Cache for the cross-pair dot. Offload to host RAM when keeping
                # it on the GPU would crowd out a later batch's backward (large
                # models); otherwise keep it on-device. The cross dot
                # (``_dot_fp64``) runs on whichever device the gradients live on.
                per_batch_grad_cache[name] = g_full.to("cpu") if grad_storage == "cpu" else g_full
            # Drop the local (GPU) reference; if it wasn't cached on-device above
            # it is freed now, so gradients can't pile up across batches.
            g_full = None

        # Cross pairs (if any) — only delta_loss + chi_pos.
        for a, b in self.cross_pairs:
            if a == b:
                continue  # already handled in the self loop
            if a not in per_batch_grad_cache or b not in per_batch_grad_cache:
                # Need both grads. If either is missing (e.g. user disabled
                # delta_loss), skip with a warning.
                continue
            cross_rows = self._compute_cross_pair(
                ckpt_id=ckpt_id,
                revision=revision,
                name_a=a,
                name_b=b,
                g_a=per_batch_grad_cache[a],
                g_b=per_batch_grad_cache[b],
                n_valid_a=per_batch_n_valid[a],
                n_valid_b=per_batch_n_valid[b],
                bundle=bundle,
                precond=precond,
            )
            result.rows.extend(cross_rows)

        if update is not None and raw_batches is not None and update_name is not None:
            result.rows.extend(
                self._compute_update_response(
                    bundle=bundle,
                    ckpt_id=ckpt_id,
                    revision=revision,
                    update=update,
                    update_name=update_name,
                    no_gradient=no_gradient,
                    grads=per_batch_grad_cache,
                    n_valid=per_batch_n_valid,
                    raw_batches=raw_batches,
                )
            )

        if is_main_rank() and self.sinks:
            for s in self.sinks:
                s.write_rows(result.rows)

        return result

    # --------------------------------------------------------- preconditioner

    def _resolve_preconditioner(
        self,
        bundle: ModelBundle,
        ckpt_id: str,
    ) -> Preconditioner | None:
        """Resolve the configured preconditioner for one checkpoint.

        Returns ``None`` when only the raw observables should be measured. The
        resolved ``P`` is moved to the model's device and checked against
        ``bundle.params``: its length must equal the number of parameter
        elements the analyzer flattens, otherwise ``P`` and the gradient are not
        describing the same parameter space and every ``chi_pos^P`` would be a
        plausible-looking wrong number. (The *order* is guaranteed upstream: a
        bundle's ``P`` and ``bundle.params`` both come from the one
        :class:`~vatis.core.param_layout.ParameterLayout` of
        ``param_names.json``.)

        The ``P`` dtype is left alone — the weighted reductions promote to fp64
        regardless, so an fp64 ``P`` on an fp32 gradient loses nothing.
        """
        spec = self.preconditioner
        if spec is None:
            return None
        device = next(bundle.model.parameters()).device
        precond: Preconditioner | None
        if isinstance(spec, str):
            # _normalize_preconditioner already rejected anything else.
            precond = Preconditioner.identity(
                sum(int(p.numel()) for p in bundle.params), device=device
            )
        elif isinstance(spec, Preconditioner):
            precond = spec
        else:
            precond = spec(ckpt_id)
            if precond is None:
                return None
            if not isinstance(precond, Preconditioner):
                raise TypeError(
                    f"preconditioner factory returned {type(precond).__name__} for "
                    f"checkpoint {ckpt_id!r}; expected a Preconditioner or None"
                )
        precond = precond.to(device=device)
        precond.validate_for(bundle.params, context=f"checkpoint {ckpt_id!r} parameters")
        return precond

    # ------------------------------------------------------------- self pair

    def _compute_self_pair(
        self,
        bundle: ModelBundle,
        ckpt_id: str,
        revision: str,
        batch_name: str,
        spec: EvalBatchSpec,
        precond: Preconditioner | None = None,
        *,
        raw_batch: Any = None,
        track_no_gradient: bool = False,
    ) -> tuple[list[ResultRow], torch.Tensor | None]:
        """Compute chi_loss / delta_loss / chi_net / chi_pos for one batch.

        With ``precond`` given, also the preconditioned triple
        (``delta_loss_P``, ``chi_net_P``, ``chi_pos_P``) — no extra backward:
        ``delta_loss_P`` re-reduces the same gradient vector with the ``P``
        weight and ``chi_net_P`` comes from the same estimator call.

        ``raw_batch`` is the batch to measure when the caller has already
        fetched it from ``spec`` (the update-response path does, once, for every
        eval batch); otherwise it is fetched here. With ``track_no_gradient`` the
        indices of the parameters that received **no** gradient from any
        micro-batch on any rank — ``autograd`` returned ``None`` for them, they
        are not in the graph — are left in ``self._last_no_gradient``.

        Returns the rows AND the flat parameter-space gradient of the total
        loss (used as a cache for cross pairs).
        """
        t0 = time.perf_counter()
        device = next(bundle.model.parameters()).device

        raw = spec.get_batch(ckpt_id) if raw_batch is None else raw_batch
        # Shard across DDP ranks; if single-GPU shard_batch is a no-op slice.
        local_batch, local_b, _global_b = shard_batch(
            raw, rank=get_rank(), world_size=get_world_size()
        )
        local_batch = move_batch(local_batch, device)

        # THE mask, built once and used by every accumulator below: the bundle's
        # ``valid_mask_fn`` intersected with ``labels != ignore_index``. See
        # ``_intersected_valid_mask_fn`` for why all three paths need it.
        valid_mask_fn = _intersected_valid_mask_fn(bundle.valid_mask_fn, self.ignore_index)
        # THE loss, likewise: ``bundle.loss_fn`` evaluated on labels that carry
        # that mask, so the loss is restricted to the positions the mask accepts.
        # Used by the ``delta_loss`` backward below and handed to the estimator.
        loss_fn = _label_folded_loss_fn(bundle.loss_fn, valid_mask_fn, self.ignore_index)

        # ----- chi_loss (per-rank partial, raw sum) -----
        # We accumulate the UN-normalized squared sum
        # ``Sigma_valid (softmax - onehot)^2`` across micro-batches and ranks,
        # then divide by N_total^2 once at the end. This avoids the
        # micro-batch normalization mistake where each micro divides by its
        # own N instead of the global N.
        chi_loss_raw_local = torch.zeros((), dtype=torch.float64, device=device)
        n_valid_local = 0
        for _start, _stop, micro in iter_micro_batches(local_batch, self.micro_batch_size):
            with torch.no_grad():
                logits = bundle.forward_fn(bundle.model, micro)
            targets = _extract_targets(micro)
            # Numerator and denominator must agree on which positions count, and
            # so must chi_net and delta_loss below -- hence the one shared mask.
            effective_mask = valid_mask_fn(micro, logits)
            raw = chi_loss_cross_entropy_unnormalized(
                logits, targets, ignore_index=self.ignore_index, attention_mask=effective_mask
            )
            chi_loss_raw_local = chi_loss_raw_local + raw.to(dtype=torch.float64)
            n_valid_local += int(effective_mask.sum().item())
            del logits

        # ----- delta_loss self (per-rank partial gradient) -----
        # We accumulate the parameter-space gradient across micro-batches.
        # Within a single rank, we sum the gradients of each micro-batch's
        # contribution to L_total. Because L_total = (1/N_total) Σ ℓ and the
        # micro-batches partition the total, the gradient sums across micros.
        # We backprop micro-by-micro, weighted by (n_valid_micro / n_valid_local)
        # to assemble grad of (1/N_local) Σ ℓ_local — i.e. the rank's
        # *local* mean loss. The all-reduce step weights it by n_valid_local
        # to recover the *global* mean loss gradient.
        flat_grad_local = _zero_param_vector(bundle.params, device)
        # Slot 0 counts the backwards that ran, slot i + 1 the backwards that
        # handed parameter i a gradient. Kept (and all-reduced) only on request.
        received = (
            torch.zeros(len(bundle.params) + 1, dtype=torch.float32, device=device)
            if track_no_gradient
            else None
        )
        n_total_for_loss = n_valid_local
        if n_total_for_loss == 0:
            # No valid tokens on this rank. Still emit zeros and let the
            # all-reduce handle the global counts.
            delta_loss_local = torch.zeros((), dtype=torch.float64, device=device)
        else:
            for _start, _stop, micro in iter_micro_batches(local_batch, self.micro_batch_size):
                logits = bundle.forward_fn(bundle.model, micro)
                # Compute the micro-batch's contribution to L_local using the
                # SAME loss_fn the user provided, on labels that carry the mask
                # (``_label_folded_loss_fn``). We then re-scale by the ratio
                # of micro / total valid tokens (the loss_fn already divides by
                # the micro's n_valid, so we multiply back to get an unscaled
                # sum and divide by total).
                # Counted over the SAME set as n_total_for_loss (the intersected
                # mask). Using the un-intersected valid_mask_fn count here made
                # delta_loss depend on micro_batch_size whenever the two masks
                # disagreed -- see _intersected_valid_mask_fn.
                n_valid_micro = int(valid_mask_fn(micro, logits).sum().item())
                if n_valid_micro == 0:
                    del logits
                    continue
                loss_micro = loss_fn(logits, micro)
                weight = float(n_valid_micro) / float(n_total_for_loss)
                # gradient of (weight * loss_micro) w.r.t. theta
                grads = torch.autograd.grad(
                    weight * loss_micro,
                    bundle.params,
                    retain_graph=False,
                    allow_unused=True,
                )
                _add_grads_into_flat(flat_grad_local, grads, bundle.params)
                if received is not None:
                    received[0] += 1.0
                    for index, grad in enumerate(grads):
                        if grad is not None:
                            received[index + 1] += 1.0
                del logits
            delta_loss_local = _sum_sq_fp64(flat_grad_local)

        # ----- chi_net (per-rank partial) -----
        method = select_chi_net_method(b_total=_global_b, requested=self.chi_net_method_request)
        estimator = build_estimator(
            method,
            n_hutchinson=self.n_hutchinson,
            distribution=self.hutchinson_distribution,
        )
        # One probe STREAM PER RANK for the same (ckpt, batch). ``probe_seed`` is
        # the per-call seed the sink records as ``hutchinson_seed``; rank 0 -- and
        # so every single-process run -- draws from it directly, rank r > 0 from a
        # seed derived from it and r (``_rank_probe_seed``).
        #
        # The ranks must NOT share one stream (they did, through 0.2.1). Each rank
        # draws the probes for its own shard, so a shared seed puts the same
        # random vectors on the k-th sample of every shard. A Rademacher probe v
        # errs on sample a by sum_{i != j} v_i v_j M_a[i, j] with M_a = J_a J_a^T,
        # hence the same v on two samples gives errors with covariance
        # 2 <offdiag(M_a), offdiag(M_b)>_F (Gaussian: 2 <M_a, M_b>_F, which also
        # counts the diagonals and is never negative). The Gram matrices of
        # different inputs are not orthogonal, so the per-rank errors are
        # positively correlated and the variance of the all-reduced chi_net lies
        # between 1x and world-size x that of INDEPENDENT rank streams on the same
        # shards (not: of a single process -- the two references coincide only
        # when every micro-batch holds one sequence). Where in that range depends
        # on the model and on the probe distribution. Measured only on CPU test
        # models of at most 1.25M parameters, nothing larger: 1.3x at 2 ranks and
        # 2.1x-2.6x at 4 ranks with Rademacher probes, 1.4x-1.9x and 2.6x-3.3x
        # with Gaussian probes.
        #
        # The generator is deliberately on the CPU even for a CUDA run: probes are
        # drawn on the CPU and copied to the device
        # (``vatis.core.probes.make_probe``), so the ``hutchinson_seed`` recorded
        # in the sink identifies the same probe set on every device. A CUDA
        # generator would make one seed mean different probes on CPU vs GPU, and
        # the two runs could then only agree to within the Hutchinson noise floor.
        probe_seed = _per_call_seed(self.seed, ckpt_id, batch_name)
        gen = torch.Generator()
        gen.manual_seed(_rank_probe_seed(probe_seed, get_rank()))
        chi_net_result: ChiNetResult = estimator.compute(
            bundle.model,
            local_batch,
            bundle.forward_fn,
            loss_fn=loss_fn,
            valid_mask_fn=valid_mask_fn,
            params=bundle.params,
            micro_batch_size=self.micro_batch_size,
            generator=gen,
            preconditioner=precond,
        )
        # Permanent tripwire. Both paths now mask with the same closure, so their
        # valid-position counts agree by construction; a disagreement means a
        # masking path drifted apart again (this repo's most expensive bug class,
        # twice) and would rescale chi_pos by the ratio of the two masks. Loud,
        # not logged: a plausible-looking wrong chi_pos is the failure mode.
        if chi_net_result.n_valid_tokens != n_valid_local:
            raise RuntimeError(
                f"chi_net covered {chi_net_result.n_valid_tokens} valid output positions but "
                f"chi_loss / n_valid counted {n_valid_local} (checkpoint {ckpt_id!r}, batch "
                f"{batch_name!r}, estimator {estimator.name!r}). The two must be the same set "
                "-- chi_pos would be rescaled by their ratio. This is an internal invariant "
                "of vatis, not a user error; please report it."
            )
        chi_net_local = chi_net_result.chi_net.to(dtype=torch.float64).to(device)
        if precond is None:
            chi_net_p_local = None
        elif chi_net_result.chi_net_p is None:
            raise RuntimeError(
                f"chi_net estimator {estimator.name!r} was given a preconditioner but "
                "returned no chi_net_p; the preconditioned observables are implemented in "
                "'hutchinson' and 'per_sequence_cv' only"
            )
        else:
            chi_net_p_local = chi_net_result.chi_net_p.to(dtype=torch.float64).to(device)
        # Only a P with a ceiling has a capped part (None otherwise).
        chi_net_p_capped_local = (
            None
            if chi_net_result.chi_net_p_capped is None
            else chi_net_result.chi_net_p_capped.to(dtype=torch.float64).to(device)
        )
        # The estimator's own variance of each of the three numbers under the
        # probe draw (``None``: a single probe per micro-batch, nothing to
        # estimate a spread from). Per-rank partial sums, like the estimates.
        chi_net_var_local = _optional_float(chi_net_result.chi_net_var)
        chi_net_p_var_local = (
            None if chi_net_p_local is None else _optional_float(chi_net_result.chi_net_p_var)
        )
        chi_net_p_capped_var_local = (
            None
            if chi_net_p_capped_local is None
            else _optional_float(chi_net_result.chi_net_p_capped_var)
        )

        # ----- All-reduce: chi_loss, chi_net, n_valid, delta_loss grad -----
        # delta_loss needs the FULL gradient (sum across ranks) to compute its
        # squared norm correctly: ||sum_r g_r||^2, not sum_r ||g_r||^2.
        # We re-weight the local grad and all-reduce-sum, then square locally.
        if is_distributed():
            import torch.distributed as dist

            chi_loss_raw_global = all_reduce_sum_scalar(float(chi_loss_raw_local), device)
            chi_net_global = all_reduce_sum_scalar(float(chi_net_local), device)
            chi_net_p_global = (
                None
                if chi_net_p_local is None
                else all_reduce_sum_scalar(float(chi_net_p_local), device)
            )
            chi_net_p_capped_global = (
                None
                if chi_net_p_capped_local is None
                else all_reduce_sum_scalar(float(chi_net_p_capped_local), device)
            )
            # The variances add across ranks exactly as across micro-batches:
            # every rank draws its own probe stream (``_rank_probe_seed``), so the
            # ranks' partial estimates are independent. Whether a variance exists
            # depends only on ``n_hutchinson`` and the preconditioner, which all
            # ranks share, so either every rank reduces it or none does.
            chi_net_var_global = _all_reduce_optional(chi_net_var_local, device)
            chi_net_p_var_global = _all_reduce_optional(chi_net_p_var_local, device)
            chi_net_p_capped_var_global = _all_reduce_optional(chi_net_p_capped_var_local, device)
            n_valid_global = int(all_reduce_sum_scalar(float(n_valid_local), device))
            # Each rank's grad is grad_theta((1/n_local) Σ_local ℓ); we want
            # grad_theta((1/n_global) Σ_all ℓ) = Σ_r (n_local_r / n_global) *
            # rank_grad_r. So multiply each rank's local grad by
            # (n_local / n_global) before all-reduce-sum.
            if n_valid_global > 0 and n_total_for_loss > 0:
                rank_weight = float(n_total_for_loss) / float(n_valid_global)
            else:
                rank_weight = 0.0
            flat_grad_local.mul_(rank_weight)
            dist.all_reduce(flat_grad_local, op=dist.ReduceOp.SUM)
            if received is not None:
                dist.all_reduce(received, op=dist.ReduceOp.SUM)
            delta_loss_global = float(_sum_sq_fp64(flat_grad_local))
        else:
            chi_loss_raw_global = float(chi_loss_raw_local)
            chi_net_global = float(chi_net_local)
            chi_net_p_global = None if chi_net_p_local is None else float(chi_net_p_local)
            chi_net_p_capped_global = (
                None if chi_net_p_capped_local is None else float(chi_net_p_capped_local)
            )
            chi_net_var_global = chi_net_var_local
            chi_net_p_var_global = chi_net_p_var_local
            chi_net_p_capped_var_global = chi_net_p_capped_var_local
            n_valid_global = n_valid_local
            delta_loss_global = float(delta_loss_local)

        # delta_loss^P = grad L^T P grad L: the same (all-reduced) gradient
        # vector, reduced again with the elementwise P weight. No backward.
        delta_loss_p_global = (
            None if precond is None else float(weighted_sum_sq_fp64(flat_grad_local, precond.diag))
        )
        # ... and once more over the entries at P's ceiling only (exactly zero,
        # without a pass, when there are none).
        delta_loss_p_capped_global: float | None = None
        if precond is not None and precond.ceiling is not None:
            delta_loss_p_capped_global = (
                float(
                    weighted_sum_sq_fp64(
                        flat_grad_local, precond.diag, min_weight=precond.capped_weight_floor
                    )
                )
                if precond.count_capped() > 0
                else 0.0
            )

        wallclock_s = time.perf_counter() - t0
        n_valid_a = max(1, n_valid_global)  # avoid div-by-zero downstream
        # Apply the global N-normalization to the chi_loss raw sum.
        chi_loss_global = chi_loss_raw_global / (float(n_valid_a) ** 2)
        norm = normalization_factor(n_valid_a, n_valid_a)
        chi_loss_norm = chi_loss_global * norm
        chi_net_norm = chi_net_global / norm
        chi_pos_value = float(chi_pos(delta_loss_global, chi_loss_global, chi_net_global))
        # chi_pos^P uses the SAME chi_loss (kernel-independent) and the same
        # normalization convention; chi_pos is invariant under the sqrt(N_A N_B)
        # factors, so the unnormalized components are used here exactly as for raw.
        chi_pos_p_value = (
            None
            if precond is None or delta_loss_p_global is None or chi_net_p_global is None
            else float(chi_pos(delta_loss_p_global, chi_loss_global, chi_net_p_global))
        )

        rows = self._emit_rows(
            ckpt_id=ckpt_id,
            revision=revision,
            batch_a=batch_name,
            batch_b=batch_name,
            n_valid_a=n_valid_a,
            n_valid_b=n_valid_a,
            chi_loss=chi_loss_global,
            chi_loss_normalized=chi_loss_norm,
            chi_net=chi_net_global,
            chi_net_normalized=chi_net_norm,
            delta_loss=delta_loss_global,
            chi_pos_value=chi_pos_value,
            n_hutchinson=self.n_hutchinson,
            hutchinson_seed=probe_seed,
            wallclock_s=wallclock_s,
            precond=precond,
            delta_loss_p=delta_loss_p_global,
            chi_net_p=chi_net_p_global,
            chi_pos_p_value=chi_pos_p_value,
            delta_loss_p_capped=delta_loss_p_capped_global,
            chi_net_p_capped=chi_net_p_capped_global,
            chi_net_method=estimator.name,
            # What the estimator that ran says it drew its probes from -- not the
            # argument this analyzer was constructed with.
            hutchinson_distribution=_optional_str(chi_net_result.extras.get("distribution")),
            chi_net_stderr=_sqrt_optional(chi_net_var_global),
            chi_net_p_stderr=_sqrt_optional(chi_net_p_var_global),
            chi_net_p_capped_stderr=_sqrt_optional(chi_net_p_capped_var_global),
        )

        # Cache the (already-reduced) full gradient for cross-pair use.
        # On non-distributed runs flat_grad_local is the full grad (unscaled).
        # On distributed runs we already wrote the reweighted+reduced version.
        self._last_n_valid = n_valid_a
        self._last_no_gradient = (
            ()
            if received is None or float(received[0]) == 0.0
            else tuple(i for i, count in enumerate(received[1:].tolist()) if count == 0.0)
        )
        return rows, flat_grad_local.detach()

    # ------------------------------------------------------------- cross pair

    def _compute_cross_pair(
        self,
        *,
        ckpt_id: str,
        revision: str,
        name_a: str,
        name_b: str,
        g_a: torch.Tensor,
        g_b: torch.Tensor,
        n_valid_a: int,
        n_valid_b: int,
        bundle: ModelBundle,
        precond: Preconditioner | None = None,
    ) -> list[ResultRow]:
        """Compute the cross delta_loss + chi_pos using cached self gradients.

        We don't recompute chi_loss / chi_net for the cross pair — those are
        per-batch quantities that were already emitted in the self pass.
        chi_pos cross is δL(A, B) / (chi_loss_A * chi_net_B) — but in
        practice users want this normalized so the √(N_A·N_B) factors cancel.
        We follow the convention from CLAUDE.md and use the unnormalized
        chi_loss / chi_net (since chi_pos is invariant under normalization).

        Contract: this method assumes that the self pairs for ``name_a`` and
        ``name_b`` have already been computed (so ``_chi_loss_cache`` and
        ``_chi_net_cache`` contain entries for both names). The public
        :meth:`run` always runs the self loop before any cross pair, so this
        is satisfied automatically. The assertion below catches direct
        callers that violate the ordering — silently falling back to a zero
        chi_loss / chi_net would mask the bug.
        """
        missing = [
            name
            for name in (name_a, name_b)
            if name not in self._chi_loss_cache
            or name not in self._chi_net_cache
            or name not in self._chi_net_method_cache
            or (precond is not None and name not in self._chi_net_p_cache)
        ]
        if missing:
            raise RuntimeError(
                f"_compute_cross_pair called for cross pair ({name_a!r}, {name_b!r}) "
                f"but the self-pair cache is missing {missing}. "
                f"Call _compute_self_pair for both batches first; "
                f"Analyzer.run() does this automatically."
            )

        # Cross delta_loss is the dot product of the cached flat grads. We
        # accumulate in fp64 via ``sum(dtype=...)`` rather than casting both
        # full-P vectors to fp64 first (which would materialize two P-sized
        # fp64 temporaries — ~21 GiB for a 1.4B model).
        delta_loss_cross_value = float(_dot_fp64(g_a, g_b))

        # Per-batch chi_loss / chi_net stashed by the self loop in
        # ``_emit_rows``. The contract above guarantees both keys exist.
        chi_loss_a = self._chi_loss_cache[name_a]
        chi_loss_b = self._chi_loss_cache[name_b]
        chi_net_a = self._chi_net_cache[name_a]
        chi_net_b = self._chi_net_cache[name_b]

        # cross chi_loss / chi_net per the mini-batch derivation (geometric
        # mean of the two batches' magnitudes).
        chi_loss_cross_value = (chi_loss_a * chi_loss_b) ** 0.5
        chi_net_cross_value = (chi_net_a * chi_net_b) ** 0.5
        # The two batches were probed with different streams (the per-call seed
        # mixes in the batch name), so their errors are independent and the
        # geometric mean inherits them to first order.
        chi_net_cross_stderr = _geometric_mean_stderr(
            chi_net_a,
            self._chi_net_stderr_cache.get(name_a),
            chi_net_b,
            self._chi_net_stderr_cache.get(name_b),
        )
        chi_pos_value = float(
            chi_pos(delta_loss_cross_value, chi_loss_cross_value, chi_net_cross_value)
        )

        # Preconditioned observables, same shape: the P-weighted gradient dot and the
        # geometric mean of the two batches' chi_net^P (i.e.
        # ||P^{1/2} J^A||_F * ||P^{1/2} J^B||_F, as the raw cross pair does).
        delta_loss_p_cross: float | None = None
        chi_net_p_cross: float | None = None
        chi_pos_p_value: float | None = None
        chi_net_p_cross_stderr: float | None = None
        if precond is not None:
            delta_loss_p_cross = float(weighted_dot_fp64(g_a, g_b, precond.diag))
            chi_net_p_cross = (self._chi_net_p_cache[name_a] * self._chi_net_p_cache[name_b]) ** 0.5
            chi_net_p_cross_stderr = _geometric_mean_stderr(
                self._chi_net_p_cache[name_a],
                self._chi_net_p_stderr_cache.get(name_a),
                self._chi_net_p_cache[name_b],
                self._chi_net_p_stderr_cache.get(name_b),
            )
            chi_pos_p_value = float(
                chi_pos(delta_loss_p_cross, chi_loss_cross_value, chi_net_p_cross)
            )

        # The same two quantities with P restricted to its ceiling entries. The
        # dot is linear in P, so this is the capped part of delta_loss_P(A, B).
        # The cross chi_net_P is a geometric mean, which is not: its capped row is
        # the geometric mean of the two batches' capped parts.
        delta_loss_p_capped_cross: float | None = None
        chi_net_p_capped_cross: float | None = None
        chi_net_p_capped_cross_stderr: float | None = None
        if precond is not None and precond.ceiling is not None:
            delta_loss_p_capped_cross = (
                float(
                    weighted_dot_fp64(
                        g_a, g_b, precond.diag, min_weight=precond.capped_weight_floor
                    )
                )
                if precond.count_capped() > 0
                else 0.0
            )
            chi_net_p_capped_cross = (
                self._chi_net_p_capped_cache[name_a] * self._chi_net_p_capped_cache[name_b]
            ) ** 0.5
            chi_net_p_capped_cross_stderr = _geometric_mean_stderr(
                self._chi_net_p_capped_cache[name_a],
                self._chi_net_p_capped_stderr_cache.get(name_a),
                self._chi_net_p_capped_cache[name_b],
                self._chi_net_p_capped_stderr_cache.get(name_b),
            )

        # One estimator for the whole run with today's selector (the request, or
        # the default, for every batch). The estimator is still resolved per
        # batch, so if two batches of a pair ever differ the row names both, in
        # pair order, instead of hiding it.
        chi_net_method = _pair_label(
            self._chi_net_method_cache[name_a], self._chi_net_method_cache[name_b]
        )
        # The probe distribution is one setting of the analyzer, so the two
        # batches agree; it is taken from what each batch's estimator reported
        # all the same, and a disagreement would be named, not hidden.
        hutchinson_distribution = _pair_label(
            self._hutchinson_distribution_cache.get(name_a),
            self._hutchinson_distribution_cache.get(name_b),
        )

        return self._emit_rows(
            ckpt_id=ckpt_id,
            revision=revision,
            batch_a=name_a,
            batch_b=name_b,
            n_valid_a=n_valid_a,
            n_valid_b=n_valid_b,
            chi_loss=chi_loss_cross_value,
            chi_loss_normalized=chi_loss_cross_value * normalization_factor(n_valid_a, n_valid_b),
            chi_net=chi_net_cross_value,
            chi_net_normalized=chi_net_cross_value / normalization_factor(n_valid_a, n_valid_b),
            delta_loss=delta_loss_cross_value,
            chi_pos_value=chi_pos_value,
            n_hutchinson=self.n_hutchinson,
            hutchinson_seed=None,
            wallclock_s=None,
            precond=precond,
            delta_loss_p=delta_loss_p_cross,
            chi_net_p=chi_net_p_cross,
            chi_pos_p_value=chi_pos_p_value,
            delta_loss_p_capped=delta_loss_p_capped_cross,
            chi_net_p_capped=chi_net_p_capped_cross,
            chi_net_method=chi_net_method,
            hutchinson_distribution=hutchinson_distribution,
            chi_net_stderr=chi_net_cross_stderr,
            chi_net_p_stderr=chi_net_p_cross_stderr,
            chi_net_p_capped_stderr=chi_net_p_capped_cross_stderr,
        )

    # -------------------------------------------------------- update response

    def _resolve_update_response(self, ckpt_id: str) -> OptimizerUpdate | None:
        """The update the optimizer applied at this checkpoint, or ``None``."""
        if self.update_response is None:
            return None
        update = self.update_response(ckpt_id)
        if update is not None and not isinstance(update, OptimizerUpdate):
            raise TypeError(
                f"update_response factory returned {type(update).__name__} for checkpoint "
                f"{ckpt_id!r}; expected an OptimizerUpdate or None"
            )
        return update

    def _compute_update_response(
        self,
        *,
        bundle: ModelBundle,
        ckpt_id: str,
        revision: str,
        update: OptimizerUpdate,
        update_name: str,
        no_gradient: Sequence[int],
        grads: dict[str, torch.Tensor],
        n_valid: dict[str, int],
        raw_batches: dict[str, Any],
    ) -> list[ResultRow]:
        """Rows for the update that was taken, on (update batch, B) for every B.

        ``update_name`` is the eval batch that is the step's update batch
        (:func:`_locate_update_batch`) and ``no_gradient`` the indices of the
        parameters it gave no gradient to. ``grads`` are the full (all-reduced)
        loss gradients the self pairs built, so every DDP rank computes the same
        dot products and nothing is reduced again; only the two loss evaluations
        of the realized change run on the rank's shard and are all-reduced.

        The realized-change work is done only for the rows that were asked for:
        ``theta_{t+1}`` is loaded only if one of the three is among
        ``self.observables``, and the two forward passes per batch run only for
        ``loss_change`` / ``loss_change_residual``.
        """
        missing = [name for name in raw_batches if name not in grads]
        if missing:
            raise RuntimeError(
                f"checkpoint {ckpt_id!r}: no gradient was kept for {missing}; the update "
                "response needs the loss gradient of every eval batch"
            )
        names = _parameter_names(bundle)
        try:
            terms = update.response(
                grads[update_name],
                grads,
                bundle.params,
                no_gradient={index: names[index] for index in no_gradient},
            )
        except ValueError as exc:
            raise ValueError(f"checkpoint {ckpt_id!r}: {exc}") from exc

        values: dict[str, dict[str, float]] = {
            name: {
                "loss_change_first_order_grad": t.grad,
                "loss_change_first_order_hist": t.hist,
                "loss_change_first_order_wd": t.wd,
                "loss_change_first_order": t.total,
            }
            for name, t in terms.items()
        }

        # The step's learning rate and clip factor, on the update batch's own
        # pair. Both come from the rule and the all-reduced gradient, so every
        # DDP rank holds the same numbers.
        lowest, highest = update.learning_rates()
        if lowest == highest:
            values[update_name]["update_lr"] = lowest
        else:
            values[update_name]["update_lr_min"] = lowest
            values[update_name]["update_lr_max"] = highest
        values[update_name]["update_clip_factor"] = update.clip_coefficient(grads[update_name])

        requested = set(self.observables)
        need_loss = bool(requested & {"loss_change", "loss_change_residual"})
        need_weights = bool(requested & {"loss_change_first_order_weights", "loss_change_residual"})
        successor = (
            update.next_weights()
            if update.next_weights is not None and (need_loss or need_weights)
            else None
        )
        if successor is not None:
            if not isinstance(successor, ModelBundle):
                raise TypeError(
                    f"next_weights returned {type(successor).__name__} for checkpoint "
                    f"{ckpt_id!r}; expected a ModelBundle"
                )
            successor.model.eval()
            for name, raw in raw_batches.items():
                if need_weights:
                    from_weights = float(
                        weight_delta_dot_fp64(grads[name], successor.params, bundle.params)
                    )
                    values[name]["loss_change_first_order_weights"] = from_weights
                if need_loss:
                    change = self._mean_loss(successor, raw) - self._mean_loss(bundle, raw)
                    values[name]["loss_change"] = change
                if need_loss and need_weights:
                    values[name]["loss_change_residual"] = change - from_weights

        rows: list[ResultRow] = []
        for name, by_observable in values.items():
            for observable in self.observables:
                if observable not in by_observable:
                    continue
                rows.append(
                    ResultRow(
                        checkpoint_id=ckpt_id,
                        revision=revision,
                        batch_a=update_name,
                        batch_b=name,
                        n_valid_a=n_valid[update_name],
                        n_valid_b=n_valid[name],
                        observable=observable,
                        value=float(by_observable[observable]),
                        n_hutchinson=None,
                        hutchinson_seed=None,
                        wallclock_s=None,
                        preconditioner=(
                            update.descriptor
                            if observable in UPDATE_RESPONSE_OBSERVABLES + UPDATE_STEP_OBSERVABLES
                            else RAW_DESCRIPTOR
                        ),
                    )
                )
        return rows

    def _mean_loss(self, bundle: ModelBundle, raw: Any) -> float:
        """The mean loss of one eval batch at ``bundle``'s weights, in fp64.

        Forward only. The same loss and the same mask as every other
        accumulator (``_label_folded_loss_fn`` / ``_intersected_valid_mask_fn``),
        assembled over micro-batches and DDP ranks as a token-weighted mean. The
        logits are upcast to fp64 before the loss, so the difference of two such
        values — the realized loss change of one step, often five orders of
        magnitude below the loss itself — is not limited by fp32 round-off in
        the cross entropy.
        """
        device = next(bundle.model.parameters()).device
        local_batch, _local_b, _global_b = shard_batch(
            raw, rank=get_rank(), world_size=get_world_size()
        )
        local_batch = move_batch(local_batch, device)
        valid_mask_fn = _intersected_valid_mask_fn(bundle.valid_mask_fn, self.ignore_index)
        loss_fn = _label_folded_loss_fn(bundle.loss_fn, valid_mask_fn, self.ignore_index)
        loss_sum = 0.0
        n_tokens = 0
        for _start, _stop, micro in iter_micro_batches(local_batch, self.micro_batch_size):
            with torch.no_grad():
                logits = bundle.forward_fn(bundle.model, micro)
            n_micro = int(valid_mask_fn(micro, logits).sum().item())
            if n_micro == 0:
                continue
            # A leaf that requires grad: the label-folding closure verifies a
            # rewritten mask through the gradient w.r.t. the logits.
            logits64 = logits.detach().to(dtype=torch.float64).requires_grad_(True)
            loss_sum += float(loss_fn(logits64, micro).detach()) * n_micro
            n_tokens += n_micro
            del logits, logits64
        if is_distributed():
            loss_sum = all_reduce_sum_scalar(loss_sum, device)
            n_tokens = int(all_reduce_sum_scalar(float(n_tokens), device))
        if n_tokens == 0:
            raise ValueError("cannot evaluate a loss on a batch without valid tokens")
        return loss_sum / n_tokens

    # ------------------------------------------------------------- emission

    _last_n_valid: int = 0
    _last_no_gradient: tuple[int, ...] = ()

    def _emit_rows(
        self,
        *,
        ckpt_id: str,
        revision: str,
        batch_a: str,
        batch_b: str,
        n_valid_a: int,
        n_valid_b: int,
        chi_loss: float,
        chi_loss_normalized: float,
        chi_net: float,
        chi_net_normalized: float,
        delta_loss: float,
        chi_pos_value: float,
        n_hutchinson: int | None,
        hutchinson_seed: int | None,
        wallclock_s: float | None,
        precond: Preconditioner | None = None,
        delta_loss_p: float | None = None,
        chi_net_p: float | None = None,
        chi_pos_p_value: float | None = None,
        delta_loss_p_capped: float | None = None,
        chi_net_p_capped: float | None = None,
        chi_net_method: str | None = None,
        hutchinson_distribution: str | None = None,
        chi_net_stderr: float | None = None,
        chi_net_p_stderr: float | None = None,
        chi_net_p_capped_stderr: float | None = None,
    ) -> list[ResultRow]:
        # Stash the per-batch chi_loss / chi_net for the cross pair use.
        # Caches are eagerly initialized in __init__.
        if batch_a == batch_b:
            self._chi_loss_cache[batch_a] = chi_loss
            self._chi_net_cache[batch_a] = chi_net
            self._chi_net_stderr_cache[batch_a] = chi_net_stderr
            if chi_net_method is not None:
                self._chi_net_method_cache[batch_a] = chi_net_method
            self._hutchinson_distribution_cache[batch_a] = hutchinson_distribution
            if chi_net_p is not None:
                self._chi_net_p_cache[batch_a] = chi_net_p
                self._chi_net_p_stderr_cache[batch_a] = chi_net_p_stderr
            if chi_net_p_capped is not None:
                self._chi_net_p_capped_cache[batch_a] = chi_net_p_capped
                self._chi_net_p_capped_stderr_cache[batch_a] = chi_net_p_capped_stderr

        values: dict[str, float] = {
            "chi_loss": chi_loss,
            "chi_loss_normalized": chi_loss_normalized,
            "chi_net": chi_net,
            "chi_net_normalized": chi_net_normalized,
            "delta_loss": delta_loss,
            "chi_pos": chi_pos_value,
        }
        descriptors = dict.fromkeys(values, RAW_DESCRIPTOR)
        if precond is not None:
            if delta_loss_p is None or chi_net_p is None or chi_pos_p_value is None:
                raise RuntimeError(
                    "a preconditioner was configured but the preconditioned triple is "
                    f"incomplete for ({batch_a!r}, {batch_b!r}): "
                    f"delta_loss_P={delta_loss_p}, chi_net_P={chi_net_p}, "
                    f"chi_pos_P={chi_pos_p_value}"
                )
            values["delta_loss_P"] = delta_loss_p
            values["chi_net_P"] = chi_net_p
            values["chi_pos_P"] = chi_pos_p_value
            for name in P_OBSERVABLES:
                descriptors[name] = precond.descriptor
            if precond.ceiling is not None:
                if delta_loss_p_capped is None or chi_net_p_capped is None:
                    raise RuntimeError(
                        f"preconditioner {precond.descriptor!r} has a ceiling but the capped "
                        f"parts are missing for ({batch_a!r}, {batch_b!r}): "
                        f"delta_loss_P_capped={delta_loss_p_capped}, "
                        f"chi_net_P_capped={chi_net_p_capped}"
                    )
                values["delta_loss_P_capped"] = delta_loss_p_capped
                values["chi_net_P_capped"] = chi_net_p_capped
                for name in P_CAPPED_OBSERVABLES:
                    descriptors[name] = precond.descriptor

        # |chi_pos| <= 1 is a Cauchy-Schwarz consequence, so a violation is not a
        # measurement — it is an inconsistency between the three components, and
        # the only symptom a user ever gets. Warn rather than raise: chi_net is a
        # stochastic estimate and a legitimate chi_pos near 1 can overshoot at
        # small n_hutchinson, so refusing to publish would break honest runs.
        for name in ("chi_pos", "chi_pos_P"):
            value = values.get(name)
            if value is not None and chi_pos_exceeds_cauchy_schwarz(value):
                warnings.warn(
                    f"{name} = {value:.6g} for ({batch_a!r}, {batch_b!r}) at checkpoint "
                    f"{ckpt_id!r} is outside the [-1, 1] range the LNP definition "
                    "guarantees. Cauchy-Schwarz makes that impossible for a consistent "
                    "triple, so the three components are not describing the same loss. "
                    "Most likely: loss_fn is not mean-reduced over the same valid tokens "
                    "chi_loss normalizes by (a sum-reduced CE inflates chi_pos by the "
                    "square of the valid-token count per micro-batch), or loss_fn's "
                    "ignore_index differs from the analyzer's. "
                    "A very small n_hutchinson can also overshoot legitimately.",
                    UserWarning,
                    stacklevel=2,
                )

        # Standard error of each sampled value under the probe draw. ``chi_loss``
        # and ``delta_loss`` (and their P / capped forms) involve no probe and
        # get none. ``chi_net_normalized`` is ``chi_net`` divided by a constant.
        # ``chi_pos = delta_loss / (chi_loss * chi_net)`` has ``chi_net`` as its
        # only sampled factor, so to first order its relative error is that of
        # ``chi_net`` -- for the cross pair, that of the geometric mean handed in.
        stderrs: dict[str, float | None] = {
            "chi_net": chi_net_stderr,
            "chi_net_normalized": (
                None
                if chi_net_stderr is None
                else chi_net_stderr / normalization_factor(n_valid_a, n_valid_b)
            ),
            "chi_pos": _ratio_stderr(chi_pos_value, chi_net, chi_net_stderr),
        }
        if precond is not None and chi_net_p is not None and chi_pos_p_value is not None:
            stderrs["chi_net_P"] = chi_net_p_stderr
            stderrs["chi_pos_P"] = _ratio_stderr(chi_pos_p_value, chi_net_p, chi_net_p_stderr)
            stderrs["chi_net_P_capped"] = chi_net_p_capped_stderr
        n_ranks = get_world_size()

        rows: list[ResultRow] = []
        for name in self.observables:
            if name not in values:
                # A P observable was requested but this checkpoint has no P (the
                # factory declined — see the step-0 policy), or a capped row was
                # requested for a P without a ceiling. Skipping the row is
                # deliberate: emitting a raw-valued or 1/eps-valued row under the
                # 'adamw' descriptor would be a wrong answer that looks plausible.
                continue
            # eps / beta2 of the P and the fraction of its coordinates at the
            # ceiling go on the rows computed with it, not on the raw rows.
            p_of_row = precond if descriptors[name] != RAW_DESCRIPTOR else None
            rows.append(
                ResultRow(
                    checkpoint_id=ckpt_id,
                    revision=revision,
                    batch_a=batch_a,
                    batch_b=batch_b,
                    n_valid_a=n_valid_a,
                    n_valid_b=n_valid_b,
                    observable=name,
                    value=float(values[name]),
                    n_hutchinson=n_hutchinson,
                    hutchinson_seed=hutchinson_seed,
                    wallclock_s=wallclock_s,
                    preconditioner=descriptors[name],
                    chi_net_method=chi_net_method,
                    n_ranks=n_ranks,
                    preconditioner_eps=None if p_of_row is None else p_of_row.eps,
                    preconditioner_beta2=None if p_of_row is None else p_of_row.beta2,
                    preconditioner_capped_fraction=(
                        None if p_of_row is None else p_of_row.capped_fraction()
                    ),
                    value_probe_stderr=stderrs.get(name),
                    micro_batch_size=self.micro_batch_size,
                    hutchinson_distribution=hutchinson_distribution,
                )
            )
        return rows


# ------------------------------------------------------------- helpers


def _normalize_preconditioner(
    spec: PreconditionerLike | None,
) -> PreconditionerLike | None:
    """Validate a ``preconditioner`` argument, mapping "no P" to ``None``.

    Accepted string values are only ``"none"`` (equivalent to ``None``) and
    ``"identity"``. In particular ``"adamw"`` is *not* accepted here: the
    analyzer sees ``ModelBundle``s and has no access to a checkpoint's saved
    optimizer state, so an Adam ``P`` must come from the source that owns that
    state (``LocalBundleSource.adamw_preconditioner()``). Silently accepting
    ``"adamw"`` and falling back to something else is exactly the wrong-answer
    mode this rejection prevents.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec == RAW_DESCRIPTOR:
            return None
        if spec == "identity":
            return spec
        raise ValueError(
            f"preconditioner={spec!r} is not a recognized shorthand; use None/'none' "
            "for the raw observables only, 'identity' for P = I, a Preconditioner instance, or a "
            "callable(checkpoint_id) -> Preconditioner | None. For an Adam P from a "
            "checkpoint bundle use LocalBundleSource.adamw_preconditioner(), which has "
            "access to the saved optimizer state."
        )
    if isinstance(spec, Preconditioner):
        return spec
    if callable(spec):
        return spec
    raise TypeError(
        f"unsupported preconditioner type: {type(spec).__name__}; expected str, "
        "Preconditioner, callable(checkpoint_id) -> Preconditioner | None, or None"
    )


def _locate_update_batch(update: OptimizerUpdate, raw_batches: dict[str, Any], ckpt_id: str) -> str:
    """The name of the eval batch that is the batch the step consumed.

    Called before any forward pass of the checkpoint, so both ways this can
    fail cost nothing: a rule that does not carry its update batch, and an
    update batch that is not among the eval batches. Batches are compared by
    their tensors (:func:`_same_batch`), never by name; the first match wins.
    """
    if update.update_batch is None:
        raise ValueError(
            f"checkpoint {ckpt_id!r}: the update rule does not carry the batch the step "
            "consumed (OptimizerUpdate.update_batch), so the analyzer cannot tell which "
            "eval batch's gradient the update is built from"
        )
    for name, raw in raw_batches.items():
        if _same_batch(raw, update.update_batch):
            return name
    raise ValueError(
        f"checkpoint {ckpt_id!r}: none of the eval batches {sorted(raw_batches)} is "
        "the batch this step consumed, so its gradient — which the update is "
        "built from — is not among the ones measured. Add the update batch to "
        "eval_batches (for a checkpoint bundle: source.batch_a_spec()); it is "
        "matched by its tensors, not by its name."
    )


def _parameter_names(bundle: ModelBundle) -> list[str]:
    """A display name for each entry of ``bundle.params`` (for error messages)."""
    by_identity = {id(p): name for name, p in bundle.model.named_parameters()}
    return [by_identity.get(id(p), f"params[{i}]") for i, p in enumerate(bundle.params)]


def _same_batch(a: Any, b: Any) -> bool:
    """Whether two batches hold the same tensors (same structure, shapes, values).

    Used to find which eval batch is the batch a step consumed. Devices and
    object identity do not matter; a tensor's dtype does.
    """
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return (
            a.dtype == b.dtype
            and tuple(a.shape) == tuple(b.shape)
            and bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_batch(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same_batch(x, y) for x, y in zip(a, b, strict=True))
    return False


def _optional_float(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _pair_label(a: str | None, b: str | None) -> str | None:
    """What a cross pair records for a per-batch setting: the common value, or
    ``"<a>+<b>"`` in pair order when the two batches differ. ``None`` when it is
    unknown for either batch."""
    if a is None or b is None:
        return None
    return a if a == b else f"{a}+{b}"


def _sqrt_optional(variance: float | None) -> float | None:
    return None if variance is None else math.sqrt(max(variance, 0.0))


def _all_reduce_optional(value: float | None, device: torch.device) -> float | None:
    """All-reduce-sum a per-rank partial that may be absent (on every rank alike)."""
    return None if value is None else all_reduce_sum_scalar(value, device)


def _ratio_stderr(
    ratio: float,
    denominator: float,
    denominator_stderr: float | None,
) -> float | None:
    """Standard error of ``ratio = c / denominator`` when only the denominator is sampled.

    To first order ``|ratio| * stderr(denominator) / |denominator|``: the
    relative error of the denominator carries over unchanged. ``None`` when the
    denominator has no standard error, or is zero (the ratio is then the
    degenerate ``chi_pos = 0`` convention, not a function one can differentiate).

    First order means the relative error must be small: the neglected terms are
    of relative size ``(stderr / denominator)**2`` in the variance. A row whose
    ``chi_net`` is known to 30% does not have a ``chi_pos`` error known to
    better than a factor.
    """
    if denominator_stderr is None or denominator == 0.0:
        return None
    return abs(ratio) * denominator_stderr / abs(denominator)


def _geometric_mean_stderr(
    a: float,
    a_stderr: float | None,
    b: float,
    b_stderr: float | None,
) -> float | None:
    """Standard error of ``sqrt(a * b)`` for independent ``a``, ``b``, to first order.

    ``sqrt(a b) * 0.5 * sqrt((s_a / a)**2 + (s_b / b)**2)`` -- half the relative
    errors, added in quadrature. This is the cross pair's ``chi_net``: the two
    batches are probed with different streams, hence independently.

    ``None`` when either side has no standard error. If a factor is zero the
    square root cannot be differentiated there; but an estimate that is a mean
    of squares is zero only if every probe value was zero, and then its own
    standard error is zero and the product is zero for every probe draw -- the
    result is ``0.0``. A zero factor with a non-zero standard error cannot come
    out of the estimators; it yields ``None`` rather than a made-up number.
    """
    if a_stderr is None or b_stderr is None:
        return None
    if a <= 0.0 or b <= 0.0:
        exact_zero = (a == 0.0 and a_stderr == 0.0) or (b == 0.0 and b_stderr == 0.0)
        return 0.0 if exact_zero else None
    return math.sqrt(a * b) * 0.5 * math.hypot(a_stderr / a, b_stderr / b)


def _should_offload_cross_grads(
    *,
    n_params: int,
    weights_bytes: int,
    n_cross_batches: int,
    total_device_bytes: int,
    fraction: float = 0.85,
) -> bool:
    """Whether cached cross-pair gradients should be offloaded to host RAM.

    Pure size arithmetic (no device access) so it is unit-testable. The cross
    dot needs ``n_cross_batches`` gradients; while one batch runs its backward
    the others' fp32 gradients (``4·P`` bytes each) sit resident alongside that
    backward's footprint — weights + the fp32 flat-grad result + a transient
    grad tuple + autograd/cuBLAS workspace, estimated as ``weights + 8·P``
    (measured ~``weights + 9.5·P`` for a 7B bf16 model; ``8·P`` is a deliberately
    optimistic floor, paired with the conservative ``fraction``). Offload when
    that total would exceed ``fraction`` of device memory.

    Returns ``False`` for ``n_cross_batches < 2`` — no cross dot holds two
    gradients at once, so there is nothing to offload.
    """
    if n_cross_batches < 2:
        return False
    self_pair_bytes = weights_bytes + 8 * n_params
    coexisting_cached = (n_cross_batches - 1) * 4 * n_params
    return (self_pair_bytes + coexisting_cached) > fraction * total_device_bytes


def _zero_param_vector(params: list[torch.nn.Parameter], device: torch.device) -> torch.Tensor:
    sizes = [p.numel() for p in params]
    if not sizes:
        raise ValueError("model has no parameters")
    return torch.zeros(sum(sizes), dtype=torch.float32, device=device)


def _sum_sq_fp64(flat: torch.Tensor, chunk: int = 1 << 25) -> torch.Tensor:
    """``sum(flat**2)`` accumulated in fp64, chunked so it never materializes a
    full-length fp64 copy.

    The naive ``flat.to(float64).pow(2).sum()`` — and even
    ``flat.pow(2).sum(dtype=float64)`` — allocates a P*8-byte temporary (~10 GiB
    at 1.4B params, which OOMs a 40 GB GPU). Chunking caps the extra to ~``chunk``
    elements while keeping the reduction in fp64.
    """
    total = torch.zeros((), dtype=torch.float64, device=flat.device)
    for i in range(0, flat.numel(), chunk):
        c = flat[i : i + chunk].to(dtype=torch.float64)
        total = total + (c * c).sum()
    return total


def _dot_fp64(a: torch.Tensor, b: torch.Tensor, chunk: int = 1 << 25) -> torch.Tensor:
    """``sum(a*b)`` accumulated in fp64, chunked (avoids the two full-length fp64
    copies the cross-pair dot would otherwise make — ~21 GiB at 1.4B params)."""
    total = torch.zeros((), dtype=torch.float64, device=a.device)
    for i in range(0, a.numel(), chunk):
        ca = a[i : i + chunk].to(dtype=torch.float64)
        cb = b[i : i + chunk].to(dtype=torch.float64)
        total = total + (ca * cb).sum()
    return total


def _add_grads_into_flat(
    flat: torch.Tensor,
    grads: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None],
    params: list[torch.nn.Parameter],
) -> None:
    offset = 0
    for p, g in zip(params, grads, strict=False):
        n = p.numel()
        if g is not None:
            flat[offset : offset + n] += g.detach().to(dtype=torch.float32).reshape(-1)
        offset += n


def _intersected_valid_mask_fn(
    valid_mask_fn: ValidMaskFn,
    ignore_index: int,
) -> EffectiveMaskFn:
    """Build THE mask every accumulator uses: ``valid_mask_fn`` AND the labels.

    A position counts iff the bundle's ``valid_mask_fn`` accepts it **and** its
    label is not ``ignore_index``. Both conditions have to hold in all three
    places, not only in ``chi_loss``:

    - a position whose label is ``ignore_index`` contributes nothing to the loss,
      so its Jacobian rows have no business in ``chi_net = ||grad_theta f||_F^2``
      (nor in ``chi_net^P``) — hence the probe handed to the estimator is masked
      with *this* mask, not with ``valid_mask_fn`` alone;
    - the per-micro-batch weight that reassembles ``grad_theta L`` from
      micro-batch gradients has to be counted over the same set the global
      denominator (``n_valid``, and hence ``chi_loss``) uses, or ``delta_loss``
      comes out rescaled *and* dependent on ``micro_batch_size``.

    The two ways the masks can disagree are handled in two places:

    - ``valid_mask_fn`` **wider** than the labels (it accepts a position whose
      label is ``ignore_index``): the intersection here removes the position
      from the counts and the probes, and ``loss_fn`` never counted it.
    - ``valid_mask_fn`` **stricter** than the labels (it rejects a position
      whose label is valid): this mask restricts the counts and the probes, but
      it cannot restrict ``loss_fn``, which only sees the batch. That half is
      :func:`_label_folded_loss_fn`: the same mask is written into the labels
      the loss is evaluated on, so the labels are the single source of truth for
      what enters the loss.

    Returning one closure and threading it through every accumulator is what
    makes "padding is handled at one place" (``CLAUDE.md`` §Conventions) true
    rather than aspirational. The mask itself is still computed by
    :func:`vatis.core.normalization.valid_token_mask`; this only composes it with
    whatever the bundle supplied.

    The result is never ``None`` (unlike a raw ``valid_mask_fn``): "no mask" and
    "everything valid" are the same thing once the labels are folded in, and
    collapsing the two removes a branch from every call site.
    """

    def fn(batch: Any, logits: torch.Tensor) -> torch.Tensor:
        vmask = valid_mask_fn(batch, logits)
        targets = _extract_targets(batch)
        return valid_token_mask(targets, ignore_index=ignore_index, attention_mask=vmask)

    return fn


def _label_folded_loss_fn(
    loss_fn: LossFn,
    effective_mask_fn: EffectiveMaskFn,
    ignore_index: int,
) -> LossFn:
    """Build THE loss every backward uses: ``loss_fn`` on labels that carry the mask.

    The counterpart of :func:`_intersected_valid_mask_fn` for the loss. A
    ``loss_fn(logits, batch)`` decides what it counts from the batch's labels
    alone, so a mask that rejects a position whose label is valid would restrict
    ``n_valid``, ``chi_loss`` and the ``chi_net`` probes while the gradient stayed
    that of the unrestricted loss. The closure returned here evaluates
    ``loss_fn`` on a batch whose labels are ``ignore_index`` wherever the
    effective mask rejects the position (:func:`_fold_mask_into_labels`) — the
    procedure HuggingFace ``Trainer`` / TRL use, where every mask is written into
    the labels before the loss and the loss is a mean over the non-ignored
    targets. A stricter mask then gives exactly what the same restriction written
    into the labels gives, at every ``micro_batch_size``.

    When the mask rejects no valid label — the shipped loaders on any batch that
    carries labels, where ``valid_mask_fn`` *is* ``labels != ignore_index`` — the
    batch is handed to ``loss_fn`` untouched, the same object, so nothing changes
    bit for bit. (A label-less dict with a padded ``attention_mask`` does gain a
    ``labels`` entry; it holds the very targets ``causal_lm_loss`` would have
    derived itself, so the values are unchanged there as well.)

    When labels were rewritten, the closure also checks that ``loss_fn`` honoured
    them: the gradient of the loss w.r.t. the logits must vanish at every
    rejected position. A ``loss_fn`` that reads its targets from somewhere else,
    or does not ignore ``ignore_index``, would otherwise be silently unmasked;
    that is refused with a ``ValueError``. So is a loss that does not depend on
    the logits it was handed at all (a ``loss_fn`` that runs its own forward):
    there the check has nothing to look at, and "could not verify" must not pass
    for "verified". The check costs one backward through the loss head (not
    through the model) and runs only in this case.

    ``ignore_index`` has to be the value ``loss_fn`` itself ignores; the analyzer
    does not guess a loss's ignore value. A batch whose rejected labels carry
    another sentinel (a value that is no class index and not ``ignore_index``) is
    refused before the loss is called, naming the value to pass as
    ``ignore_index``; any exception the loss raises on rewritten labels is
    re-raised with that explanation and the original chained.
    """

    def fn(logits: torch.Tensor, batch: Any) -> torch.Tensor:
        mask = effective_mask_fn(batch, logits)
        folded = _fold_mask_into_labels(batch, mask, ignore_index)
        if folded is batch:
            return loss_fn(logits, batch)

        # Labels are about to reach ``loss_fn`` with ``ignore_index`` written into
        # them. That only works if ``ignore_index`` is the value ``loss_fn``
        # ignores -- the analyzer cannot know the loss's own ignore value and
        # does not guess it. Two layers, in this order:
        #
        # 1. Before the loss is called: a label the mask rejects that cannot be a
        #    class index and is not ``ignore_index`` is another sentinel, i.e. the
        #    batch (and so its loss) uses a different ignore value. Refusing here
        #    matters on CUDA, where the loss would die in a device-side assert
        #    that cannot be caught and turned into a message.
        # 2. If the loss raises anyway, say what vatis did to its labels and keep
        #    the original exception chained.
        rejected_labels = _extract_targets(batch)[~mask]
        rejected_labels = rejected_labels[rejected_labels != ignore_index]
        n_classes = logits.shape[-1]
        foreign = rejected_labels[(rejected_labels < 0) | (rejected_labels >= n_classes)]
        if foreign.numel():
            values = sorted({int(v) for v in torch.unique(foreign).tolist()})
            if len(values) == 1:
                diagnosis = (
                    f"every such label is {values[0]}, so that is presumably the value "
                    f"loss_fn ignores: pass ignore_index={values[0]} to Analyzer(...) / "
                    "analyze(...)"
                )
            else:
                diagnosis = (
                    f"they are {values[:8]}; the labels must use one ignore value, and it "
                    "must be passed as ignore_index to Analyzer(...) / analyze(...)"
                )
            raise ValueError(
                "valid_mask_fn rejects positions whose label is neither a class index "
                f"(the model has {n_classes} classes) nor the analyzer's "
                f"ignore_index={ignore_index}: {diagnosis}. The analyzer's ignore_index must "
                "be the value loss_fn ignores. vatis would otherwise write "
                f"ignore_index={ignore_index} over those labels before calling loss_fn, and a "
                "loss that ignores a different value fails on it (IndexError on the CPU, an "
                "uncatchable device-side assert on CUDA)."
            )
        try:
            loss = loss_fn(logits, folded)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            hint = ""
            if rejected_labels.numel() and bool((rejected_labels == rejected_labels[0]).all()):
                constant = int(rejected_labels[0].item())
                hint = (
                    f" Every label the mask rejects is {constant}; if that is the value "
                    f"loss_fn ignores, pass ignore_index={constant}."
                )
            raise RuntimeError(
                f"loss_fn raised {type(exc).__name__} on a batch whose labels vatis had "
                "rewritten: valid_mask_fn rejects positions whose label is valid, so vatis "
                f"wrote its ignore_index={ignore_index} into the labels at those positions "
                "before calling loss_fn. A loss_fn that ignores a different value cannot "
                "handle that. The analyzer's ignore_index must be the value loss_fn ignores: "
                f"pass that value as ignore_index to Analyzer(...) / analyze(...).{hint} (If "
                "the failure has another cause, it is the chained exception.)"
            ) from exc

        grad_logits = None
        if loss.requires_grad:
            (grad_logits,) = torch.autograd.grad(loss, logits, retain_graph=True, allow_unused=True)
        if grad_logits is None:
            # The loss is not a function of the logits it was handed (a loss_fn
            # that runs its own forward, or returns a detached value), so there
            # is nothing to inspect. Not being able to verify is not the same as
            # verified: refuse.
            raise ValueError(
                "valid_mask_fn rejects positions whose label is valid, so vatis "
                f"wrote ignore_index={ignore_index} into the labels at those "
                "positions before calling loss_fn -- but the loss returned does not "
                "depend on the logits vatis passed in (no gradient reaches them), so "
                "vatis cannot verify that loss_fn honours the mask. loss_fn must "
                "compute the loss from its `logits` argument. Refusing to continue: "
                "an unverified loss could be the unrestricted one, and delta_loss and "
                "chi_pos would then describe a loss over positions that n_valid, "
                "chi_loss and chi_net exclude."
            )
        n_leaking = int(torch.count_nonzero(grad_logits[~mask]).item())
        if n_leaking:
            raise ValueError(
                "valid_mask_fn rejects positions whose label is valid, so vatis "
                f"wrote ignore_index={ignore_index} into the labels at those "
                "positions before calling loss_fn -- but the loss still depends on "
                f"the logits there ({n_leaking} non-zero gradient entries at "
                "rejected positions). loss_fn must read its targets from the "
                "batch field vatis masks ('labels', else 'targets', for a dict "
                "batch; element 1 of a tuple/list batch) and must ignore "
                f"ignore_index={ignore_index}. Refusing to continue: delta_loss and "
                "chi_pos would describe a loss over positions that n_valid, "
                "chi_loss and chi_net exclude."
            )
        return loss

    return fn


def _fold_mask_into_labels(batch: Any, mask: torch.Tensor, ignore_index: int) -> Any:
    """Return ``batch`` with ``ignore_index`` at every label ``mask`` rejects.

    ``mask`` is the effective mask of :func:`_intersected_valid_mask_fn`. The
    labels are the tensor :func:`_extract_targets` reads, and the result has the
    same form as the input: ``labels`` (else ``targets``) of a dict batch,
    element 1 of a tuple/list batch. A dict batch that carries only
    ``input_ids`` gains a ``labels`` entry holding its (masked) next-token
    targets. Any other batch form has no labels to write into and is refused by
    ``_extract_targets`` — never passed on unmasked.

    If the mask rejects no position whose label is valid, ``batch`` itself is
    returned (the same object, no copy): the labels already say everything the
    mask says. Nothing is modified in place either way.
    """
    targets = _extract_targets(batch)
    keep = mask.to(dtype=torch.bool)
    if not bool(((targets != ignore_index) & ~keep).any()):
        return batch
    folded = targets.masked_fill(~keep, ignore_index)
    if isinstance(batch, dict):
        key = "labels"
        if not isinstance(batch.get("labels"), torch.Tensor) and isinstance(
            batch.get("targets"), torch.Tensor
        ):
            key = "targets"
        return {**batch, key: folded}
    # ``_extract_targets`` accepted it, so this is a tuple/list whose element 1
    # is the target tensor.
    return type(batch)(folded if i == 1 else item for i, item in enumerate(batch))


def _extract_targets(batch: Any) -> torch.Tensor:
    """Extract the labels tensor for the chi_loss closed form."""
    if isinstance(batch, dict):
        if "labels" in batch and isinstance(batch["labels"], torch.Tensor):
            return batch["labels"]
        if "targets" in batch and isinstance(batch["targets"], torch.Tensor):
            return batch["targets"]
        # If neither is present, try to derive from input_ids using the
        # standard shift convention.
        if "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
            from vatis.data.collate import shifted_lm_targets

            return shifted_lm_targets(batch["input_ids"])
        raise ValueError("dict batch missing 'labels'/'targets'/'input_ids'")
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        cand = batch[1]
        if isinstance(cand, torch.Tensor):
            return cand
    raise ValueError(f"cannot extract targets from batch of type {type(batch).__name__}")


def _per_call_seed(base_seed: int, ckpt_id: str, batch_name: str) -> int:
    """Mix base seed with checkpoint id + batch name → 32-bit positive seed.

    We deliberately do NOT use Python's ``hash()`` here — it's randomized
    per process via ``PYTHONHASHSEED``, which would make the same nominal
    ``seed`` produce different probe sequences across runs. SHA-256 is
    overkill for the task but keeps the result deterministic everywhere.
    """
    import hashlib

    payload = f"{base_seed}|{ckpt_id}|{batch_name}".encode()
    digest = hashlib.sha256(payload).digest()
    # Map first 4 bytes to a 32-bit positive int (torch.Generator accepts <2^32).
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _rank_probe_seed(call_seed: int, rank: int) -> int:
    """Seed of the probe stream DDP rank ``rank`` draws from, for one call.

    ``call_seed`` is :func:`_per_call_seed`'s value, i.e. the ``hutchinson_seed``
    column. **Rank 0 gets it unchanged**, so a single-process run — which is
    rank 0 of a world of one — draws exactly the stream that column names, and
    separating the ranks' streams did not change any single-process result.
    Every other rank gets its own seed, ``sha256(f"{call_seed}|rank={rank}")``
    reduced modulo ``2**31 - 1`` like the per-call seed. The world size does not
    enter: it already determines which samples a rank holds.

    Distinct ranks collide only by a hash accident (~``R^2 / 2^32`` for
    ``R`` ranks); a collision would cost that one call the variance the separate
    streams buy, never correctness — the estimate is unbiased either way.
    """
    if rank < 0:
        raise ValueError(f"rank must be >= 0, got {rank}")
    if rank == 0:
        return call_seed
    import hashlib

    digest = hashlib.sha256(f"{call_seed}|rank={rank}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _label(spec: ModelSpec, revision: str | None) -> str:
    if isinstance(spec, str):
        return spec if revision is None else f"{spec}@{revision}"
    return getattr(spec, "identifier", repr(spec))


# ------------------------------------------------------------- one-shot API


def analyze(
    *,
    model: ModelSpec,
    revisions: Sequence[str] | None = None,
    eval_batches: dict[str, Any],
    observables: Sequence[str] | None = None,
    preconditioner: PreconditionerLike | None = None,
    update_response: UpdateResponseFactory | None = None,
    chi_net_method: str | None = None,
    n_hutchinson: int = 32,
    hutchinson_distribution: str = "rademacher",
    micro_batch_size: int = 1,
    cross_pairs: list[tuple[str, str]] | None = None,
    cross_grad_storage: str = "auto",
    sink: ResultSink | str | list[ResultSink] | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: str = "bf16",
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    num_gpus: int = 1,  # accepted for parity; the analyzer ignores it inside the worker
    distributed: str = "ddp",  # accepted for parity
) -> list[CheckpointResult]:
    """One-shot wrapper around :class:`Analyzer`.

    For ``num_gpus > 1`` you should call ``python -m vatis run ...`` instead
    so the CLI can launch torchrun. Calling this function directly with
    ``num_gpus > 1`` will run on a single process; the DDP code paths only
    activate when the process group is already initialized (i.e. when
    launched under torchrun).

    Arguments are those of :class:`Analyzer`; in particular ``ignore_index`` must
    be the value ``loss_fn`` ignores.
    """
    del num_gpus, distributed  # informational only at this layer

    analyzer_obj = Analyzer(
        eval_batches=eval_batches,
        observables=observables,
        preconditioner=preconditioner,
        update_response=update_response,
        chi_net_method=chi_net_method,
        n_hutchinson=n_hutchinson,
        hutchinson_distribution=hutchinson_distribution,
        micro_batch_size=micro_batch_size,
        cross_pairs=cross_pairs,
        cross_grad_storage=cross_grad_storage,
        sink=sink,
        seed=seed,
        device=device,
        dtype=dtype,
        ignore_index=ignore_index,
    )

    revs: list[str | None]
    if isinstance(model, ModelBundle):
        # Pre-built bundle: revisions are just labels.
        revs = list(revisions) if revisions is not None else [""]
    else:
        revs = list(revisions) if revisions is not None else [None]
    specs: list[tuple[ModelSpec, str | None]] = [(model, r) for r in revs]

    return analyzer_obj.run(specs)


# Re-export for callers that ``from vatis import Callable as ...``
del Callable
