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

Preconditioned ruler
--------------------

When a ``preconditioner`` is configured the analyzer measures the **second**
ruler in the same pass: ``delta_loss_P``, ``chi_net_P`` and ``chi_pos_P``, the
LNP triple of the effective kernel ``Theta_P = J P J^T`` the optimizer actually
trains on (``00_context.md`` §"Preconditioned ruler"). It costs no extra
backward — ``delta_loss_P`` re-reduces the gradient vector step 3 already built
with the elementwise ``P`` weight, and ``chi_net_P`` comes out of the same
estimator call and the same probes as ``chi_net``. ``chi_loss`` is
kernel-independent and is not recomputed. Raw numerics are unchanged, bit for
bit: the P accumulators are additive, never in the raw path.

The rows land in the existing long format as new ``observable`` values, plus one
``preconditioner`` descriptor column that carries ``"none"`` on every raw row.
"""

from __future__ import annotations

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
from vatis.models.hf import ModelBundle, ValidMaskFn, load_hf_model
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

# Observables of the preconditioned ruler. Emitted only when a ``preconditioner``
# is configured, and always tagged with that preconditioner's descriptor in the
# ``preconditioner`` column.
#
# There is deliberately no ``chi_net_P_normalized``: the spec (03 §3) names three
# P observables and asks for a *minimal* schema extension. The sigma-normalized
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

# What a ``preconditioner`` argument may be:
#   - None / "none"                  -> raw ruler only (the default, and the
#                                       behaviour of every pre-WP0 caller)
#   - "identity"                     -> P = I, the SGD ruler measured through the
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
        preconditioner: the diagonal ``P`` of the preconditioned ruler, or
            ``None`` (default) for the raw ruler only. See
            :data:`PreconditionerLike`. For a checkpoint bundle use
            ``LocalBundleSource.adamw_preconditioner()``, which reads each
            checkpoint's saved (pre-step) optimizer state.
        chi_net_method: ``"hutchinson"``, ``"per_sequence_cv"``, ``"opacus"``,
            or ``None`` for auto.
        n_hutchinson: number of probe vectors per micro-batch.
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
            this with the checkpoint id and batch name to get a per-call
            generator seed that is shared across DDP ranks.
        device: model device for single-GPU mode. In DDP mode this is
            overridden by ``cuda:LOCAL_RANK``.
        dtype: forward dtype for HF models loaded by name. Ignored when the
            user passes a fully built ModelBundle.
        ignore_index: ignore-index for the chi_loss closed form. Defaults to
            HF's -100.
    """

    def __init__(
        self,
        *,
        eval_batches: dict[str, Any],
        observables: Sequence[str] | None = None,
        preconditioner: PreconditionerLike | None = None,
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
        valid_observables = ALL_OBSERVABLES + P_OBSERVABLES
        if observables is None:
            observables = ALL_OBSERVABLES + (
                P_OBSERVABLES if self.preconditioner is not None else ()
            )
        unknown = set(observables) - set(valid_observables)
        if unknown:
            raise ValueError(f"unknown observables {sorted(unknown)}; valid: {valid_observables}")
        requested_p = set(observables) & set(P_OBSERVABLES)
        if requested_p and self.preconditioner is None:
            raise ValueError(
                f"observables {sorted(requested_p)} belong to the preconditioned ruler "
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
        # Same, for the preconditioned ruler's chi_net^P (the cross pair takes
        # the geometric mean of the two batches' values, exactly as for raw).
        self._chi_net_p_cache: dict[str, float] = {}

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

    def _resolve_cross_grad_storage(self, bundle: ModelBundle) -> str:
        """Decide where cached cross-pair gradients live: ``"gpu"`` or ``"cpu"``.

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
        device = self.device
        if device.type != "cuda" or not torch.cuda.is_available():
            return "gpu"  # no host-offload target that would help
        cross_names = {n for pair in self.cross_pairs for n in pair if pair[0] != pair[1]}
        n_params = sum(p.numel() for p in bundle.params)
        weights_bytes = sum(p.numel() * p.element_size() for p in bundle.params)
        total = torch.cuda.get_device_properties(device).total_memory
        offload = _should_offload_cross_grads(
            n_params=n_params,
            weights_bytes=weights_bytes,
            n_cross_batches=len(cross_names),
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
        precond = self._resolve_preconditioner(bundle, ckpt_id)

        # Self pairs first. We cache a batch's full parameter-space gradient
        # ONLY if some cross pair will consume it via the dot product. Caching it
        # unconditionally keeps a P-sized vector resident (e.g. ~29 GB fp32 for a
        # 7B model) while subsequent batches run their own backward — for large
        # models that retained gradient is enough to OOM even when no cross pair
        # was requested. So we cache selectively and drop the local reference
        # otherwise, freeing it before the next batch's backward.
        cross_names = {n for pair in self.cross_pairs for n in pair if pair[0] != pair[1]}
        grad_storage = self._resolve_cross_grad_storage(bundle) if cross_names else "gpu"
        per_batch_grad_cache: dict[str, torch.Tensor] = {}
        per_batch_n_valid: dict[str, int] = {}

        for name, spec in self.eval_batches.items():
            self_result, g_full = self._compute_self_pair(
                bundle, ckpt_id, revision, name, spec, precond
            )
            result.rows.extend(self_result)
            per_batch_n_valid[name] = self._last_n_valid
            if g_full is not None and name in cross_names:
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

        Returns ``None`` when the raw ruler alone should be measured. The
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
    ) -> tuple[list[ResultRow], torch.Tensor | None]:
        """Compute chi_loss / delta_loss / chi_net / chi_pos for one batch.

        With ``precond`` given, also the preconditioned triple
        (``delta_loss_P``, ``chi_net_P``, ``chi_pos_P``) — no extra backward:
        ``delta_loss_P`` re-reduces the same gradient vector with the ``P``
        weight and ``chi_net_P`` comes from the same estimator call.

        Returns the rows AND the flat parameter-space gradient of the total
        loss (used as a cache for cross pairs).
        """
        t0 = time.perf_counter()
        device = next(bundle.model.parameters()).device

        raw = spec.get_batch(ckpt_id)
        # Shard across DDP ranks; if single-GPU shard_batch is a no-op slice.
        local_batch, local_b, _global_b = shard_batch(
            raw, rank=get_rank(), world_size=get_world_size()
        )
        local_batch = move_batch(local_batch, device)

        # THE mask, built once and used by every accumulator below: the bundle's
        # ``valid_mask_fn`` intersected with ``labels != ignore_index``. See
        # ``_intersected_valid_mask_fn`` for why all three paths need it.
        valid_mask_fn = _intersected_valid_mask_fn(bundle.valid_mask_fn, self.ignore_index)

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
        n_total_for_loss = n_valid_local
        if n_total_for_loss == 0:
            # No valid tokens on this rank. Still emit zeros and let the
            # all-reduce handle the global counts.
            delta_loss_local = torch.zeros((), dtype=torch.float64, device=device)
        else:
            for _start, _stop, micro in iter_micro_batches(local_batch, self.micro_batch_size):
                logits = bundle.forward_fn(bundle.model, micro)
                # Compute the micro-batch's contribution to L_local using the
                # SAME loss_fn the user provided. We then re-scale by the ratio
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
                loss_micro = bundle.loss_fn(logits, micro)
                weight = float(n_valid_micro) / float(n_total_for_loss)
                # gradient of (weight * loss_micro) w.r.t. theta
                grads = torch.autograd.grad(
                    weight * loss_micro,
                    bundle.params,
                    retain_graph=False,
                    allow_unused=True,
                )
                _add_grads_into_flat(flat_grad_local, grads, bundle.params)
                del logits
            delta_loss_local = _sum_sq_fp64(flat_grad_local)

        # ----- chi_net (per-rank partial) -----
        method = select_chi_net_method(b_total=_global_b, requested=self.chi_net_method_request)
        estimator = build_estimator(
            method,
            n_hutchinson=self.n_hutchinson,
            distribution=self.hutchinson_distribution,
        )
        # Probe seed must be SHARED across ranks for the same (ckpt, batch).
        # The generator is deliberately on the CPU even for a CUDA run: probes are
        # drawn on the CPU and copied to the device
        # (``vatis.core.probes.make_probe``), so the ``hutchinson_seed`` recorded
        # in the sink identifies the same probe set on every device. A CUDA
        # generator would make one seed mean different probes on CPU vs GPU, and
        # the two runs could then only agree to within the Hutchinson noise floor.
        probe_seed = _per_call_seed(self.seed, ckpt_id, batch_name)
        gen = torch.Generator()
        gen.manual_seed(probe_seed)
        chi_net_result: ChiNetResult = estimator.compute(
            bundle.model,
            local_batch,
            bundle.forward_fn,
            loss_fn=bundle.loss_fn,
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
                "returned no chi_net_p; the preconditioned ruler is implemented in "
                "'hutchinson' and 'per_sequence_cv' only"
            )
        else:
            chi_net_p_local = chi_net_result.chi_net_p.to(dtype=torch.float64).to(device)

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
            delta_loss_global = float(_sum_sq_fp64(flat_grad_local))
        else:
            chi_loss_raw_global = float(chi_loss_raw_local)
            chi_net_global = float(chi_net_local)
            chi_net_p_global = None if chi_net_p_local is None else float(chi_net_p_local)
            n_valid_global = n_valid_local
            delta_loss_global = float(delta_loss_local)

        # delta_loss^P = grad L^T P grad L: the same (all-reduced) gradient
        # vector, reduced again with the elementwise P weight. No backward.
        delta_loss_p_global = (
            None if precond is None else float(weighted_sum_sq_fp64(flat_grad_local, precond.diag))
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
        )

        # Cache the (already-reduced) full gradient for cross-pair use.
        # On non-distributed runs flat_grad_local is the full grad (unscaled).
        # On distributed runs we already wrote the reweighted+reduced version.
        self._last_n_valid = n_valid_a
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
        chi_pos_value = float(
            chi_pos(delta_loss_cross_value, chi_loss_cross_value, chi_net_cross_value)
        )

        # Preconditioned ruler, same shape: the P-weighted gradient dot and the
        # geometric mean of the two batches' chi_net^P (i.e.
        # ||P^{1/2} J^A||_F * ||P^{1/2} J^B||_F, per 00_context.md).
        delta_loss_p_cross: float | None = None
        chi_net_p_cross: float | None = None
        chi_pos_p_value: float | None = None
        if precond is not None:
            delta_loss_p_cross = float(weighted_dot_fp64(g_a, g_b, precond.diag))
            chi_net_p_cross = (self._chi_net_p_cache[name_a] * self._chi_net_p_cache[name_b]) ** 0.5
            chi_pos_p_value = float(
                chi_pos(delta_loss_p_cross, chi_loss_cross_value, chi_net_p_cross)
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
        )

    # ------------------------------------------------------------- emission

    _last_n_valid: int = 0

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
    ) -> list[ResultRow]:
        # Stash the per-batch chi_loss / chi_net for the cross pair use.
        # Caches are eagerly initialized in __init__.
        if batch_a == batch_b:
            self._chi_loss_cache[batch_a] = chi_loss
            self._chi_net_cache[batch_a] = chi_net
            if chi_net_p is not None:
                self._chi_net_p_cache[batch_a] = chi_net_p

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
                    "chi_loss normalizes by (a sum-reduced CE inflates chi_pos by exactly "
                    "N_valid^2), or loss_fn's ignore_index differs from the analyzer's. "
                    "A very small n_hutchinson can also overshoot legitimately.",
                    UserWarning,
                    stacklevel=2,
                )

        rows: list[ResultRow] = []
        for name in self.observables:
            if name not in values:
                # A P observable was requested but this checkpoint has no P (the
                # factory declined — see the step-0 policy). Skipping the row is
                # deliberate: emitting a raw-valued or 1/eps-valued row under the
                # 'adamw' descriptor would be a wrong answer that looks plausible.
                continue
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
            "for the raw ruler, 'identity' for P = I, a Preconditioner instance, or a "
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
    num_gpus: int = 1,  # accepted for parity; the analyzer ignores it inside the worker
    distributed: str = "ddp",  # accepted for parity
) -> list[CheckpointResult]:
    """One-shot wrapper around :class:`Analyzer`.

    For ``num_gpus > 1`` you should call ``python -m vatis run ...`` instead
    so the CLI can launch torchrun. Calling this function directly with
    ``num_gpus > 1`` will run on a single process; the DDP code paths only
    activate when the process group is already initialized (i.e. when
    launched under torchrun).
    """
    del num_gpus, distributed  # informational only at this layer

    analyzer_obj = Analyzer(
        eval_batches=eval_batches,
        observables=observables,
        preconditioner=preconditioner,
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
