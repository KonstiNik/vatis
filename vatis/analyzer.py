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
"""

from __future__ import annotations

import time
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
)
from vatis.core.observables import (
    chi_loss_cross_entropy_unnormalized,
    chi_pos,
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
from vatis.models.hf import ModelBundle, load_hf_model
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
        observables: which observables to compute and emit. Defaults to all
            of ``ALL_OBSERVABLES``.
        chi_net_method: ``"hutchinson"``, ``"per_sequence_cv"``, ``"opacus"``,
            or ``None`` for auto.
        n_hutchinson: number of probe vectors per micro-batch.
        hutchinson_distribution: ``"rademacher"`` (default) or ``"gaussian"``.
        micro_batch_size: number of sequences per backward. Default 1.
        cross_pairs: optional list of ``(name_a, name_b)`` tuples to compute
            ``delta_loss(A, B)`` (and the corresponding ``chi_pos``) for.
            Self pairs ``(name, name)`` are always computed.
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
        observables: Sequence[str] = ALL_OBSERVABLES,
        chi_net_method: str | None = None,
        n_hutchinson: int = 32,
        hutchinson_distribution: str = "rademacher",
        micro_batch_size: int = 1,
        cross_pairs: list[tuple[str, str]] | None = None,
        sink: ResultSink | str | list[ResultSink] | None = None,
        seed: int = 0,
        device: torch.device | str = "cpu",
        dtype: str = "bf16",
        ignore_index: int = DEFAULT_IGNORE_INDEX,
    ) -> None:
        self.eval_batches: dict[str, EvalBatchSpec] = normalize_eval_batches(eval_batches)
        if not self.eval_batches:
            raise ValueError("at least one eval batch is required")

        unknown = set(observables) - set(ALL_OBSERVABLES)
        if unknown:
            raise ValueError(f"unknown observables {sorted(unknown)}; valid: {ALL_OBSERVABLES}")
        self.observables = tuple(observables)
        self.chi_net_method_request = chi_net_method
        self.n_hutchinson = n_hutchinson
        self.hutchinson_distribution = hutchinson_distribution
        self.micro_batch_size = micro_batch_size
        self.cross_pairs = list(cross_pairs or [])
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
        result = CheckpointResult(checkpoint_id=ckpt_id, revision=revision)
        bundle.model.eval()
        # Always require_grad for the analyzer's backwards.
        for p in bundle.params:
            p.requires_grad_(True)

        # Self pairs first; we cache the per-batch grads (parameter-vector flat)
        # so the cross pairs can re-use them via dot product.
        per_batch_grad_cache: dict[str, torch.Tensor] = {}
        per_batch_n_valid: dict[str, int] = {}

        for name, spec in self.eval_batches.items():
            self_result, g_full = self._compute_self_pair(bundle, ckpt_id, revision, name, spec)
            result.rows.extend(self_result)
            # The cached gradient lives on the model device; sized P (params).
            if g_full is not None:
                per_batch_grad_cache[name] = g_full
            per_batch_n_valid[name] = self._last_n_valid

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
            )
            result.rows.extend(cross_rows)

        if is_main_rank() and self.sinks:
            for s in self.sinks:
                s.write_rows(result.rows)

        return result

    # ------------------------------------------------------------- self pair

    def _compute_self_pair(
        self,
        bundle: ModelBundle,
        ckpt_id: str,
        revision: str,
        batch_name: str,
        spec: EvalBatchSpec,
    ) -> tuple[list[ResultRow], torch.Tensor | None]:
        """Compute chi_loss / delta_loss / chi_net / chi_pos for one batch.

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
            vmask = bundle.valid_mask_fn(micro, logits)
            raw = chi_loss_cross_entropy_unnormalized(
                logits, _extract_targets(micro), ignore_index=self.ignore_index
            )
            chi_loss_raw_local = chi_loss_raw_local + raw.to(dtype=torch.float64)
            if vmask is None:
                n_valid_local += int(torch.tensor(logits.shape[:-1]).prod().item())
            else:
                n_valid_local += int(vmask.sum().item())
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
                vmask_micro = bundle.valid_mask_fn(micro, logits)
                n_valid_micro = (
                    int(vmask_micro.sum().item())
                    if vmask_micro is not None
                    else int(torch.tensor(logits.shape[:-1]).prod().item())
                )
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
            delta_loss_local = (flat_grad_local.to(dtype=torch.float64) ** 2).sum()

        # ----- chi_net (per-rank partial) -----
        method = select_chi_net_method(b_total=_global_b, requested=self.chi_net_method_request)
        estimator = build_estimator(
            method,
            n_hutchinson=self.n_hutchinson,
            distribution=self.hutchinson_distribution,
        )
        # Probe seed must be SHARED across ranks for the same (ckpt, batch).
        probe_seed = _per_call_seed(self.seed, ckpt_id, batch_name)
        gen = torch.Generator(device=device)
        gen.manual_seed(probe_seed)
        chi_net_result: ChiNetResult = estimator.compute(
            bundle.model,
            local_batch,
            bundle.forward_fn,
            loss_fn=bundle.loss_fn,
            valid_mask_fn=bundle.valid_mask_fn,
            params=bundle.params,
            micro_batch_size=self.micro_batch_size,
            generator=gen,
        )
        chi_net_local = chi_net_result.chi_net.to(dtype=torch.float64).to(device)

        # ----- All-reduce: chi_loss, chi_net, n_valid, delta_loss grad -----
        # delta_loss needs the FULL gradient (sum across ranks) to compute its
        # squared norm correctly: ||sum_r g_r||^2, not sum_r ||g_r||^2.
        # We re-weight the local grad and all-reduce-sum, then square locally.
        if is_distributed():
            import torch.distributed as dist

            chi_loss_raw_global = all_reduce_sum_scalar(float(chi_loss_raw_local), device)
            chi_net_global = all_reduce_sum_scalar(float(chi_net_local), device)
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
            delta_loss_global = float((flat_grad_local.to(torch.float64) ** 2).sum())
        else:
            chi_loss_raw_global = float(chi_loss_raw_local)
            chi_net_global = float(chi_net_local)
            n_valid_global = n_valid_local
            delta_loss_global = float(delta_loss_local)

        wallclock_s = time.perf_counter() - t0
        n_valid_a = max(1, n_valid_global)  # avoid div-by-zero downstream
        # Apply the global N-normalization to the chi_loss raw sum.
        chi_loss_global = chi_loss_raw_global / (float(n_valid_a) ** 2)
        norm = normalization_factor(n_valid_a, n_valid_a)
        chi_loss_norm = chi_loss_global * norm
        chi_net_norm = chi_net_global / norm
        chi_pos_value = float(chi_pos(delta_loss_global, chi_loss_global, chi_net_global))

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
            if name not in self._chi_loss_cache or name not in self._chi_net_cache
        ]
        if missing:
            raise RuntimeError(
                f"_compute_cross_pair called for cross pair ({name_a!r}, {name_b!r}) "
                f"but the self-pair cache is missing {missing}. "
                f"Call _compute_self_pair for both batches first; "
                f"Analyzer.run() does this automatically."
            )

        # Cross delta_loss is just the dot product of the cached flat grads.
        delta_loss_cross_value = float((g_a.to(torch.float64) * g_b.to(torch.float64)).sum())

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
    ) -> list[ResultRow]:
        # Stash the per-batch chi_loss / chi_net for the cross pair use.
        # Caches are eagerly initialized in __init__.
        if batch_a == batch_b:
            self._chi_loss_cache[batch_a] = chi_loss
            self._chi_net_cache[batch_a] = chi_net

        values = {
            "chi_loss": chi_loss,
            "chi_loss_normalized": chi_loss_normalized,
            "chi_net": chi_net,
            "chi_net_normalized": chi_net_normalized,
            "delta_loss": delta_loss,
            "chi_pos": chi_pos_value,
        }
        rows: list[ResultRow] = []
        for name in self.observables:
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
                )
            )
        return rows


# ------------------------------------------------------------- helpers


def _zero_param_vector(params: list[torch.nn.Parameter], device: torch.device) -> torch.Tensor:
    sizes = [p.numel() for p in params]
    if not sizes:
        raise ValueError("model has no parameters")
    return torch.zeros(sum(sizes), dtype=torch.float32, device=device)


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
    observables: Sequence[str] = ALL_OBSERVABLES,
    chi_net_method: str | None = None,
    n_hutchinson: int = 32,
    hutchinson_distribution: str = "rademacher",
    micro_batch_size: int = 1,
    cross_pairs: list[tuple[str, str]] | None = None,
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
        chi_net_method=chi_net_method,
        n_hutchinson=n_hutchinson,
        hutchinson_distribution=hutchinson_distribution,
        micro_batch_size=micro_batch_size,
        cross_pairs=cross_pairs,
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
