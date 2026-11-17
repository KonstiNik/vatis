"""Timing, self-checks and reporting of the Llama example.

Support code for ``run.py``, moved out of it so that the script shows the vatis
calls. Three groups:

- **timing**: ``timed_model_specs`` wraps ``source.model_specs()`` and records
  the load and analysis time and the peak GPU memory of every checkpoint;
  ``performance_table`` and ``write_csv`` report them.
- **self-checks**: ``identity_recheck`` (``P = I`` must reproduce the raw
  values) and ``epsilon_ceiling_diagnostic`` (how much of the AdamW ``P`` sits
  at the ``1/eps`` ceiling, and how much of ``chi_net^P`` those coordinates
  carry). Both run the analyzer again on a few checkpoints.
- **reporting**: the printed header, the printed ceiling records and the
  ``run_summary.json`` record.
"""

from __future__ import annotations

import argparse
import platform
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from llama_bundle import BATCH_NAMES

from vatis.analyzer import Analyzer
from vatis.core.preconditioner import (
    RAW_DESCRIPTOR,
    Preconditioner,
    epsilon_cap_mask,
    resolve_adamw_hparams,
)
from vatis.models.local_bundle import (
    LocalBundleSource,
    adamw_preconditioner_from_state,
)
from vatis.sinks.base import ResultRow

RAW_TO_P = {
    "delta_loss": "delta_loss_P",
    "chi_net": "chi_net_P",
    "chi_pos": "chi_pos_P",
}


# --------------------------------------------------------------- timing plumbing


@dataclass
class CheckpointTiming:
    """One row of the performance table."""

    checkpoint_id: str
    step: int
    load_s: float
    analysis_s: float
    peak_gpu_mb: float | None


def timed_model_specs(
    source: LocalBundleSource,
    timings: list[CheckpointTiming],
    *,
    device: torch.device,
) -> Iterator[tuple[Any, str]]:
    """``source.model_specs()`` with per-checkpoint wall time recorded.

    A generator's body resumes only when the consumer asks for the next item, so
    the code after ``yield`` runs exactly when the analyzer has finished that
    checkpoint: ``t_after - t_before`` is that checkpoint's analysis time, and
    the bundle build/load time is measured separately. Peak GPU memory is read
    per checkpoint from the torch allocator (this process only — a co-tenant
    process on the same card is invisible to it, as it should be).
    """
    cuda = device.type == "cuda"
    for checkpoint in source:
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        bundle = checkpoint.model_bundle
        t1 = time.perf_counter()
        yield bundle, checkpoint.revision
        t2 = time.perf_counter()
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None
        timings.append(
            CheckpointTiming(
                checkpoint_id=checkpoint.identifier,
                step=checkpoint.step,
                load_s=t1 - t0,
                analysis_s=t2 - t1,
                peak_gpu_mb=peak,
            )
        )


# ------------------------------------------------------------------ self-checks


def all_rows(results: list[Any]) -> list[ResultRow]:
    return [row for result in results for row in result.rows]


def identity_recheck(
    source: LocalBundleSource,
    analyzer: Analyzer,
    step: int,
) -> dict[str, Any]:
    """P = I must reproduce the raw columns exactly. Acceptance criterion 5.

    ``analyzer`` is the sweep's configuration with ``preconditioner="identity"``
    and no sink. Compared *within one run*: the raw rows of a preconditioned
    pass are bit-identical to a raw-only pass, so the honest check is
    ``chi_pos_P == chi_pos`` from the same probes, same checkpoint, same batch.
    """
    checkpoint = source.checkpoint_for(step)
    rows = all_rows(analyzer.run([(checkpoint.model_bundle, checkpoint.revision)]))

    def key(row: ResultRow) -> tuple[str, str, str]:
        return (row.checkpoint_id, row.batch_a, row.batch_b)

    raw = {(key(r), r.observable): r.value for r in rows if r.preconditioner == RAW_DESCRIPTOR}
    preconditioned = {
        (key(r), r.observable): r.value for r in rows if r.observable in RAW_TO_P.values()
    }
    worst, worst_at, compared = 0.0, None, 0
    for (row_key, raw_name), raw_value in raw.items():
        p_name = RAW_TO_P.get(raw_name)
        if p_name is None or (row_key, p_name) not in preconditioned:
            continue
        compared += 1
        deviation = abs(preconditioned[(row_key, p_name)] - raw_value)
        if deviation > worst:
            worst, worst_at = deviation, f"{row_key} {raw_name}"
    checkpoint.release()
    return {
        "step": step,
        "checkpoint_id": checkpoint.identifier,
        "pairs_compared": compared,
        "worst_abs_deviation": worst,
        "worst_at": worst_at,
        "tolerance": 1e-6,
        "passed": compared > 0 and worst <= 1e-6,
    }


def ceiling_split_steps(steps: list[int], n_requested: int) -> list[int]:
    """The checkpoints that get the ``chi_net^P`` ceiling split: evenly spread over
    the checkpoints that have optimizer state (all but step 0)."""
    with_state = [s for s in steps if s != 0]
    n_eps = max(0, min(n_requested, len(with_state)))
    if n_eps:
        picks = np.unique(np.linspace(0, len(with_state) - 1, n_eps).round().astype(int))
        return [with_state[i] for i in picks]
    return []


def epsilon_ceiling_diagnostic(
    source: LocalBundleSource,
    args: argparse.Namespace,
    eval_batches: dict[str, dict[str, torch.Tensor]],
    chi_net_steps: list[int],
) -> list[dict[str, Any]]:
    """Share of the AdamW P diagonal at ``1/eps``, and of ``chi_net^P`` it carries.

    The diagonal share is cheap (a mask over the saved ``exp_avg_sq``) and is
    reported for every checkpoint that has optimizer state. The ``chi_net^P``
    share needs the estimator: ``Tr(J P J^T)`` is linear in ``P`` at fixed
    probes, so running the same checkpoint/batch/seed twice — once with the real
    ``P``, once with the ceiling entries zeroed — splits the trace exactly into a
    live and a dead-direction part. That costs two extra estimator passes per
    checkpoint, hence only a few checkpoints get it.

    Nothing here changes the preconditioner: ``P`` is the frozen convention and
    is used as-is for every published row.
    """
    out: list[dict[str, Any]] = []
    layout = source.layout
    vocab_size = int(source.metadata["model_config"]["vocab_size"])
    tokens_in_b = {int(t) for t in eval_batches["B"]["input_ids"].unique().tolist()}
    for checkpoint in source.checkpoints:
        state = checkpoint.optimizer_state
        if state.is_empty:
            out.append(
                {
                    "step": checkpoint.step,
                    "optimizer_state": "empty",
                    "note": "step-0 checkpoint: no AdamW P exists, P rows skipped",
                }
            )
            checkpoint.release()
            continue
        _beta2, eps = resolve_adamw_hparams(state.hparams, source="saved param_groups")
        precond = adamw_preconditioner_from_state(state, metadata=source.metadata)
        capped = epsilon_cap_mask(precond.diag, eps=eps)
        n_capped, n_total = int(capped.sum()), int(precond.diag.numel())
        record: dict[str, Any] = {
            "step": checkpoint.step,
            "eps": eps,
            "n_capped": n_capped,
            "n_total": n_total,
            "diag_capped_share": n_capped / n_total,
            "p_min": float(precond.diag.min()),
            "p_max": float(precond.diag.max()),
        }
        if n_capped:
            # Which tensors the dead coordinates live in — "unused vocabulary
            # rows" is the expectation, and the layout can confirm or refute it.
            per_param = sorted(
                (
                    (name, int(capped[layout.slice_of(name)].sum()), layout.numel_of(name))
                    for name, _shape in layout
                ),
                key=lambda item: -item[1],
            )
            record["capped_by_param"] = [
                {"name": name, "n_capped": n, "numel": numel}
                for name, n, numel in per_param[:5]
                if n
            ]
            record["dead_vocab_rows"] = dead_vocab_rows(
                capped, layout=layout, vocab_size=vocab_size, tokens_in_b=tokens_in_b
            )
        if checkpoint.step in chi_net_steps and n_capped:
            full = _chi_net_p(args, eval_batches, checkpoint, precond.diag.clone())
            live_diag = precond.diag.clone()
            live_diag[capped] = 0.0
            live = _chi_net_p(args, eval_batches, checkpoint, live_diag)
            record["chi_net_p_full"] = full
            record["chi_net_p_live_only"] = live
            record["chi_net_p_ceiling_share"] = (full - live) / full if full else float("nan")
        out.append(record)
        checkpoint.release()
    return out


def dead_vocab_rows(
    capped: torch.Tensor,
    *,
    layout: Any,
    vocab_size: int,
    tokens_in_b: set[int],
) -> list[dict[str, Any]]:
    """Ceiling coordinates attributed to whole, never-updated vocabulary rows.

    A vocabulary-shaped parameter (first dimension ``vocab_size``) has one row per
    token, and an embedding row only receives gradient when its token appears in
    a training batch — so an unused token's row is capped in full. Those rows are
    harmless for ``chi_net^P`` only if the same token is also absent from the
    *measurement* batch, because then its Jacobian column is identically zero.
    The overlap count is therefore the mechanism behind the ``chi_net^P`` ceiling
    share, and is what makes an exact ``0.00%`` an explanation rather than a
    rounding artifact.
    """
    out: list[dict[str, Any]] = []
    for name, shape in layout:
        if len(shape) != 2 or int(shape[0]) != vocab_size:
            continue
        rows = capped[layout.slice_of(name)].reshape(tuple(int(s) for s in shape)).all(dim=1)
        dead = {int(i) for i in torch.nonzero(rows).flatten().tolist()}
        out.append(
            {
                "name": name,
                "n_dead_rows": len(dead),
                "vocab_size": vocab_size,
                "n_dead_rows_present_in_batch_b": len(dead & tokens_in_b),
                "n_distinct_tokens_in_batch_b": len(tokens_in_b),
            }
        )
    return out


def _chi_net_p(
    args: argparse.Namespace,
    eval_batches: dict[str, dict[str, torch.Tensor]],
    checkpoint: Any,
    diag: torch.Tensor,
) -> float:
    """``chi_net^P`` on the eval batch B for an explicitly supplied P diagonal."""
    analyzer = Analyzer(
        eval_batches={"B": eval_batches["B"]},
        preconditioner=Preconditioner(diag=diag, descriptor="explicit"),
        chi_net_method=args.chi_net_method,
        n_hutchinson=args.n_hutchinson,
        micro_batch_size=args.micro_batch_size,
        sink=None,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )
    for row in all_rows(analyzer.run([(checkpoint.model_bundle, checkpoint.revision)])):
        if row.observable == "chi_net_P":
            return float(row.value)
    raise AssertionError("no chi_net_P row produced")


# ------------------------------------------------------------------- reporting


def print_run_header(args: argparse.Namespace, source: LocalBundleSource, steps: list[int]) -> None:
    """What is about to be measured: the bundle, the model, the estimator settings."""
    metadata = source.metadata
    print(f"  bundle          : {args.bundle}")
    print(f"  run             : {source.run_name}  ({metadata.get('written_by')})")
    print(f"  model           : {metadata.get('model_class')} {metadata['model_config']}")
    print(
        f"  parameters      : {source.layout.total_numel:,} in {len(source.layout)} tensors "
        f"(metadata says {metadata.get('num_parameters')})"
    )
    print(
        f"  optimizer       : {metadata.get('optimizer_class')} "
        f"beta2={metadata['optimizer_hparams']['beta2']} eps={metadata['optimizer_hparams']['eps']}"
    )
    print(
        f"  data            : {metadata.get('dataset_id')} B_train={metadata.get('B_train')} "
        f"seq_len={metadata.get('seq_len')} total_steps={metadata.get('total_steps_planned')}"
    )
    print(f"  checkpoints     : {len(steps)} {steps}")
    print(f"  device / dtype  : {args.device} / {args.dtype}")
    print(
        f"  estimator       : {args.chi_net_method or 'hutchinson (vatis default)'} n_hutchinson={args.n_hutchinson} "
        f"micro_batch_size={args.micro_batch_size} seed={args.seed}"
    )


def print_epsilon_ceiling(eps_records: list[dict[str, Any]]) -> None:
    """One line per checkpoint of ``epsilon_ceiling_diagnostic``."""
    for record in eps_records:
        if record.get("optimizer_state") == "empty":
            print(f"    step {record['step']:>4}: {record['note']}")
            continue
        share = f"{record['diag_capped_share']:6.2%}"
        extra = ""
        if "chi_net_p_ceiling_share" in record:
            extra = (
                f"   chi_net^P from ceiling: {record['chi_net_p_ceiling_share']:6.2%} "
                f"(full {record['chi_net_p_full']:.4e})"
            )
            for vocab in record.get("dead_vocab_rows", []):
                extra += (
                    f"\n              dead {vocab['name']} rows: {vocab['n_dead_rows']}/"
                    f"{vocab['vocab_size']}, of which "
                    f"{vocab['n_dead_rows_present_in_batch_b']} also occur in batch B "
                    f"(only those can contribute)"
                )
        print(
            f"    step {record['step']:>4}: P at 1/eps {record['n_capped']:>8}/"
            f"{record['n_total']} ({share})   P range "
            f"[{record['p_min']:.3e}, {record['p_max']:.3e}]{extra}"
        )


def performance_table(
    timings: list[CheckpointTiming],
    *,
    total_s: float,
    script_s: float,
    peak_gpu_mb: float | None,
    n_rows: int,
) -> str:
    """The performance table as text. Spec 05 §3: a deliverable, not a nicety.

    ``total_s`` is the sweep — every checkpoint loaded, raw and preconditioned
    observables, rows into the sink — and is the number to budget a grid run
    with. ``script_s`` adds this example's extra diagnostics (the ``P = I``
    re-check and the epsilon-ceiling split, several extra estimator passes)
    which a production sweep would not pay.
    """
    # peak_gpu_mb is the allocator's peak since the last reset, i.e. of the last
    # checkpoint (timed_model_specs resets it per checkpoint); the table reports
    # the per-checkpoint peaks, their mean and their largest value instead.
    peaks = [t.peak_gpu_mb for t in timings if t.peak_gpu_mb is not None]
    peak_mean = f"{sum(peaks) / len(peaks):.0f}" if peaks else "-"
    peak_max = f"{max(peaks):.0f}" if peaks else "-"
    lines = [
        f"{'checkpoint':>16} {'step':>6} {'load_s':>8} {'analysis_s':>11} {'peak_gpu_mb':>12}",
    ]
    for row in timings:
        peak = "-" if row.peak_gpu_mb is None else f"{row.peak_gpu_mb:.0f}"
        lines.append(
            f"{row.checkpoint_id:>16} {row.step:>6} {row.load_s:>8.2f} "
            f"{row.analysis_s:>11.2f} {peak:>12}"
        )
    if timings:
        analysis = [t.analysis_s for t in timings]
        loads = [t.load_s for t in timings]
        lines.append(
            f"{'mean':>16} {'':>6} {sum(loads) / len(loads):>8.2f} "
            f"{sum(analysis) / len(analysis):>11.2f} {peak_mean:>12}"
        )
        lines.append(f"{'total':>16} {'':>6} {sum(loads):>8.2f} {sum(analysis):>11.2f} {'':>12}")
    lines.append("")
    lines.append(f"  checkpoints          : {len(timings)}")
    lines.append(f"  result rows          : {n_rows}")
    lines.append(
        f"  sweep wall time      : {total_s:.1f} s  (load + raw and preconditioned + sink)"
    )
    lines.append(f"  script wall time     : {script_s:.1f} s  (+ P=I re-check, epsilon split)")
    if peak_gpu_mb is None:
        lines.append("  peak GPU memory      : n/a (CPU run)")
        lines.append(
            "  caveat               : wall times are as-measured on whatever this host was "
            "doing at the time, not tuned or best-of-N."
        )
    else:
        lines.append(
            f"  peak GPU memory      : {peak_max} MiB, the largest per-checkpoint peak "
            "(torch allocator, this process)"
        )
        lines.append(
            "  caveat               : the GPU may be shared (the allocator only sees this "
            "process); wall times are as-measured, not tuned or best-of-N."
        )
    return "\n".join(lines)


def write_csv(path: Path, timings: list[CheckpointTiming]) -> None:
    header = "checkpoint_id,step,load_s,analysis_s,peak_gpu_mb\n"
    body = "".join(
        f"{t.checkpoint_id},{t.step},{t.load_s:.4f},{t.analysis_s:.4f},"
        f"{'' if t.peak_gpu_mb is None else f'{t.peak_gpu_mb:.1f}'}\n"
        for t in timings
    )
    path.write_text(header + body)


def run_summary(
    args: argparse.Namespace,
    source: LocalBundleSource,
    *,
    used_random_batches: bool,
    training_fallback: bool,
    identity: dict[str, Any],
    eps_records: list[dict[str, Any]],
    nonfinite: list[Any],
    timings: list[CheckpointTiming],
    total_s: float,
    script_s: float,
    peak_gpu_mb: float | None,
    n_rows: int,
    table: str,
    skip_warnings: list[str],
) -> dict[str, Any]:
    """The ``run_summary.json`` record: what the parquet has no column for."""
    metadata = source.metadata
    ckpt_meta = []
    for checkpoint in source.checkpoints:
        meta = checkpoint.meta
        ckpt_meta.append(
            {
                "checkpoint_id": checkpoint.identifier,
                "revision": checkpoint.revision,
                "step": checkpoint.step,
                "train_loss": meta.train_loss,
                "lr": meta.lr,
                "tokens_seen": meta.tokens_seen,
                "wall_time": meta.wall_time,
            }
        )

    return {
        "kind": "wp0_two_rulers_llama",
        "bundle": str(args.bundle),
        "eval_batches_dir": str(args.eval_batches),
        "random_token_measurement_batches": used_random_batches,
        "random_token_training_run": training_fallback,
        "scientifically_empty": used_random_batches or training_fallback,
        "config": {
            "device": str(args.device),
            "dtype": args.dtype,
            "chi_net_method": args.chi_net_method or "hutchinson",  # the resolved name
            "n_hutchinson": args.n_hutchinson,
            "micro_batch_size": args.micro_batch_size,
            "seed": args.seed,
            "batch_names": list(BATCH_NAMES),
            "cross_pairs": [[n, "B"] for n in BATCH_NAMES if n != "B"],
        },
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else None,
            "gpu_contended": "not checked -- the card may be shared; timings are "
            "as-measured, untuned"
            if args.device.type == "cuda"
            else None,
        },
        "bundle_metadata": {
            "run_name": source.run_name,
            "model_class": metadata.get("model_class"),
            "model_config": metadata["model_config"],
            "num_parameters": metadata.get("num_parameters"),
            "num_param_tensors": metadata.get("num_param_tensors"),
            "optimizer_class": metadata.get("optimizer_class"),
            "optimizer_hparams": dict(metadata["optimizer_hparams"]),
            "dataset_id": metadata.get("dataset_id"),
            "tokenizer": metadata.get("tokenizer"),
            "B_train": metadata.get("B_train"),
            "seq_len": metadata.get("seq_len"),
            "loss_ignore_index": source.loss_ignore_index,
            "total_steps_planned": metadata.get("total_steps_planned"),
            "bundle_steps_spec": metadata.get("bundle_steps_spec"),
            "learning_rate": metadata.get("learning_rate"),
            "warmup_steps": metadata.get("warmup_steps"),
        },
        "checkpoints": ckpt_meta,
        "identity_recheck": identity,
        "epsilon_ceiling": eps_records,
        "non_finite_rows": nonfinite,
        "performance": {
            "per_checkpoint": [asdict(t) for t in timings],
            "total_wall_s": total_s,
            "script_wall_s": script_s,
            "peak_gpu_mb": peak_gpu_mb,
            "n_rows": n_rows,
            "table": table,
        },
        "step0_p_skip_warnings": skip_warnings,
    }
