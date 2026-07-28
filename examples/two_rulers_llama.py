"""WP0 GPU end-to-end example, ANALYSIS side: both rulers on a real Llama bundle.

Phase 1 of the two-phase pattern (see ``CLAUDE.md`` §Research workflow): this
script is the **expensive compute step**. It reads a checkpoint bundle written by
the training stack, runs the raw and AdamW rulers over every checkpoint on the
GPU, and writes

- ``examples/output/results.parquet``  — the canonical long-format result table
- ``examples/output/run_summary.json`` — run metadata the parquet has no column
  for (per-checkpoint train loss / lr from the bundle, the performance table,
  the P = I re-check, the epsilon-ceiling diagnostic, the data-fallback flag)
- ``examples/output/performance.csv``  — the performance table on its own

Phase 2 (``examples/two_rulers_llama_evaluate.py``) turns those two files into
the figure and the printed tables. It imports no vatis code, which is what keeps
the parquet honest as *the* contract with downstream consumers.

The training half lives in the other repo
(``antonio_code/runs/example_bundle_run.sh``) and the two never share a process:
the training stack pins one torch build, vatis another, and the **on-disk bundle
is the entire interface**. Run the training script first; it prints the exact
invocation for this one.

    python examples/two_rulers_llama.py --bundle <run>/bundle \
        --eval-batches <run>/eval_batches --device cuda

What is measured, and why it is shaped this way:

- **k = 2 fixed measurement batches (A1, A2) + 1 eval batch (B)**, all from the
  held-out tail of the shard, all frozen across checkpoints so a trajectory is a
  trajectory and not a resampling artifact. ``cross_pairs`` adds
  ``delta_loss(A_i, B)`` and ``chi_pos(A_i, B)`` at zero extra backward cost.
- **Both rulers in one pass.** ``preconditioner=source.adamw_preconditioner()``
  builds ``P = diag(1/(sqrt(v_hat) + eps))`` from each checkpoint's own saved
  (pre-step) optimizer state; the P observables come out of the same probes as
  the raw ones, so raw values are bit-identical to a ``preconditioner=None`` run.
- **A step-0 checkpoint has no AdamW state**, so its P rows are deliberately
  skipped (with a warning) — pre-step semantics means "before the first update"
  and torch creates ``exp_avg_sq`` lazily inside the first ``step()``. Every
  table and the figure therefore has to cope with a checkpoint that has raw rows
  and no P rows; they do, by leaving the point out rather than plotting a zero.
- **The epsilon ceiling is reported, not hidden.** With the frozen convention
  ``P = 1/(sqrt(v_hat) + eps)``, a coordinate the optimizer never moved sits at
  ``1/eps`` — ~1e7x a live coordinate's weight — and unused vocabulary rows are
  exactly such coordinates. Nothing is clipped, floored or masked (the
  convention is a project decision); instead this script measures the share of
  the P diagonal at the ceiling and, on a few checkpoints, the share of
  ``chi_net^P`` those coordinates carry on the held-out batch B. Both go into
  the summary and onto the figure. Read a ``chi_pos^P`` without that number at
  your peril.

No estimator changes are needed for any of the above, and none are made here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vatis.analyzer import Analyzer  # noqa: E402
from vatis.core.preconditioner import (  # noqa: E402
    RAW_DESCRIPTOR,
    Preconditioner,
    epsilon_cap_mask,
    resolve_adamw_hparams,
)
from vatis.data.collate import shifted_lm_targets  # noqa: E402
from vatis.models import load_local_bundle  # noqa: E402
from vatis.models.local_bundle import (  # noqa: E402
    LocalBundleSource,
    adamw_preconditioner_from_state,
)
from vatis.sinks.base import ResultRow  # noqa: E402
from vatis.sinks.parquet import ParquetSink  # noqa: E402

# The three held-out batches, in the order the training script dumps them:
# two measurement batches and one eval batch. Names land in the parquet.
BATCH_NAMES = ("A1", "A2", "B")
EVAL_BATCH_GLOB = "eval_batch_*.npy"
FALLBACK_MARKER = "DATA_FALLBACK"

RAW_TO_P = {
    "delta_loss": "delta_loss_P",
    "chi_net": "chi_net_P",
    "chi_pos": "chi_pos_P",
}

DEFAULT_OUT = Path(__file__).parent / "output"


# --------------------------------------------------------------- model factory


class WrappedLlama(torch.nn.Module):
    """The project Llama under the ``model.`` parameter-name prefix.

    The bundle's ``param_names.json`` records the *training* LightningModule's
    names (``model.tok_embeddings.weight``, ...) because that is the module the
    optimizer was built from. A factory returning a bare ``Transformer`` is
    therefore rejected by the loader's canonical-layout check — correctly, since
    a silent name mismatch would mean a permuted P. The fix belongs in the
    factory, which is exactly what this wrapper is.

    ``forward(tokens) -> logits`` keeps
    :func:`vatis.models.local_bundle.plain_lm_forward` working unchanged.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.model = inner

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.model(tokens)
        return out


def llama_factory(metadata: Any) -> torch.nn.Module:
    """``model_factory(metadata) -> nn.Module`` for the project Llama.

    vatis stays project-agnostic: the architecture is described by the bundle's
    own ``model_config`` and instantiated by user code (this function), never by
    vatis. The import needs the training repo's *pure-torch model definition*
    only — no lightning, no perspic, nothing that would pull the training stack's
    torch build into this process.
    """
    llama_src = os.environ.get("WP0_LLAMA_SRC", "/workspace/antonio_code")
    if llama_src not in sys.path:
        sys.path.insert(0, llama_src)
    try:
        from llama_model import ModelArgs, Transformer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            f"cannot import the project Llama from {llama_src!r} (needs llama_model/model.py). "
            "Point WP0_LLAMA_SRC at the training repo."
        ) from exc
    config = dict(metadata["model_config"])
    return WrappedLlama(Transformer(ModelArgs(**config)))


# ------------------------------------------------------------------- batches


def load_measurement_batches(
    directory: Path,
    *,
    seq_len: int,
    vocab_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], bool]:
    """The held-out measurement batches, or a loud random-token fallback.

    Files are ``[B, seq_len]`` int32 token ids in the same whole-sequence
    convention as the bundle's ``batch_a.npy``, so a batch is turned into an LM
    batch the same way the loader does it (``labels`` = shift by one, last
    position ignored). Padding is *not* masked: this training stack's criterion
    masks nothing and ``batch_a.npy`` carries no mask, so masking here would
    measure a different objective than the one that was trained.

    Returns ``(batches, used_random_fallback)``.
    """
    paths = sorted(directory.glob(EVAL_BATCH_GLOB)) if directory.is_dir() else []
    if len(paths) >= len(BATCH_NAMES):
        batches: dict[str, dict[str, torch.Tensor]] = {}
        for name, path in zip(BATCH_NAMES, paths[: len(BATCH_NAMES)], strict=True):
            array = np.load(path, allow_pickle=False)
            ids = torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.int64, device=device)
            if ids.ndim != 2:
                raise ValueError(
                    f"{path}: expected a 2-D [B, seq_len] token array, got {ids.shape}"
                )
            batches[name] = {"input_ids": ids, "labels": shifted_lm_targets(ids)}
            print(f"    {name}: {path.name} shape={tuple(ids.shape)}")
        return batches, False

    # Spec 05 §1: no shard -> random tokens, LOUDLY, and logged. This path is a
    # pipeline demo only; chi_net is the Jacobian norm *evaluated at the input*,
    # so off-distribution tokens make every value scientifically empty.
    banner = "#" * 78
    message = (
        f"{banner}\n"
        f"##  WARNING: RANDOM-TOKEN FALLBACK -- THIS ANALYSIS IS SCIENTIFICALLY EMPTY.\n"
        f"##  No {EVAL_BATCH_GLOB} files under {directory} (found {len(paths)}, "
        f"need {len(BATCH_NAMES)}), so the measurement batches are uniformly\n"
        f"##  random token ids. The pipeline is exercised end to end; the numbers\n"
        f"##  mean nothing. Re-run the training script to dump the held-out batches.\n"
        f"{banner}"
    )
    print(message, flush=True)
    warnings.warn(
        "random-token measurement batches: no held-out shard found, values are "
        "scientifically empty (pipeline demo only)",
        UserWarning,
        stacklevel=2,
    )
    generator = torch.Generator(device="cpu").manual_seed(20260728)
    fallback: dict[str, dict[str, torch.Tensor]] = {}
    for name in BATCH_NAMES:
        ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), generator=generator, dtype=torch.int64
        ).to(device)
        fallback[name] = {"input_ids": ids, "labels": shifted_lm_targets(ids)}
    return fallback, True


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


# ------------------------------------------------------------------- analysis


def build_analyzer(
    args: argparse.Namespace,
    eval_batches: dict[str, dict[str, torch.Tensor]],
    preconditioner: Any,
    sink: ParquetSink | None,
) -> Analyzer:
    """One place where the estimator configuration is fixed for every pass.

    ``per_sequence_cv`` is the auto-pick for these batch sizes (B <= 32) and the
    method the golden gate uses, so the example exercises the same path.
    """
    return Analyzer(
        eval_batches=eval_batches,
        preconditioner=preconditioner,
        chi_net_method=args.chi_net_method,
        n_hutchinson=args.n_hutchinson,
        micro_batch_size=args.micro_batch_size,
        cross_pairs=[(name, "B") for name in eval_batches if name != "B"],
        sink=sink,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )


def all_rows(results: list[Any]) -> list[ResultRow]:
    return [row for result in results for row in result.rows]


def identity_recheck(
    source: LocalBundleSource,
    args: argparse.Namespace,
    eval_batches: dict[str, dict[str, torch.Tensor]],
    step: int,
) -> dict[str, Any]:
    """P = I must reproduce the raw columns exactly. Acceptance criterion 5.

    Compared *within one run*: the raw rows of a preconditioned pass are
    bit-identical to a raw-only pass, so the honest check is
    ``chi_pos_P == chi_pos`` from the same probes, same checkpoint, same batch.
    """
    checkpoint = source.checkpoint_for(step)
    analyzer = build_analyzer(args, eval_batches, "identity", None)
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


def performance_table(
    timings: list[CheckpointTiming],
    *,
    total_s: float,
    peak_gpu_mb: float | None,
    n_rows: int,
) -> str:
    """The performance table as text. Spec 05 §3: a deliverable, not a nicety."""
    peak_total = "-" if peak_gpu_mb is None else f"{peak_gpu_mb:.0f}"
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
            f"{sum(analysis) / len(analysis):>11.2f} {peak_total:>12}"
        )
        lines.append(f"{'total':>16} {'':>6} {sum(loads):>8.2f} {sum(analysis):>11.2f} {'':>12}")
    lines.append("")
    lines.append(f"  checkpoints          : {len(timings)}")
    lines.append(f"  result rows          : {n_rows}")
    lines.append(f"  end-to-end wall time : {total_s:.1f} s")
    lines.append(f"  peak GPU memory      : {peak_total} MiB (torch allocator, this process)")
    lines.append(
        "  caveat               : measured on a SHARED GPU (a co-tenant process was "
        "resident); wall times are as-measured, not tuned or best-of-N."
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


# ------------------------------------------------------------------------ main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    default_root = os.environ.get("WP0_EXAMPLE_ROOT")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(os.environ["WP0_BUNDLE_DIR"])
        if "WP0_BUNDLE_DIR" in os.environ
        else (Path(default_root) / "bundle" if default_root else None),
        help="checkpoint bundle root (metadata.json, param_names.json, checkpoints/). "
        "Defaults to $WP0_BUNDLE_DIR, else $WP0_EXAMPLE_ROOT/bundle.",
    )
    parser.add_argument(
        "--eval-batches",
        type=Path,
        default=None,
        help="directory of held-out eval_batch_*.npy files (default: <bundle>/../eval_batches)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device; 'cuda' (default) falls back to CPU with a warning when "
        "no CUDA device is visible, so the example stays runnable anywhere",
    )
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--n-hutchinson", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--chi-net-method", default="per_sequence_cv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit-checkpoints",
        type=int,
        default=0,
        help="analyse only the first N checkpoints (0 = all); for smoke tests",
    )
    parser.add_argument(
        "--eps-cap-checkpoints",
        type=int,
        default=3,
        help="how many checkpoints get the (2 extra passes) chi_net^P ceiling split",
    )
    args = parser.parse_args(argv)
    if args.bundle is None:
        parser.error(
            "no bundle given: pass --bundle, or set WP0_BUNDLE_DIR / WP0_EXAMPLE_ROOT. "
            "The bundle is written by the training side "
            "(antonio_code/runs/example_bundle_run.sh)."
        )
    args.bundle = args.bundle.expanduser().resolve()
    if args.eval_batches is None:
        args.eval_batches = args.bundle.parent / "eval_batches"
    args.eval_batches = Path(args.eval_batches).expanduser().resolve()
    args.chi_net_method = args.chi_net_method or None
    return args


def resolve_device(requested: str) -> torch.device:
    """``--device`` with a clean CPU fallback. No hardcoded devices anywhere."""
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(
            f"note: no CUDA device visible, falling back to CPU (requested {requested!r}). "
            "The example is correct but much slower there.",
            flush=True,
        )
        return torch.device("cpu")
    return device


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "results.parquet"
    if parquet_path.exists():
        parquet_path.unlink()  # ParquetSink truncates on re-open; be explicit

    print("=" * 78)
    print("WP0 example -- vatis two-ruler analysis of a real Llama checkpoint bundle")
    print("=" * 78)

    source = load_local_bundle(
        args.bundle,
        llama_factory,
        dtype=args.dtype,
        device=args.device,
        steps=None,
    )
    metadata = source.metadata
    steps = list(source.steps)
    if args.limit_checkpoints:
        steps = steps[: args.limit_checkpoints]
        source = load_local_bundle(
            args.bundle, llama_factory, dtype=args.dtype, device=args.device, steps=steps
        )

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
        f"  estimator       : {args.chi_net_method or 'auto'} n_hutchinson={args.n_hutchinson} "
        f"micro_batch_size={args.micro_batch_size} seed={args.seed}"
    )

    print("  measurement batches (k=2 fixed A + 1 eval B, held-out tail of the shard):")
    eval_batches, used_random_batches = load_measurement_batches(
        args.eval_batches,
        seq_len=int(metadata["seq_len"]),
        vocab_size=int(metadata["model_config"]["vocab_size"]),
        batch_size=int(metadata.get("B_train") or 8),
        device=args.device,
    )
    training_fallback = (args.bundle.parent / FALLBACK_MARKER).is_file()
    if training_fallback:
        print(
            f"  !! {FALLBACK_MARKER} present: the TRAINING run used random tokens too. "
            "Nothing in this analysis is a measurement."
        )

    # ---- the sweep: both rulers, every checkpoint, one pass.
    timings: list[CheckpointTiming] = []
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    t_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer = build_analyzer(
            args, eval_batches, source.adamw_preconditioner(), ParquetSink(str(parquet_path))
        )
        results = analyzer.run(timed_model_specs(source, timings, device=args.device))
    total_s = time.perf_counter() - t_start
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated(args.device) / 2**20 if args.device.type == "cuda" else None
    )
    # Nothing is swallowed: the run's warnings are captured only so the step-0
    # policy can be reported as a result rather than scrolling past.
    skip_warnings = [str(w.message) for w in caught if "optimizer state is empty" in str(w.message)]
    for warning in caught:
        tag = "step-0 P skip" if "optimizer state is empty" in str(warning.message) else "warning"
        print(f"\n  {tag}: {str(warning.message).splitlines()[0]}")

    rows = all_rows(results)
    p_steps = sorted({r.checkpoint_id for r in rows if r.observable in RAW_TO_P.values()})
    nonfinite = [
        (r.checkpoint_id, r.observable, r.batch_a, r.batch_b, r.value)
        for r in rows
        if not np.isfinite(r.value)
    ]
    print(f"\n  rows written    : {len(rows)} -> {parquet_path}")
    print(f"  P rows present  : {len(p_steps)} of {len(timings)} checkpoints")
    print(f"  non-finite      : {len(nonfinite)}")
    if nonfinite:
        for item in nonfinite[:8]:
            print(f"      {item}")

    # ---- acceptance: P = I equals the raw columns (on the last checkpoint).
    identity = identity_recheck(source, args, eval_batches, steps[-1])
    print(
        f"\n  P=I re-check    : {identity['pairs_compared']} pairs at "
        f"{identity['checkpoint_id']}, worst |P - raw| = "
        f"{identity['worst_abs_deviation']:.3e} (tolerance {identity['tolerance']:.0e}) -> "
        f"{'PASS' if identity['passed'] else 'FAIL'}"
    )

    # ---- the epsilon ceiling, reported with the numbers.
    with_state = [s for s in steps if s != 0]
    n_eps = max(0, min(args.eps_cap_checkpoints, len(with_state)))
    if n_eps:
        picks = np.unique(np.linspace(0, len(with_state) - 1, n_eps).round().astype(int))
        chi_net_steps = [with_state[i] for i in picks]
    else:
        chi_net_steps = []
    print(f"\n  epsilon-ceiling diagnostic (chi_net^P split at steps {chi_net_steps}):")
    eps_records = epsilon_ceiling_diagnostic(source, args, eval_batches, chi_net_steps)
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

    # ---- performance table: print + save.
    table = performance_table(timings, total_s=total_s, peak_gpu_mb=peak_gpu_mb, n_rows=len(rows))
    print("\n" + "=" * 78)
    print("performance (analysis phase; a co-tenant process shares this GPU)")
    print("=" * 78)
    print(table)
    write_csv(out_dir / "performance.csv", timings)

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

    summary: dict[str, Any] = {
        "kind": "wp0_two_rulers_llama",
        "bundle": str(args.bundle),
        "eval_batches_dir": str(args.eval_batches),
        "random_token_measurement_batches": used_random_batches,
        "random_token_training_run": training_fallback,
        "scientifically_empty": used_random_batches or training_fallback,
        "config": {
            "device": str(args.device),
            "dtype": args.dtype,
            "chi_net_method": args.chi_net_method,
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
            "gpu_contended": "yes -- another process shares this card; timings are "
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
            "peak_gpu_mb": peak_gpu_mb,
            "n_rows": len(rows),
            "table": table,
        },
        "step0_p_skip_warnings": skip_warnings,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {parquet_path}")
    print(f"wrote {out_dir / 'performance.csv'}")
    print(f"wrote {out_dir / 'run_summary.json'}")
    print("\nNext (phase 2, cheap, no vatis imports -- figure + tables):")
    print(f"  python examples/two_rulers_llama_evaluate.py --dir {out_dir}")
    return 0 if not nonfinite and identity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
