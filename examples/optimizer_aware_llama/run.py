"""Raw and preconditioned observables over a real Llama checkpoint bundle. Step 1 of 2: compute.

Reads a checkpoint bundle written by the training repository, measures the raw
observables and their AdamW-preconditioned versions at every checkpoint in one
pass, and writes

- ``results.parquet``  — the long-format result rows
- ``run_summary.json`` — what the parquet has no column for: per-checkpoint
  train loss and learning rate from the bundle, the performance table, the
  ``P = I`` re-check, the epsilon-ceiling records, the random-token flag
- ``performance.csv``  — the performance table on its own

into ``--out`` (default: this folder). ``evaluate.py`` turns them into the
figures and the printed tables and imports no vatis code. The question, what
it needs, the run time and the findings are in ``README.md`` and
``details.md`` next to this script.

    python examples/optimizer_aware_llama/run.py --bundle <run>/bundle \\
        --eval-batches <run>/eval_batches --device cuda

The training half lives in the training repository
(``runs/example_bundle_run.sh``) and the two never share a process: the
training stack pins one torch build, vatis another, and the **on-disk bundle is
the entire interface**. Run the training script first; it prints the exact
invocation for this one.

The script reads top to bottom as the four steps of a use of vatis: load the
bundle, choose the batches, call vatis, write the results. Support code lives
next to it: ``llama_bundle.py`` (the model factory and the measurement
batches) and ``llama_checks.py`` (timing, the two self-checks, the summary).

What is measured, and why it is shaped this way:

- **k = 2 fixed measurement batches (A1, A2) + 1 eval batch (B)**, all from the
  held-out tail of the shard, all frozen across checkpoints so a trajectory is a
  trajectory and not a resampling artifact. ``cross_pairs`` adds
  ``delta_loss(A_i, B)`` and ``chi_pos(A_i, B)`` at zero extra backward cost.
- **One objective for every row.** The held-out batches are scored with the pad
  convention the bundle records (``source.loss_ignore_index``), the same one the
  loader applies to the update batch: pad targets are ignored iff the training
  loss ignored them. The run behind the committed output masks nothing in its
  criterion, and its numbers count every position.
- **Raw and preconditioned in one pass.** ``preconditioner=source.adamw_preconditioner()``
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
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository, for vatis
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this folder, for its support modules

from llama_bundle import (  # noqa: E402
    BATCH_NAMES,
    EVAL_BATCH_GLOB,
    FALLBACK_MARKER,
    WrappedLlama,
    llama_factory,
    lm_batch_from_tokens,
    load_measurement_batches,
)
from llama_checks import (  # noqa: E402
    RAW_TO_P,
    CheckpointTiming,
    all_rows,
    ceiling_split_steps,
    dead_vocab_rows,
    epsilon_ceiling_diagnostic,
    identity_recheck,
    performance_table,
    print_epsilon_ceiling,
    print_run_header,
    run_summary,
    timed_model_specs,
    write_csv,
)

from vatis.analyzer import Analyzer  # noqa: E402
from vatis.models import load_local_bundle  # noqa: E402
from vatis.sinks.parquet import ParquetSink  # noqa: E402

# Moved to the two support modules; still importable from this script.
__all__ = [
    "BATCH_NAMES",
    "EVAL_BATCH_GLOB",
    "FALLBACK_MARKER",
    "RAW_TO_P",
    "CheckpointTiming",
    "WrappedLlama",
    "dead_vocab_rows",
    "llama_factory",
    "lm_batch_from_tokens",
    "load_measurement_batches",
]

DEFAULT_OUT = Path(__file__).parent


def build_analyzer(
    args: argparse.Namespace,
    eval_batches: dict[str, dict[str, torch.Tensor]],
    preconditioner: Any,
    sink: ParquetSink | None,
) -> Analyzer:
    """One place where the estimator configuration is fixed for every pass.

    ``per_sequence_cv`` is this script's default (``--chi-net-method``) and the
    method the golden gate uses, so the example exercises the same path. It is
    not vatis's own default, which is ``hutchinson``.
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


def main(argv: list[str] | None = None) -> int:
    t_script = time.perf_counter()
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
    print("vatis example -- raw and preconditioned observables of a real Llama checkpoint bundle")
    print("=" * 78)

    # ---- 1. Load the bundle: per checkpoint the weights, the saved optimizer
    # state and the training metadata; the model is rebuilt by llama_factory.
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
    print_run_header(args, source, steps)

    # ---- 2. Choose the batches. The pad convention the training run recorded
    # (None = its loss masked nothing). The loader applies it to the update
    # batch; the held-out batches below get the same one, so every row describes
    # the same objective.
    loss_ignore_index = source.loss_ignore_index
    print(
        "  loss padding    : "
        + (
            "unmasked -- every target counts (the bundle records no loss_ignore_index)"
            if loss_ignore_index is None
            else f"MASKED -- targets equal to {loss_ignore_index} are ignored, as in training"
        )
    )
    print("  measurement batches (k=2 fixed A + 1 eval B, held-out tail of the shard):")
    eval_batches, used_random_batches = load_measurement_batches(
        args.eval_batches,
        seq_len=int(metadata["seq_len"]),
        vocab_size=int(metadata["model_config"]["vocab_size"]),
        batch_size=int(metadata.get("B_train") or 8),
        device=args.device,
        loss_ignore_index=loss_ignore_index,
    )
    training_fallback = (args.bundle.parent / FALLBACK_MARKER).is_file()
    if training_fallback:
        print(
            f"  !! {FALLBACK_MARKER} present: the TRAINING run used random tokens too. "
            "Nothing in this analysis is a measurement."
        )

    # ---- 3. Call vatis: raw and preconditioned observables, every checkpoint,
    # one pass. The AdamW preconditioner is read from each checkpoint's saved
    # optimizer state; timed_model_specs only records the time per checkpoint.
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

    # ---- Self-check 1: P = I reproduces the raw values (on the last checkpoint).
    identity = identity_recheck(
        source, build_analyzer(args, eval_batches, "identity", None), steps[-1]
    )
    print(
        f"\n  P=I re-check    : {identity['pairs_compared']} pairs at "
        f"{identity['checkpoint_id']}, worst |P - raw| = "
        f"{identity['worst_abs_deviation']:.3e} (tolerance {identity['tolerance']:.0e}) -> "
        f"{'PASS' if identity['passed'] else 'FAIL'}"
    )

    # ---- Self-check 2: the epsilon ceiling, reported with the numbers.
    chi_net_steps = ceiling_split_steps(steps, args.eps_cap_checkpoints)
    print(f"\n  epsilon-ceiling diagnostic (chi_net^P split at steps {chi_net_steps}):")
    eps_records = epsilon_ceiling_diagnostic(source, args, eval_batches, chi_net_steps)
    print_epsilon_ceiling(eps_records)

    # ---- 4. Write the results: the performance table, then run_summary.json
    # (results.parquet was written by the sink during step 3).
    script_s = time.perf_counter() - t_script
    table = performance_table(
        timings,
        total_s=total_s,
        script_s=script_s,
        peak_gpu_mb=peak_gpu_mb,
        n_rows=len(rows),
    )
    print("\n" + "=" * 78)
    print(f"performance (analysis phase, {args.device})")
    print("=" * 78)
    print(table)
    write_csv(out_dir / "performance.csv", timings)

    summary = run_summary(
        args,
        source,
        used_random_batches=used_random_batches,
        training_fallback=training_fallback,
        identity=identity,
        eps_records=eps_records,
        nonfinite=nonfinite,
        timings=timings,
        total_s=total_s,
        script_s=script_s,
        peak_gpu_mb=peak_gpu_mb,
        n_rows=len(rows),
        table=table,
        skip_warnings=skip_warnings,
    )
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {parquet_path}")
    print(f"wrote {out_dir / 'performance.csv'}")
    print(f"wrote {out_dir / 'run_summary.json'}")
    print("\nNext (phase 2, cheap, no vatis imports -- figure + tables):")
    print(f"  python examples/optimizer_aware_llama/evaluate.py --dir {out_dir}")
    return 0 if not nonfinite and identity["passed"] else 1


# ------------------------------------------------------------------ command line


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


if __name__ == "__main__":
    raise SystemExit(main())
