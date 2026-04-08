"""argparse-based CLI for vatis.

Usage:

    python -m vatis run \\
        --model EleutherAI/pythia-160m \\
        --revisions step1000 step2000 step10000 \\
        --eval-batch val=path/to/val_batch.pt \\
        --sink results.parquet \\
        --num-gpus 4

For ``num_gpus > 1`` this re-execs under ``torchrun`` and dispatches to
``vatis._worker``. For ``num_gpus == 1`` it runs in-process.

The CLI deliberately exposes a small surface — the rich Python API in
``vatis.analyze`` is the recommended way to use vatis programmatically.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from vatis.analyzer import ALL_OBSERVABLES, Analyzer
from vatis.distributed.launcher import launch_torchrun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vatis", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run analysis on one or more checkpoints")
    run.add_argument(
        "--model",
        required=True,
        help="HuggingFace hub identifier (e.g. EleutherAI/pythia-160m)",
    )
    run.add_argument(
        "--revisions",
        nargs="+",
        default=[None],
        help="one or more revisions to load (e.g. step1000 step2000)",
    )
    run.add_argument(
        "--eval-batch",
        action="append",
        required=True,
        help="named eval batch in the form NAME=PATH where PATH is a "
        "torch.save'd dict with at least 'input_ids'. Repeat for "
        "multiple batches.",
    )
    run.add_argument(
        "--observables",
        nargs="+",
        default=list(ALL_OBSERVABLES),
        help=f"observables to compute. default: all of {list(ALL_OBSERVABLES)}",
    )
    run.add_argument(
        "--chi-net-method",
        choices=("hutchinson", "per_sequence_cv", "opacus"),
        default=None,
        help="chi_net estimator. default: auto-select based on batch size.",
    )
    run.add_argument("--n-hutchinson", type=int, default=32)
    run.add_argument(
        "--hutchinson-distribution",
        choices=("rademacher", "gaussian"),
        default="rademacher",
    )
    run.add_argument("--micro-batch-size", type=int, default=1)
    run.add_argument(
        "--cross-pair",
        action="append",
        default=[],
        help="A=B form: compute delta_loss(A, B). Repeat for multiple pairs.",
    )
    run.add_argument("--sink", required=True, help="output parquet path")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--num-gpus", type=int, default=1)
    run.add_argument(
        "--dtype",
        choices=("fp32", "bf16", "fp16"),
        default="bf16",
    )
    run.add_argument(
        "--device",
        default=None,
        help="device override. default: cuda:LOCAL_RANK if launched under "
        "torchrun, else cuda:0 if available, else cpu.",
    )
    run.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="passed through to transformers.from_pretrained",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.error(f"unknown command: {args.command}")

    if args.num_gpus > 1 and "RANK" not in __import__("os").environ:
        # Re-launch under torchrun. Strip the --num-gpus flag from the
        # forwarded args (the worker side defaults to 1 since RANK is set).
        forwarded = [a for a in (argv if argv is not None else sys.argv[1:])]
        return launch_torchrun(args.num_gpus, forwarded)

    return run_from_args(args)


def run_from_args(args: argparse.Namespace) -> int:
    """Run analysis from a parsed argparse Namespace.

    Used by both the in-process CLI and the worker side under torchrun.
    """
    eval_batches = _parse_eval_batches(args.eval_batch)
    cross_pairs = _parse_cross_pairs(args.cross_pair)

    # Resolve device.
    device = _resolve_device(args.device)

    analyzer = Analyzer(
        eval_batches=eval_batches,
        observables=args.observables,
        chi_net_method=args.chi_net_method,
        n_hutchinson=args.n_hutchinson,
        hutchinson_distribution=args.hutchinson_distribution,
        micro_batch_size=args.micro_batch_size,
        cross_pairs=cross_pairs,
        sink=args.sink,
        seed=args.seed,
        device=device,
        dtype=args.dtype,
    )

    specs: list[tuple[str, str | None]] = [(args.model, rev) for rev in args.revisions]
    analyzer.run(specs)
    return 0


def _parse_eval_batches(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--eval-batch arg {item!r} must be in NAME=PATH form")
        name, path = item.split("=", 1)
        path = path.strip()
        if not name.strip():
            raise SystemExit(f"--eval-batch arg has empty name: {item!r}")
        loaded = torch.load(Path(path), map_location="cpu", weights_only=True)
        out[name.strip()] = loaded
    return out


def _parse_cross_pairs(items: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--cross-pair arg {item!r} must be in A=B form")
        a, b = item.split("=", 1)
        out.append((a.strip(), b.strip()))
    return out


def _resolve_device(override: str | None) -> torch.device:
    import os

    if override is not None:
        return torch.device(override)
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{int(local_rank)}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


if __name__ == "__main__":
    raise SystemExit(main())
