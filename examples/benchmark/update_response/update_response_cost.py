"""What does measuring the step that was taken cost? Wall-clock and peak memory.

Times the analyzer over a few checkpoints of a native bundle in three modes:

    baseline   the existing raw rows only (no update rule)
    formula    + ``update_response=source.update_response(realized=False)``:
               the ``loss_change_first_order*`` rows and the two step rows
    realized   + the realized rows as well (``loss_change_first_order_weights``,
               ``loss_change``, ``loss_change_residual``), which load
               ``theta_{t+1}`` and run two more forward passes per eval batch

Run it on an SGD bundle and on an AdamW bundle to see both what the feature adds
to the existing analysis and what the AdamW rule costs next to the SGD rule.

**Every measurement runs in its own process**, so peak memory is that
measurement's own. Per process this records

- ``wall_s``: wall-clock of ``Analyzer.run`` over the selected checkpoints,
  checkpoint loading included (``time.perf_counter``);
- ``peak_rss_mb``: the peak resident set size of the process over its whole
  life, ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (interpreter, torch and
  the model included);
- ``start_rss_mb``: the resident set size just before the bundle is opened
  (``/proc/self/statm``), so ``peak_rss_mb - start_rss_mb`` is what the analysis
  itself added on top of the imports;
- with ``--device cuda``: ``peak_gpu_mb``, ``torch.cuda.max_memory_allocated()``
  of the process after the run (the peak of this process's own tensors on the
  card; not what other processes hold, and not the allocator's cached blocks).
  The wall-clock is then taken after ``torch.cuda.synchronize()``.

The eval batches are the checkpoint's own update batch and ``--held-out`` held-out
batches (default one), in every mode, so the modes differ only in the update rule. Real batches from
the bundle are used because the update batch has to be the one the step
consumed; no observable value is reported here.

Three sub-commands::

    # one measurement, in this process; prints one JSON line
    python update_response_cost.py worker --bundle <run>/bundle \\
        --eval-batches <run>/eval_batches --mode formula --clip-grad-norm 1.0 \\
        --steps 1 661

    # every mode x repeat for one bundle, each in a fresh process; appends JSON lines
    python update_response_cost.py sweep --label adamw --bundle <run>/bundle \\
        --eval-batches <run>/eval_batches --clip-grad-norm 1.0 --steps 1 661 \\
        --repeats 3 --out results.jsonl

    # the markdown table of a results file
    python update_response_cost.py table results.jsonl

The model comes from ``--model-factory`` (a ``"module:function"`` import string,
as for ``load_local_bundle``); the default builds the project Llama from the
directory named by ``VATIS_LLAMA_MODEL_PATH``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

MODES = ("baseline", "formula", "realized")
DEFAULT_LLAMA_PATH = "/workspace/antonio_code"


# --------------------------------------------------------------- model factory


def llama_factory(metadata: Any) -> Any:
    """``model_factory(metadata)`` for the project Llama (``model.``-prefixed names)."""
    import torch

    source = os.environ.get("VATIS_LLAMA_MODEL_PATH", DEFAULT_LLAMA_PATH)
    if source not in sys.path:
        sys.path.insert(0, source)
    from llama_model import ModelArgs, Transformer  # type: ignore[import-not-found]

    class Wrapped(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.model = inner

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.model(tokens)
            return out

    return Wrapped(Transformer(ModelArgs(**dict(metadata["model_config"]))))


# ---------------------------------------------------------------------- worker


def _rss_mb() -> float:
    """Current resident set size in MB (Linux ``/proc/self/statm``)."""
    with open("/proc/self/statm", encoding="ascii") as handle:
        resident_pages = int(handle.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / 2**20


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    from vatis.analyzer import Analyzer
    from vatis.data.collate import shifted_lm_targets
    from vatis.models.local_bundle import load_local_bundle

    torch.manual_seed(0)
    if args.threads:
        torch.set_num_threads(args.threads)
    start_rss = _rss_mb()

    factory = args.model_factory or llama_factory
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    source = load_local_bundle(args.bundle, factory, steps=args.steps, dtype="fp32", device=device)
    paths = sorted(Path(args.eval_batches).glob("eval_batch_*.npy"))
    if len(paths) < args.held_out:
        raise SystemExit(
            f"{args.eval_batches}: {len(paths)} held-out batches, need {args.held_out}"
        )
    eval_batches: dict[str, Any] = {"train": source.batch_a_spec()}
    for index, path in enumerate(reversed(paths[-args.held_out :])):
        ids = torch.from_numpy(np.ascontiguousarray(np.load(path))).to(torch.int64).to(device)
        labels = shifted_lm_targets(ids)
        if source.loss_ignore_index is not None:
            labels = labels.masked_fill(labels == source.loss_ignore_index, -100)
        eval_batches[f"val{index}"] = {"input_ids": ids, "labels": labels}

    update_response = None
    if args.mode != "baseline":
        clip = None if args.clip_grad_norm == "none" else float(args.clip_grad_norm)
        update_response = source.update_response(
            clip_grad_norm=clip, realized=args.mode == "realized"
        )
    analyzer = Analyzer(
        eval_batches=eval_batches,
        update_response=update_response,
        chi_net_method="hutchinson",
        n_hutchinson=args.n_hutchinson,
        micro_batch_size=args.micro_batch_size,
        seed=0,
        device=device,
        dtype="fp32",
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = analyzer.run(source.model_specs())
    if cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    observables = sorted({row.observable for result in results for row in result.rows})
    n_params = source.layout.total_numel
    return {
        "label": args.label,
        "mode": args.mode,
        "bundle": Path(args.bundle).parent.name,
        "optimizer": source.metadata.optimizer_name,
        "steps": list(source.steps),
        "n_params": n_params,
        "update_batch_shape": list(source.checkpoints[0].batch_a_tokens.shape),
        "held_out_shape": list(ids.shape),
        "n_eval_batches": len(eval_batches),
        "n_hutchinson": args.n_hutchinson,
        "micro_batch_size": args.micro_batch_size,
        "threads": torch.get_num_threads(),
        "wall_s": wall,
        "wall_s_per_checkpoint": wall / len(source.steps),
        "start_rss_mb": start_rss,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if cuda else "cpu",
        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None,
        "n_rows": sum(len(result.rows) for result in results),
        "has_first_order_rows": "loss_change_first_order" in observables,
        "has_realized_rows": "loss_change" in observables,
        "load_average_1min": os.getloadavg()[0],
        "torch": torch.__version__,
    }


# ----------------------------------------------------------------------- sweep


def run_sweep(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Repeats in the outer loop: a drift in the load of a shared machine then
    # hits every mode alike instead of one mode's block of runs.
    for repeat in range(args.first_repeat, args.first_repeat + args.repeats):
        for mode in MODES:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--label",
                args.label,
                "--bundle",
                args.bundle,
                "--eval-batches",
                args.eval_batches,
                "--mode",
                mode,
                "--clip-grad-norm",
                str(args.clip_grad_norm),
                "--n-hutchinson",
                str(args.n_hutchinson),
                "--micro-batch-size",
                str(args.micro_batch_size),
                "--threads",
                str(args.threads),
                "--held-out",
                str(args.held_out),
                "--device",
                args.device,
                "--steps",
                *[str(step) for step in args.steps],
            ]
            if args.model_factory:
                cmd += ["--model-factory", args.model_factory]
            done = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if done.returncode != 0:
                raise RuntimeError(f"worker failed ({mode}):\n{done.stderr[-2000:]}")
            record = json.loads(done.stdout.strip().splitlines()[-1])
            record["repeat"] = repeat
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"{args.label:>6} {mode:<9} repeat {repeat}: {record['wall_s']:7.1f} s, "
                f"peak RSS {record['peak_rss_mb']:7.0f} MB",
                flush=True,
            )


# ----------------------------------------------------------------------- table


def _spread(values: list[float], fmt: str) -> str:
    return (
        f"{format(statistics.median(values), fmt)} "
        f"({format(min(values), fmt)} – {format(max(values), fmt)})"
    )


def make_table(path: Path) -> str:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    labels = list(dict.fromkeys(r["label"] for r in records))
    on_gpu = any(r.get("peak_gpu_mb") is not None for r in records)
    lines = [
        "| bundle | mode | runs | wall-clock per checkpoint, s: median (min – max) | "
        "vs baseline | peak RSS, MB: median (min – max) | vs baseline, MB | "
        "peak − start RSS, MB (median) |"
        + (" peak GPU memory, MB: median (min – max) | vs baseline, MB |" if on_gpu else ""),
        "|---|---|---|---|---|---|---|---|" + ("---|---|" if on_gpu else ""),
    ]
    for label in labels:
        base_wall = base_rss = base_gpu = None
        for mode in MODES:
            rows = [r for r in records if r["label"] == label and r["mode"] == mode]
            if not rows:
                continue
            wall = [r["wall_s_per_checkpoint"] for r in rows]
            rss = [r["peak_rss_mb"] for r in rows]
            grown = [r["peak_rss_mb"] - r["start_rss_mb"] for r in rows]
            gpu = [r["peak_gpu_mb"] for r in rows if r.get("peak_gpu_mb") is not None]
            if mode == "baseline":
                base_wall, base_rss = statistics.median(wall), statistics.median(rss)
                base_gpu = statistics.median(gpu) if gpu else None
            gpu_cells = ""
            if on_gpu and gpu:
                more = (
                    "—"
                    if mode == "baseline" or base_gpu is None
                    else f"{statistics.median(gpu) - base_gpu:+.0f}"
                )
                gpu_cells = f" {_spread(gpu, '.0f')} | {more} |"
            elif on_gpu:
                gpu_cells = " — | — |"
            ratio = (
                "—"
                if mode == "baseline" or not base_wall
                else (f"×{statistics.median(wall) / base_wall:.2f}")
            )
            extra = (
                "—"
                if mode == "baseline" or base_rss is None
                else (f"{statistics.median(rss) - base_rss:+.0f}")
            )
            lines.append(
                f"| {label} ({rows[0]['optimizer']}) | {mode} | {len(rows)} | "
                f"{_spread(wall, '.1f')} | {ratio} | {_spread(rss, '.0f')} | {extra} | "
                f"{statistics.median(grown):.0f} |" + gpu_cells
            )
    first = records[0]
    header = (
        f"- model: {first['n_params']:,} parameters "
        f"(one fp32 parameter vector = {first['n_params'] * 4 / 2**20:.0f} MB)\n"
        f"- checkpoints per run: steps {first['steps']}; eval batches: the update batch "
        f"{first['update_batch_shape']} plus held-out batches of {first['held_out_shape']} "
        f"tokens (number of eval batches per label: "
        f"{ {label: next(r['n_eval_batches'] for r in records if r['label'] == label) for label in labels} })\n"
        f"- `chi_net_method=hutchinson`, `n_hutchinson` per label: "
        f"{ {label: next(r['n_hutchinson'] for r in records if r['label'] == label) for label in labels} }, "
        f"`micro_batch_size={first['micro_batch_size']}`, fp32, device "
        f"{first.get('device_name', 'cpu')}, "
        f"{first['threads']} torch threads, torch {first['torch']}\n"
        f"- 1-minute load average of the machine during the runs: "
        f"{min(r['load_average_1min'] for r in records):.1f} – "
        f"{max(r['load_average_1min'] for r in records):.1f}\n"
    )
    return header + "\n" + "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--label", default="bundle")
        p.add_argument("--bundle", required=True)
        p.add_argument("--eval-batches", required=True)
        p.add_argument("--steps", type=int, nargs="+", required=True)
        p.add_argument("--clip-grad-norm", default="none", help="max norm, or 'none'")
        p.add_argument("--n-hutchinson", type=int, default=2)
        p.add_argument("--micro-batch-size", type=int, default=8)
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--held-out", type=int, default=1, help="number of held-out eval batches")
        p.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
        p.add_argument("--model-factory", default=None, help="'module:function'")

    worker = sub.add_parser("worker", help="one measurement in this process")
    common(worker)
    worker.add_argument("--mode", choices=MODES, required=True)
    sweep = sub.add_parser("sweep", help="every mode x repeat, each in its own process")
    common(sweep)
    sweep.add_argument("--repeats", type=int, default=3)
    sweep.add_argument(
        "--first-repeat",
        type=int,
        default=0,
        help="number of the first repeat. For sweeps that are interleaved by calling "
        "this script once per repeat: without it every record would say repeat 0",
    )
    sweep.add_argument("--out", required=True)
    table = sub.add_parser("table", help="markdown table of a results file")
    table.add_argument("results")

    args = parser.parse_args()
    if args.command == "worker":
        print(json.dumps(run_worker(args)))
    elif args.command == "sweep":
        run_sweep(args)
    else:
        print(make_table(Path(args.results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
