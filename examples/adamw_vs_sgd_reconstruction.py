"""Which update rule reconstructs the step an AdamW run took? Phase 1: compute.

A small check, not a study. For a few checkpoints of a native bundle whose
optimizer was AdamW, and for the step's own update batch and one held-out
batch ``B``, it measures the first-order loss change of the step that was
actually taken,

    truth = grad L^B . (theta_{t+1} - theta_t)      from the two stored weight files,

and three predictions of it that use only what was known *before* the step:

    optimizer's rule   the update rebuilt from the saved optimizer state
                       (``loss_change_first_order``: for AdamW the sum of the
                       gradient, momentum-history and weight-decay parts)
    SGD rule           the same step as plain SGD would have taken it: no
                       momentum, no weight decay, the same learning rate, the
                       gradient the optimizer was handed,
                       ``-lr * clip_factor * grad L^B . grad L^A``
                       (and the same without the clip factor)
    -lr * delta_loss_P the hypothetical update ``-lr * P_t grad L^A`` with the
                       pre-step AdamW preconditioner, for reference

plus the realized loss change ``L_B(theta_{t+1}) - L_B(theta_t)``, so that what
first order itself misses is visible next to them.

This script is the expensive half: it runs vatis over the bundle and writes

- ``<out>/results.parquet``  the long-format rows (everything the comparison
  needs is a row: the learning rate and the clip factor of each step included)
- ``<out>/run_summary.json`` the settings of the run

``examples/adamw_vs_sgd_reconstruction_evaluate.py`` turns those into the table
and the figure and imports no vatis code.

    python examples/adamw_vs_sgd_reconstruction.py --bundle <run>/bundle \\
        --eval-batches <run>/eval_batches --clip-grad-norm 1.0 --device cuda

What the bundle has to provide: fp32 weight files, and for every step used the
*next* step as well — the truth is a difference of two stored weight vectors.
With ``--steps auto`` (default) the script takes the steps whose successor is
stored and thins them evenly to ``--max-steps``; it prints which it used.
Gradient clipping has to be stated (``--clip-grad-norm``) unless the bundle's
``metadata.json`` records both the value and the algorithm.

The model comes from ``--model-factory`` (``"module:function"``, as for
``load_local_bundle``); the default builds the project Llama from the directory
named by ``VATIS_LLAMA_MODEL_PATH``.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vatis.analyzer import Analyzer  # noqa: E402
from vatis.data.collate import shifted_lm_targets  # noqa: E402
from vatis.models import load_local_bundle  # noqa: E402
from vatis.sinks.parquet import ParquetSink  # noqa: E402

DEFAULT_OUT = Path(__file__).parent / "adamw_vs_sgd_reconstruction"
DEFAULT_LLAMA_PATH = "/workspace/antonio_code"
UPDATE_BATCH = "update_batch"
HELD_OUT = "held_out"


class WrappedLlama(torch.nn.Module):
    """The project Llama under the ``model.`` parameter-name prefix of its bundles."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.model = inner

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.model(tokens)
        return out


def llama_factory(metadata: Any) -> torch.nn.Module:
    """``model_factory(metadata) -> nn.Module`` for the project Llama."""
    source = os.environ.get("VATIS_LLAMA_MODEL_PATH", DEFAULT_LLAMA_PATH)
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from llama_model import ModelArgs, Transformer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            f"cannot import the project Llama from {source!r} (needs llama_model/model.py). "
            "Point VATIS_LLAMA_MODEL_PATH at the training repo, or pass --model-factory."
        ) from exc
    return WrappedLlama(Transformer(ModelArgs(**dict(metadata["model_config"]))))


def steps_with_successor(all_steps: tuple[int, ...]) -> list[int]:
    """The steps ``t`` for which the bundle also holds step ``t + 1``."""
    present = set(all_steps)
    return [step for step in all_steps if step + 1 in present]


def thin(steps: list[int], limit: int) -> list[int]:
    """At most ``limit`` of ``steps``, evenly spaced in rank, first and last kept."""
    if len(steps) <= limit:
        return steps
    picks = {round(i * (len(steps) - 1) / (limit - 1)) for i in range(limit)}
    return [steps[i] for i in sorted(picks)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True, help="bundle run directory")
    parser.add_argument(
        "--eval-batches",
        required=True,
        help="directory with eval_batch_*.npy ([B, seq_len] token ids); the last one is used",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["auto"],
        help="steps to measure (each needs its successor in the bundle), or 'auto'",
    )
    parser.add_argument("--max-steps", type=int, default=12, help="cap for --steps auto")
    parser.add_argument(
        "--clip-grad-norm",
        default="metadata",
        help="'metadata' (read the bundle), 'none' (the run did not clip), or the max norm",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument(
        "--n-hutchinson",
        type=int,
        default=1,
        help="probes for chi_net, which this comparison does not use (1 keeps it cheap)",
    )
    parser.add_argument("--model-factory", default=None, help="'module:function'")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    torch.manual_seed(0)
    factory = args.model_factory or llama_factory
    device = torch.device(args.device)

    everything = load_local_bundle(args.bundle, factory)
    usable = steps_with_successor(everything.steps)
    if args.steps == ["auto"]:
        steps = thin(usable, args.max_steps)
    else:
        steps = sorted({int(step) for step in args.steps})
        missing = [step for step in steps if step not in usable]
        if missing:
            raise SystemExit(
                f"step(s) {missing} have no stored successor (or are not in the bundle); the "
                f"truth needs theta_t and theta_(t+1). Steps with a stored successor: {usable}"
            )
    if not steps:
        raise SystemExit("the bundle holds no two consecutive steps; nothing to compare")
    print(
        f"bundle {args.bundle}: {len(everything.steps)} checkpoints, {len(usable)} with a "
        f"stored successor: {usable}"
    )
    print(f"measuring steps {steps}")

    clip: float | str | None
    if args.clip_grad_norm == "metadata":
        clip = "metadata"
    elif args.clip_grad_norm == "none":
        clip = None
    else:
        clip = float(args.clip_grad_norm)

    source = load_local_bundle(args.bundle, factory, steps=steps, dtype="fp32", device=device)
    paths = sorted(Path(args.eval_batches).glob("eval_batch_*.npy"))
    if not paths:
        raise SystemExit(f"{args.eval_batches}: no eval_batch_*.npy")
    ids = torch.from_numpy(np.ascontiguousarray(np.load(paths[-1]))).to(torch.int64).to(device)
    labels = shifted_lm_targets(ids)
    if source.loss_ignore_index is not None:
        # Score the held-out batch with the pad convention of the run, as the
        # loader does for the update batch.
        labels = labels.masked_fill(labels == source.loss_ignore_index, -100)
    held_out = {"input_ids": ids, "labels": labels}

    args.out.mkdir(parents=True, exist_ok=True)
    analyzer = Analyzer(
        eval_batches={UPDATE_BATCH: source.batch_a_spec(), HELD_OUT: held_out},
        preconditioner=source.preconditioner(),
        update_response=source.update_response(clip_grad_norm=clip),
        cross_pairs=[(UPDATE_BATCH, HELD_OUT)],
        chi_net_method="hutchinson",
        n_hutchinson=args.n_hutchinson,
        micro_batch_size=args.micro_batch_size,
        sink=ParquetSink(str(args.out / "results.parquet")),
        seed=0,
        device=device,
        dtype="fp32",
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = analyzer.run(source.model_specs())
    wall = time.perf_counter() - started
    skipped = sorted({str(w.message) for w in caught if "SKIPPED" in str(w.message)})
    for message in skipped:
        print("WARNING:", message)

    observables = {row.observable for result in results for row in result.rows}
    if "loss_change_first_order_weights" not in observables:
        raise SystemExit(
            "no realized rows were measured, so there is no truth to compare with: the "
            "weight files have to be stored as fp32 (see the warnings above)"
        )
    descriptors = sorted(
        {
            row.preconditioner
            for result in results
            for row in result.rows
            if row.observable == "loss_change_first_order"
        }
    )
    summary = {
        "bundle": Path(args.bundle).resolve().parent.name,
        "optimizer": source.metadata.optimizer_name,
        "update_rule": descriptors,
        "steps": steps,
        "steps_with_stored_successor": usable,
        "n_checkpoints_in_bundle": len(everything.steps),
        "clip_grad_norm": args.clip_grad_norm,
        "loss_ignore_index": source.loss_ignore_index,
        "n_parameters": source.layout.total_numel,
        "update_batch_shape": list(source.checkpoints[0].batch_a_tokens.shape),
        "held_out_batch": paths[-1].name,
        "held_out_shape": list(ids.shape),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": "fp32",
        "micro_batch_size": args.micro_batch_size,
        "n_hutchinson": args.n_hutchinson,
        "torch": torch.__version__,
        "wall_s": wall,
    }
    (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(steps)} checkpoints in {wall:.1f} s on {summary['device_name']}")
    print(f"wrote {args.out / 'results.parquet'} and {args.out / 'run_summary.json'}")
    print(f"next: python examples/adamw_vs_sgd_reconstruction_evaluate.py --dir {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
