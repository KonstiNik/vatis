"""Which update rule reproduces the step an AdamW run took? Step 1 of 2: measure.

Runs vatis over a few checkpoints of a native bundle and writes
``results.parquet`` and ``run_summary.json``; the evaluate script next to this
one turns them into the tables and the figure. The question, the definitions
and the answer are in ``adamw_vs_sgd_reconstruction/README.md``.

    python examples/adamw_vs_sgd_reconstruction.py --bundle <run>/bundle \\
        --held-out <run>/eval_batches/eval_batch_2.npy --clip-grad-norm 1.0 \\
        --steps 0 1 2 5 9 661 665 --device cuda

Every step needs its successor in the bundle, stored as fp32: the comparison is
against the difference of two stored weight vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from two_rulers_llama import llama_factory, lm_batch_from_tokens  # noqa: E402

from vatis.analyzer import Analyzer  # noqa: E402
from vatis.models import load_local_bundle  # noqa: E402

DEFAULT_OUT = Path(__file__).parent / "adamw_vs_sgd_reconstruction"
UPDATE_BATCH = "update_batch"  # the batch the optimizer step was computed on
HELD_OUT = "held_out"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True, help="bundle run directory")
    parser.add_argument("--held-out", required=True, help=".npy of [B, seq_len] token ids")
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument(
        "--clip-grad-norm",
        default="metadata",
        help="max gradient norm of the run, 'none' if it did not clip, or 'metadata' "
        "to read it from the bundle (which must then record value and algorithm)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--model-factory", default=None, help="'module:function'")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    choice = args.clip_grad_norm
    clip = "metadata" if choice == "metadata" else None if choice == "none" else float(choice)
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. The bundle: weights, saved optimizer state and update batch per step.
    source = load_local_bundle(
        args.bundle,
        args.model_factory or llama_factory,
        steps=sorted(args.steps),
        dtype="fp32",
        device=args.device,
    )
    # A held-out batch, scored with the run's own pad convention.
    ids = torch.from_numpy(np.load(args.held_out)).to(dtype=torch.int64, device=args.device)
    held_out = lm_batch_from_tokens(ids, source.loss_ignore_index)

    # 2. The analysis.
    analyzer = Analyzer(
        eval_batches={UPDATE_BATCH: source.batch_a_spec(), HELD_OUT: held_out},
        # The step that was taken: the loss_change* rows, update_lr, update_clip_factor.
        update_response=source.update_response(clip_grad_norm=clip),
        # delta_loss_P, with the preconditioner the optimizer had before the step.
        preconditioner=source.preconditioner(),
        # delta_loss(update batch, held-out batch): the SGD rule on the held-out batch.
        cross_pairs=[(UPDATE_BATCH, HELD_OUT)],
        chi_net_method="hutchinson",
        n_hutchinson=1,  # chi_net is not used here; one probe keeps it cheap
        micro_batch_size=args.micro_batch_size,
        sink=str(args.out / "results.parquet"),
        seed=0,
        device=args.device,
        dtype="fp32",
    )

    # 3. Run it: one set of rows per checkpoint, written to the parquet file.
    started = time.perf_counter()
    results = analyzer.run(source.model_specs())
    wall = time.perf_counter() - started

    for result in results:
        if not any(row.observable == "loss_change_first_order_weights" for row in result.rows):
            raise SystemExit(
                f"{result.revision}: no loss_change_first_order_weights row. The step and its "
                "successor must both be in the bundle with fp32 weight files."
            )
    summary = {
        "bundle": Path(args.bundle).resolve().parent.name,
        "optimizer": source.metadata.optimizer_name,
        "n_parameters": source.layout.total_numel,
        "clip_grad_norm": choice,
        "update_batch_shape": list(source.checkpoints[0].batch_a_tokens.shape),
        "held_out_shape": list(ids.shape),
        "device": torch.cuda.get_device_name(args.device) if "cuda" in args.device else "cpu",
        "wall_s": round(wall, 1),
    }
    (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(results)} checkpoints in {wall:.1f} s; wrote {args.out}/results.parquet")
    print(f"next: python examples/adamw_vs_sgd_reconstruction_evaluate.py --dir {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
