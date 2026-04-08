"""Worker script invoked under torchrun by ``test_ddp_loopback``.

Each rank initializes a process group, runs the analyzer on a fixed shared
batch, and writes its result row dict to a JSON file at the path passed via
``--out``. The driver test reads back rank 0's file and compares against the
single-GPU baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Make the project root importable so ``tests.fixtures.tiny_transformer``
# resolves regardless of the working directory.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.fixtures.tiny_transformer import (  # noqa: E402
    TinyTransformer,
    causal_lm_loss,
    causal_lm_valid_mask,
    make_tiny_lm_batch,
)
from vatis import analyze  # noqa: E402
from vatis.distributed.ddp import (  # noqa: E402
    get_rank,
    init_distributed,
    shutdown_distributed,
)
from vatis.models.hf import ModelBundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-hutchinson", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    args = parser.parse_args()

    init_distributed()
    try:
        torch.manual_seed(args.seed)
        model = TinyTransformer().eval()
        for p in model.parameters():
            p.requires_grad_(True)
        bundle = ModelBundle(
            model=model,
            params=list(model.parameters()),
            forward_fn=lambda m, b: m(b["input_ids"]),
            loss_fn=causal_lm_loss,
            valid_mask_fn=causal_lm_valid_mask,
            identifier="toy@step0",
        )

        batch = make_tiny_lm_batch(batch_size=args.batch_size, seed=args.seed, device="cpu")
        results = analyze(
            model=bundle,
            revisions=["step0"],
            eval_batches={"val": batch},
            n_hutchinson=args.n_hutchinson,
            micro_batch_size=args.micro_batch_size,
            sink=None,
            device="cpu",
        )
        if get_rank() == 0:
            rows = [r.to_dict() for r in results[0].rows]
            Path(args.out).write_text(json.dumps(rows))
    finally:
        shutdown_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
