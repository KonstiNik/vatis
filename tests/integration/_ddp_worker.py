"""Worker script invoked under torchrun by ``test_ddp_loopback``.

Each rank initializes a process group, runs the analyzer on a fixed shared
batch, and writes its result row dict to a JSON file at the path passed via
``--out``. The driver test reads back rank 0's file and compares against the
single-GPU baseline.

``--pin-probes`` swaps ``make_probe`` for :class:`ShapeKeyedProbe`, which makes
``chi_net`` a *deterministic functional of the shard*. That is what lets the
driver compare DDP against single-process **exactly** instead of within a
sampling bound: with the shipped RNG the two runs draw different probes for
different samples (a rank's generator restarts at the same seed for its own
shard), so only a statistical comparison is possible and it hides plumbing bugs
of a few percent. Unbiasedness is tested elsewhere
(``tests/unit/test_chi_net_probe_exactness.py``); here we are testing sharding
and the all-reduce.
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

import vatis.core.chi_net.hutchinson as _hutch_mod  # noqa: E402
import vatis.core.chi_net.per_seq_cv as _cv_mod  # noqa: E402
from tests.fixtures.tiny_transformer import (  # noqa: E402
    TinyTransformer,
    causal_lm_loss,
    causal_lm_valid_mask,
    make_tiny_lm_batch,
)
from vatis import analyze  # noqa: E402
from vatis.core.preconditioner import Preconditioner  # noqa: E402
from vatis.distributed.ddp import (  # noqa: E402
    get_rank,
    init_distributed,
    shutdown_distributed,
)
from vatis.models.hf import ModelBundle  # noqa: E402


class ShapeKeyedProbe:
    """A probe determined entirely by its shape — the same vector at every call.

    Deliberately *not* a fresh draw. Keying on a call counter would not work: a
    rank's counter restarts while the single-process counter keeps going, so the
    two runs would apply different probes to the same samples. Keying on shape
    alone makes each micro-batch's contribution identical in both runs, hence the
    sum over shards is identical too.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __call__(
        self,
        shape: tuple[int, ...],
        *,
        distribution: str = "rademacher",  # noqa: ARG002
        device: object = None,
        dtype: torch.dtype = torch.float32,
        generator: object = None,  # noqa: ARG002
    ) -> torch.Tensor:
        key = tuple(int(d) for d in shape)
        gen = torch.Generator().manual_seed(self.seed + 7919 * len(key) + sum(key))
        raw = torch.randint(0, 2, key, generator=gen, dtype=torch.int64)
        return (raw.to(dtype=torch.float64) * 2.0 - 1.0).to(device=device, dtype=dtype)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-hutchinson", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--chi-net-method", default=None)
    parser.add_argument(
        "--pin-probes",
        action="store_true",
        help="replace make_probe with a shape-keyed constant (exact DDP comparison)",
    )
    parser.add_argument(
        "--preconditioner",
        action="store_true",
        help="also measure the P triple, with an explicit spread P",
    )
    args = parser.parse_args()

    if args.pin_probes:
        pinned = ShapeKeyedProbe(4242)
        _hutch_mod.make_probe = pinned  # type: ignore[assignment]
        _cv_mod.make_probe = pinned  # type: ignore[assignment]

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

        precond = None
        if args.preconditioner:
            n_params = sum(p.numel() for p in model.parameters())
            gen = torch.Generator().manual_seed(5)
            diag = torch.exp(torch.randn(n_params, generator=gen) * 2.0)
            precond = Preconditioner.from_diagonal(diag, descriptor="explicit")

        batch = make_tiny_lm_batch(
            batch_size=args.batch_size, seed=args.seed, pad_fraction=0.25, device="cpu"
        )
        results = analyze(
            model=bundle,
            revisions=["step0"],
            eval_batches={"val": batch},
            preconditioner=precond,
            chi_net_method=args.chi_net_method,
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
