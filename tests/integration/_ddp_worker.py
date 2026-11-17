"""Worker script invoked under torchrun by ``test_ddp_loopback``.

Each rank initializes a process group and runs the analyzer on a fixed shared
batch. Rank 0 writes its result rows to the JSON file passed via ``--out``; the
driver test reads that file back and compares against the single-GPU baseline.
Every other rank writes its (in-memory) rows to ``<out>.rank<r>``, for the tests
that check what each rank records.

``--pin-probes`` swaps ``make_probe`` for :class:`ShapeKeyedProbe`, which makes
``chi_net`` a *deterministic functional of the shard*. That is what lets the
driver compare DDP against single-process **exactly** instead of within a
sampling bound: with the shipped RNG the two runs draw different probes for
the same samples (rank 0 draws the per-call stream for its shard, every other
rank draws its own rank-derived stream, while a single process runs the per-call
stream through the whole batch), so DDP and single-process can only be compared
statistically. Unbiasedness is tested elsewhere
(``tests/unit/test_chi_net_probe_exactness.py``); here we are testing sharding
and the all-reduce.

``--duplicate-half`` builds the batch as two copies of a half-size batch, so the
two ranks of a 2-rank run hold the *same* samples. That is the configuration in
which ranks replaying one probe stream is visible without any statistics: the
second rank then reproduces the first rank's number bit for bit.

``--ceiling`` (with ``--preconditioner``) swaps the spread ``P`` for
:func:`ceiling_preconditioner`: a ``P`` that has a ceiling and entries sitting
at it, so the run also produces the two ``*_capped`` rows. Without it no DDP
test sees them — a ``P`` without a ceiling has no capped part to all-reduce.
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


# The ceiling of the ``--ceiling`` P; the band that counts as "at the ceiling" is
# ``>= CEILING * (1 - 1e-6)``. Chosen with the fraction below so that the capped
# part is of the same order as the rest of each P observable — a share near 0 or
# near 1 would let a wrong capped row hide behind the other part.
CEILING = 50.0


def ceiling_preconditioner(n_params: int) -> Preconditioner:
    """A spread explicit ``P`` with a ceiling and a seeded tenth of its entries at it.

    The remaining entries are capped at half the ceiling, so which coordinates
    are "at the ceiling" is unambiguous. The entries at the ceiling are spread
    over every parameter, hence every rank's shard contributes to the capped
    part of ``chi_net_P`` — which is what makes its all-reduce observable.
    Shared by the worker and the single-process baseline of the driver test.
    """
    gen = torch.Generator().manual_seed(5)
    diag = torch.exp(torch.randn(n_params, generator=gen) * 2.0).clamp_(max=0.5 * CEILING)
    at_ceiling = torch.rand(n_params, generator=gen) < 0.1
    diag[at_ceiling] = CEILING
    return Preconditioner.from_diagonal(diag, descriptor="explicit", ceiling=CEILING)


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
    parser.add_argument(
        "--ceiling",
        action="store_true",
        help="with --preconditioner: a P with a ceiling and entries at it (adds the capped rows)",
    )
    parser.add_argument(
        "--duplicate-half",
        action="store_true",
        help="batch = two copies of a (batch-size // 2)-sample batch: identical shards on 2 ranks",
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
            if args.ceiling:
                precond = ceiling_preconditioner(n_params)
            else:
                gen = torch.Generator().manual_seed(5)
                diag = torch.exp(torch.randn(n_params, generator=gen) * 2.0)
                precond = Preconditioner.from_diagonal(diag, descriptor="explicit")

        if args.duplicate_half:
            half = make_tiny_lm_batch(
                batch_size=args.batch_size // 2, seed=args.seed, pad_fraction=0.25, device="cpu"
            )
            batch = {k: torch.cat([v, v]) for k, v in half.items()}
        else:
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
        rows = [r.to_dict() for r in results[0].rows]
        if get_rank() == 0:
            Path(args.out).write_text(json.dumps(rows))
        else:
            # The in-memory rows of the other ranks, for the tests that check what
            # every rank records (the ``hutchinson_seed`` column). The values are
            # the same all-reduced totals rank 0 holds.
            Path(f"{args.out}.rank{get_rank()}").write_text(json.dumps(rows))
    finally:
        shutdown_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
