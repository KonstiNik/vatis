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

``--update-response`` adds a second eval batch and an explicit update rule
(:func:`update_response_setup`), so the run also produces the
``loss_change_first_order*`` rows and — the rule supplies a second set of weights — the
realized-change rows. Those involve no probes, so they must agree with a single
process as they are; what the test sees is whether the dot products use the
all-reduced gradients and whether the two loss evaluations are all-reduced.

``--gated`` (with ``--update-response``) uses :class:`GatedTiny` and marks one
sequence of the update batch (:func:`gate_batch`): the model then has a
parameter that is in the graph on one rank's shard only. Whether the ranks agree
that it received a gradient is what the all-reduce of that record is for.

``--adamw`` (with ``--preconditioner``) uses :func:`adamw_preconditioner`
instead: a ``P`` built by ``Preconditioner.adamw`` from a seeded second moment
with zeros in it, so the rows carry ``preconditioner_eps`` /
``preconditioner_beta2`` / ``preconditioner_capped_fraction`` as well.
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
from vatis.core.update_response import AdamHyper, OptimizerUpdate  # noqa: E402
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


# A sequence that starts with this token switches the gate of :class:`GatedTiny` on.
GATE_TOKEN = 7


class GatedTiny(torch.nn.Module):
    """The tiny transformer plus one parameter that only some data reaches.

    ``gate`` scales the logits of the sequences that start with ``GATE_TOKEN``.
    A batch without such a sequence leaves it out of the graph altogether:
    autograd then returns ``None`` for it, not zeros.
    """

    def __init__(self) -> None:
        super().__init__()
        self.inner = TinyTransformer()
        self.gate = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.inner(input_ids)
        hit = input_ids[:, 0] == GATE_TOKEN
        if bool(hit.any()):
            logits = logits * (1.0 + self.gate * hit.to(logits.dtype)[:, None, None])
        return logits


def gate_batch(batch: dict[str, torch.Tensor], rows: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """A copy of ``batch`` in which exactly the sequences ``rows`` start with ``GATE_TOKEN``.

    The first input token is no target of the shifted loss, so the labels stay.
    """
    ids = batch["input_ids"].clone()
    first = ids[:, 0]
    first[first == GATE_TOKEN] = GATE_TOKEN + 1
    first[list(rows)] = GATE_TOKEN
    return {**batch, "input_ids": ids}


def update_response_setup(
    bundle: ModelBundle, update_batch: dict[str, torch.Tensor]
) -> OptimizerUpdate:
    """An explicit AdamW update rule for ``bundle``, with a successor to step to.

    Seeded moments after three updates, weight decay on, and ``theta_{t+1}`` a
    seeded small displacement of the model's weights (it does not have to be the
    step the rule describes: the realized-change rows only compare two sets of
    weights). Shared by the worker and the single-process baseline of the driver
    test, like :func:`ceiling_preconditioner`.
    """
    gen = torch.Generator().manual_seed(11)
    params = bundle.params
    successor_model = type(bundle.model)().eval()
    with torch.no_grad():
        for new, old in zip(successor_model.parameters(), params, strict=True):
            new.copy_(old + 1e-3 * torch.randn(old.shape, generator=gen))
    successor = ModelBundle(
        model=successor_model,
        params=list(successor_model.parameters()),
        forward_fn=bundle.forward_fn,
        loss_fn=bundle.loss_fn,
        valid_mask_fn=bundle.valid_mask_fn,
        identifier="toy@step1",
    )
    return OptimizerUpdate(
        family="adam",
        hypers=[AdamHyper(lr=1e-2, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)]
        * len(params),
        step=3,
        first_moment=[1e-2 * torch.randn(p.shape, generator=gen) for p in params],
        second_moment=[1e-3 * torch.rand(p.shape, generator=gen) for p in params],
        # Low enough to engage: the clip factor then depends on the norm of the
        # WHOLE gradient, which a rank only has after the all-reduce.
        clip_grad_norm=0.5,
        descriptor="adamw_update",
        update_batch=update_batch,
        next_weights=lambda: successor,
    )


# The hyper-parameters of the ``--adamw`` P. ``eps`` is large on purpose: the
# coordinates with a zero second moment then carry a share of ``chi_net_P``
# rather than practically all of it.
ADAMW_EPS, ADAMW_BETA2, ADAMW_STEP = 0.05, 0.95, 10


def adamw_second_moment(n_params: int) -> torch.Tensor:
    """A seeded second moment with a tenth of its entries exactly zero."""
    gen = torch.Generator().manual_seed(5)
    exp_avg_sq = torch.exp(torch.randn(n_params, generator=gen) * 2.0)
    exp_avg_sq[torch.rand(n_params, generator=gen) < 0.1] = 0.0
    return exp_avg_sq


def adamw_preconditioner(n_params: int) -> Preconditioner:
    """``P = 1 / (sqrt(v_hat) + eps)`` from :func:`adamw_second_moment`.

    Shared by the worker and the single-process side of the driver test.
    """
    return Preconditioner.adamw(
        adamw_second_moment(n_params), step=ADAMW_STEP, beta2=ADAMW_BETA2, eps=ADAMW_EPS
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-hutchinson", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--chi-net-method", default=None)
    parser.add_argument("--hutchinson-distribution", default="rademacher")
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
        "--adamw",
        action="store_true",
        help="with --preconditioner: an AdamW P with zero second moments (eps/beta2 columns)",
    )
    parser.add_argument(
        "--duplicate-half",
        action="store_true",
        help="batch = two copies of a (batch-size // 2)-sample batch: identical shards on 2 ranks",
    )
    parser.add_argument(
        "--update-response",
        action="store_true",
        help="add a second eval batch and an explicit update rule (update-response rows)",
    )
    parser.add_argument(
        "--gated",
        action="store_true",
        help="with --update-response: a parameter that only rank 0's shard gives a gradient to",
    )
    args = parser.parse_args()

    if args.pin_probes:
        pinned = ShapeKeyedProbe(4242)
        _hutch_mod.make_probe = pinned  # type: ignore[assignment]
        _cv_mod.make_probe = pinned  # type: ignore[assignment]

    init_distributed()
    try:
        torch.manual_seed(args.seed)
        model = (GatedTiny() if args.gated else TinyTransformer()).eval()
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
            elif args.adamw:
                precond = adamw_preconditioner(n_params)
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
        eval_batches = {"val": batch}
        update = None
        if args.update_response:
            eval_batches["other"] = make_tiny_lm_batch(
                batch_size=args.batch_size, seed=args.seed + 1, pad_fraction=0.25, device="cpu"
            )
            if args.gated:
                # Sequence 0 is in rank 0's shard; no other sequence has the token.
                batch = eval_batches["val"] = gate_batch(batch, (0,))
                eval_batches["other"] = gate_batch(eval_batches["other"], ())
            update = update_response_setup(bundle, batch)
        results = analyze(
            model=bundle,
            revisions=["step0"],
            eval_batches=eval_batches,
            preconditioner=precond,
            update_response=None if update is None else (lambda ckpt_id: update),
            chi_net_method=args.chi_net_method,
            hutchinson_distribution=args.hutchinson_distribution,
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
