"""DDP ranks must not replay one Hutchinson probe stream — and rank 0 must keep it.

**The defect these tests pin.** Every rank used to seed its generator with the
*same* per-call seed and then draw the probes for its own shard, so the ``k``-th
sample of every shard was probed with the same random vectors. A Rademacher
probe's error on a sample is ``sum_{i != j} v_i v_j M_ij`` with
``M = J J^T`` the sample's output-space Gram matrix, so two samples probed with
the same ``v`` have error covariance ``2 <offdiag(M_a), offdiag(M_b)>_F``. That
is far from zero: the Gram matrices of different inputs share much of their
off-diagonal structure (the same unembedding, the same input-independent
parameters). The per-rank errors therefore added almost coherently and the
variance of the all-reduced ``chi_net`` grew by up to the world size — the
extreme, exactly ``R``, being reached when the shards hold the same samples.

The tests below emulate the per-rank calls in one process by patching the rank
and world size the analyzer sees. ``is_distributed()`` stays ``False``, so the
analyzer skips the all-reduce and the ``chi_net`` row it returns is that rank's
*partial* sum — exactly the number a real rank hands to the all-reduce. The real
``torchrun`` path is covered in ``tests/integration/test_ddp_loopback.py``.

**The other half of the contract.** Rank 0 — and therefore every single-process
run — keeps the stream named by the ``hutchinson_seed`` column, so single-process
results for a given ``seed`` are unchanged by the rank-distinct streams.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import vatis.analyzer as analyzer_mod
from tests.fixtures.tiny_transformer import (
    TinyTransformer,
    causal_lm_loss,
    causal_lm_valid_mask,
    make_tiny_lm_batch,
)
from vatis import analyze
from vatis.core.chi_net import build_estimator
from vatis.core.preconditioner import Preconditioner
from vatis.core.probes import make_probe
from vatis.distributed.sharding import shard_batch
from vatis.models.hf import ModelBundle

_VOCAB, _SEQ = 32, 8


def _bundle() -> ModelBundle:
    """A small transformer in which **every** parameter is live.

    ``TinyTransformer`` zero-initialises all 1-D parameters, LayerNorm gains
    included, so at init its logits are identically zero and only the final
    LayerNorm receives a gradient. The statistical test below must not rest on
    that degenerate Jacobian, so the gains are set to 1 and the weight matrices
    to an ordinary ``N(0, 1/fan_in)`` scale.
    """
    torch.manual_seed(0)
    model = TinyTransformer(
        vocab_size=_VOCAB, seq_len=_SEQ, d_model=32, n_heads=2, n_layers=1, d_ff=64
    ).eval()
    gen = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "ln" in name and name.endswith("weight"):
                p.fill_(1.0)
            elif p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=gen) / p.shape[-1] ** 0.5)
    return ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=lambda m, b: m(b["input_ids"]),
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        identifier="toy@step0",
    )


def _batch(batch_size: int, seed: int = 3) -> dict[str, torch.Tensor]:
    return make_tiny_lm_batch(batch_size=batch_size, seq_len=_SEQ, vocab_size=_VOCAB, seed=seed)


def _explicit_p(bundle: ModelBundle) -> Preconditioner:
    n_params = sum(p.numel() for p in bundle.params)
    gen = torch.Generator().manual_seed(5)
    return Preconditioner.from_diagonal(
        torch.exp(torch.randn(n_params, generator=gen) * 2.0), descriptor="explicit"
    )


def _rank_partials(
    monkeypatch: pytest.MonkeyPatch,
    bundle: ModelBundle,
    batch: dict[str, torch.Tensor],
    *,
    world_size: int,
    **analyze_kwargs: Any,
) -> list[dict[str, float]]:
    """Per rank, the observables that rank computes on its shard (see module docstring)."""
    partials = []
    for rank in range(world_size):
        with monkeypatch.context() as patch:
            patch.setattr(analyzer_mod, "get_rank", lambda rank=rank: rank)
            patch.setattr(analyzer_mod, "get_world_size", lambda: world_size)
            results = analyze(
                model=bundle,
                revisions=["step0"],
                eval_batches={"val": batch},
                sink=None,
                **analyze_kwargs,
            )
        partials.append({r.observable: r.value for r in results[0].rows})
    return partials


# --------------------------------------------------------------------------- the defect


@pytest.mark.parametrize("distribution", ["rademacher", "gaussian"])
@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_ranks_holding_the_same_samples_draw_different_probes(
    monkeypatch: pytest.MonkeyPatch, method: str, distribution: str
) -> None:
    """Deterministic form of the defect: identical shards used to give identical partials.

    With the batch ``[a, b, a, b]`` on two ranks both shards are ``[a, b]``. A rank
    that restarts the same stream computes bit-for-bit what rank 0 computed, so the
    second rank adds no information: the "2-rank" estimate is rank 0's, doubled,
    with twice the variance per sample of a real 4-sample estimate.
    """
    bundle = _bundle()
    half = _batch(2)
    batch = {k: torch.cat([v, v]) for k, v in half.items()}
    p0, p1 = _rank_partials(
        monkeypatch,
        bundle,
        batch,
        world_size=2,
        seed=0,
        chi_net_method=method,
        hutchinson_distribution=distribution,
        n_hutchinson=4,
        micro_batch_size=1,
        preconditioner=_explicit_p(bundle),
    )
    # Same samples, so everything that does not involve a probe agrees exactly ...
    assert p0["chi_loss"] == p1["chi_loss"]
    assert p0["delta_loss"] == p1["delta_loss"]
    # ... and the probe-dependent traces must NOT (they are sampled independently).
    for observable in ("chi_net", "chi_net_P"):
        rel = abs(p0[observable] - p1[observable]) / abs(p0[observable])
        assert rel > 1e-4, (
            f"{observable}: rank 0 and rank 1 returned {p0[observable]!r} and "
            f"{p1[observable]!r} for the same samples -- the ranks are replaying one probe stream"
        )


@pytest.mark.slow
@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_summed_rank_estimates_are_not_positively_correlated(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Statistical form: ``Var(sum_r x_r) / sum_r Var(x_r)`` must be ~1, not ~world size.

    ``x_r`` is rank ``r``'s partial ``chi_net``, one *different* sample per rank,
    over 96 analyzer seeds. Independent ranks give 1 in expectation, and the
    estimator of this ratio at 8 ranks and 96 seeds has a standard deviation of
    about ``(2/R) sqrt(C(R,2)/N) = 0.135``. Replayed streams gave 3.19
    (``hutchinson``) and 3.14 (``per_sequence_cv``) on exactly this configuration;
    per-rank streams give 1.11 for both. The bound of 2 is seven standard
    deviations above the independent value; everything is seeded, so the outcome
    is deterministic, not merely unlikely to flip.
    """
    world_size, n_seeds = 8, 96
    bundle = _bundle()
    batch = _batch(world_size)
    x = torch.empty((n_seeds, world_size), dtype=torch.float64)
    for seed in range(n_seeds):
        partials = _rank_partials(
            monkeypatch,
            bundle,
            batch,
            world_size=world_size,
            seed=seed,
            chi_net_method=method,
            n_hutchinson=2,
            micro_batch_size=1,
        )
        x[seed] = torch.tensor([p["chi_net"] for p in partials], dtype=torch.float64)
    ratio = float(x.sum(dim=1).var() / x.var(dim=0).sum())
    assert ratio < 2.0, (
        f"variance of the summed per-rank chi_net is {ratio:.2f}x the sum of the per-rank "
        f"variances at {world_size} ranks: the ranks' estimation errors are correlated"
    )
    # Not suspiciously anti-correlated either (that would also mean coupled streams).
    assert ratio > 0.5, ratio


# ------------------------------------------------- single-process results are unchanged


def test_per_call_seed_values_are_pinned() -> None:
    """The seed derivation is part of the reproducibility contract: literal values."""
    assert analyzer_mod._per_call_seed(0, "toy@step0", "val") == 73036076
    assert analyzer_mod._per_call_seed(7, "toy@step0", "val") == 1293874550
    assert analyzer_mod._per_call_seed(0, "org/model@step1000", "train") == 1018485237
    # ... and what that seed means as a probe (platform-independent integers).
    gen = torch.Generator().manual_seed(73036076)
    assert make_probe((2, 8), generator=gen).to(torch.int64).tolist() == [
        [1, -1, 1, -1, -1, -1, -1, 1],
        [-1, 1, -1, 1, -1, 1, -1, -1],
    ]


@pytest.mark.parametrize("distribution", ["rademacher", "gaussian"])
@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_single_process_chi_net_is_exactly_the_hutchinson_seed_stream(
    method: str, distribution: str
) -> None:
    """One process: ``chi_net`` is the estimator run on a generator seeded with the
    row's ``hutchinson_seed`` — bit for bit, and that seed is ``_per_call_seed``.

    Together with the literal pins above this fixes what a given ``seed`` means in
    a single-process run, independently of how other DDP ranks derive theirs.
    """
    bundle = _bundle()
    batch = _batch(5)
    precond = _explicit_p(bundle)
    results = analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"val": batch},
        preconditioner=precond,
        chi_net_method=method,
        hutchinson_distribution=distribution,
        n_hutchinson=3,
        micro_batch_size=2,
        sink=None,
        seed=7,
    )
    rows = {r.observable: r for r in results[0].rows}
    assert rows["chi_net"].hutchinson_seed == analyzer_mod._per_call_seed(7, "toy@step0", "val")
    assert rows["chi_net"].hutchinson_seed == 1293874550

    estimator = build_estimator(method, n_hutchinson=3, distribution=distribution)
    direct = estimator.compute(
        bundle.model,
        batch,
        bundle.forward_fn,
        loss_fn=bundle.loss_fn,
        valid_mask_fn=causal_lm_valid_mask,
        params=bundle.params,
        micro_batch_size=2,
        generator=torch.Generator().manual_seed(1293874550),
        preconditioner=precond,
    )
    assert direct.chi_net_p is not None
    assert rows["chi_net"].value == float(direct.chi_net.to(dtype=torch.float64))
    assert rows["chi_net_P"].value == float(direct.chi_net_p.to(dtype=torch.float64))


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_rank_zero_keeps_the_single_process_stream(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Rank 0 of a multi-rank run draws exactly what a single process draws for the
    same samples and seed, whatever the world size."""
    bundle = _bundle()
    batch = _batch(6)
    kwargs: dict[str, Any] = {
        "seed": 11,
        "chi_net_method": method,
        "n_hutchinson": 3,
        "micro_batch_size": 1,
    }
    for world_size in (2, 3):
        shard0, _, _ = shard_batch(batch, rank=0, world_size=world_size)
        (alone,) = _rank_partials(monkeypatch, bundle, shard0, world_size=1, **kwargs)
        rank0 = _rank_partials(monkeypatch, bundle, batch, world_size=world_size, **kwargs)[0]
        assert rank0["chi_net"] == alone["chi_net"], world_size


# ------------------------------------------------------------------ the seeding rule


def test_rank_probe_seed_rule() -> None:
    """Rank 0 -> the call seed itself; other ranks -> distinct, pinned, valid seeds."""
    rank_seed = analyzer_mod._rank_probe_seed
    for call_seed in (0, 1, 73036076, 2**31 - 2):
        assert rank_seed(call_seed, 0) == call_seed
        seeds = [rank_seed(call_seed, rank) for rank in range(64)]
        assert len(set(seeds)) == 64, "two ranks would share a stream"
        assert all(0 <= s < 2**31 - 1 for s in seeds)
        assert seeds == [rank_seed(call_seed, rank) for rank in range(64)], "not deterministic"
    # Literal values: a rank's stream is a documented function of
    # (hutchinson_seed, rank) -- sha256(f"{seed}|rank={rank}") -- and must not drift.
    assert [rank_seed(73036076, rank) for rank in (1, 2, 3)] == [
        1359491800,
        1714191631,
        1058022375,
    ]
    assert rank_seed(1293874550, 1) == 1468076391
    with pytest.raises(ValueError, match="rank"):
        rank_seed(0, -1)


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_each_rank_partial_is_the_estimator_on_its_documented_stream(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Reproducibility: seed, rank and world size determine a rank's contribution.

    Rank ``r`` computes the estimator on its shard with a generator seeded
    ``_rank_probe_seed(hutchinson_seed, r)``; nothing else enters.
    """
    world_size = 3
    bundle = _bundle()
    batch = _batch(7)  # uneven: shards of 3, 2, 2
    partials = _rank_partials(
        monkeypatch,
        bundle,
        batch,
        world_size=world_size,
        seed=0,
        chi_net_method=method,
        n_hutchinson=3,
        micro_batch_size=2,
    )
    estimator = build_estimator(method, n_hutchinson=3, distribution="rademacher")
    for rank in range(world_size):
        shard, _, _ = shard_batch(batch, rank=rank, world_size=world_size)
        gen = torch.Generator().manual_seed(analyzer_mod._rank_probe_seed(73036076, rank))
        direct = estimator.compute(
            bundle.model,
            shard,
            bundle.forward_fn,
            loss_fn=bundle.loss_fn,
            valid_mask_fn=causal_lm_valid_mask,
            params=bundle.params,
            micro_batch_size=2,
            generator=gen,
        )
        assert partials[rank]["chi_net"] == float(direct.chi_net.to(dtype=torch.float64)), rank
