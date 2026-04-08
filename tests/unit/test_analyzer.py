"""End-to-end unit tests for the single-GPU Analyzer + ParquetSink path."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from tests.fixtures.tiny_transformer import (
    TinyMLP,
    TinyTransformer,
    causal_lm_loss,
    causal_lm_valid_mask,
    make_tiny_lm_batch,
    make_tiny_mlp_batch,
    mlp_loss,
    mlp_valid_mask,
)
from vatis import analyze
from vatis.analyzer import ALL_OBSERVABLES, Analyzer
from vatis.models.hf import ModelBundle
from vatis.sinks.parquet import ParquetSink


def _build_lm_bundle() -> ModelBundle:
    torch.manual_seed(0)
    model = TinyTransformer().eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=lambda m, b: m(b["input_ids"]),
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        identifier="toy@step0",
    )


def _build_mlp_bundle() -> ModelBundle:
    torch.manual_seed(0)
    model = TinyMLP().eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=lambda m, b: m(b[0]),
        loss_fn=mlp_loss,
        valid_mask_fn=mlp_valid_mask,
        identifier="mlp@step0",
    )


def test_analyzer_emits_all_observables_for_lm() -> None:
    bundle = _build_lm_bundle()
    batch = make_tiny_lm_batch(batch_size=4, seed=0)
    results = analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"val": batch},
        n_hutchinson=8,
        micro_batch_size=2,
        sink=None,
    )
    assert len(results) == 1
    rows = results[0].rows
    obs_set = {r.observable for r in rows}
    assert obs_set == set(ALL_OBSERVABLES)
    # All values are finite numbers.
    for r in rows:
        assert r.value == r.value  # not NaN
        assert -1e30 < r.value < 1e30


def test_analyzer_writes_parquet(tmp_path: Path) -> None:
    bundle = _build_lm_bundle()
    batch = make_tiny_lm_batch(batch_size=4, seed=0)
    out = tmp_path / "results.parquet"
    analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"val_a": batch, "val_b": batch},
        n_hutchinson=4,
        micro_batch_size=2,
        sink=str(out),
    )
    assert out.exists()
    table = pq.read_table(out)
    # 6 observables × 2 batches × 1 checkpoint = 12 rows.
    assert table.num_rows == 12
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    assert set(cols["batch_a"]) == {"val_a", "val_b"}
    assert set(cols["observable"]) == set(ALL_OBSERVABLES)
    assert all(c == "toy@step0" for c in cols["checkpoint_id"])


def test_analyzer_chi_loss_invariant_to_micro_batch_size() -> None:
    """chi_loss / delta_loss should not depend on how the batch is split."""
    bundle = _build_lm_bundle()
    batch = make_tiny_lm_batch(batch_size=4, seed=0)
    obs_to_check = ("chi_loss", "delta_loss", "chi_loss_normalized")
    values: dict[str, list[float]] = {o: [] for o in obs_to_check}
    for mb in (1, 2, 4):
        results = analyze(
            model=bundle,
            revisions=[f"step{mb}"],
            eval_batches={"val": batch},
            n_hutchinson=4,
            micro_batch_size=mb,
            sink=None,
        )
        rows = results[0].rows
        for o in obs_to_check:
            v = next(r.value for r in rows if r.observable == o)
            values[o].append(v)
    for o in obs_to_check:
        v0 = values[o][0]
        for v in values[o][1:]:
            assert abs(v - v0) <= max(1e-7, abs(v0) * 1e-5), (
                f"{o} not micro-batch invariant: {values[o]}"
            )


def test_analyzer_with_padding_uses_attention_mask() -> None:
    bundle = _build_lm_bundle()
    batch = make_tiny_lm_batch(batch_size=4, seed=0, pad_fraction=0.25)
    results = analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"val_pad": batch},
        n_hutchinson=4,
        micro_batch_size=2,
        sink=None,
    )
    rows = results[0].rows
    # The valid token count should be less than B*S.
    n_valid = next(r.n_valid_a for r in rows if r.observable == "chi_loss")
    assert 0 < n_valid < 4 * 16


def test_analyzer_works_for_mlp() -> None:
    bundle = _build_mlp_bundle()
    x, y = make_tiny_mlp_batch(batch_size=8, seed=0)
    batch = (x, y)
    # Use Analyzer directly so we can avoid the dict-based default labels.
    results = analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"clf": batch},
        n_hutchinson=4,
        micro_batch_size=4,
        sink=None,
    )
    rows = results[0].rows
    assert len(rows) == len(ALL_OBSERVABLES)
    cl = next(r.value for r in rows if r.observable == "chi_loss")
    cn = next(r.value for r in rows if r.observable == "chi_net")
    dl = next(r.value for r in rows if r.observable == "delta_loss")
    cp = next(r.value for r in rows if r.observable == "chi_pos")
    assert cl > 0
    assert cn > 0
    assert dl >= 0
    # chi_pos = delta_loss / (chi_loss * chi_net)
    expected_cp = dl / (cl * cn) if cl * cn > 0 else 0.0
    assert abs(cp - expected_cp) < 1e-6


def test_analyzer_invalid_observable_raises() -> None:
    with pytest.raises(ValueError, match="unknown observables"):
        Analyzer(
            eval_batches={"v": torch.zeros(2, 3)},
            observables=["chi_bogus"],
        )


def test_compute_cross_pair_before_self_pair_raises() -> None:
    """Calling _compute_cross_pair before the corresponding self pairs is a
    contract violation. The analyzer should fail loudly with a clear message
    pointing the caller at the contract, not silently fall back to zeros.
    """
    bundle = _build_lm_bundle()
    batch = make_tiny_lm_batch(batch_size=2, seed=0)
    analyzer = Analyzer(
        eval_batches={"a": batch, "b": batch},
        cross_pairs=[("a", "b")],
        n_hutchinson=2,
        micro_batch_size=2,
        sink=None,
    )
    # Caches are initialized but empty — calling _compute_cross_pair directly
    # without first running the self loop must raise.
    fake_grad = torch.zeros(sum(p.numel() for p in bundle.params))
    with pytest.raises(RuntimeError, match=r"self-pair cache is missing"):
        analyzer._compute_cross_pair(
            ckpt_id="ckpt0",
            revision="r0",
            name_a="a",
            name_b="b",
            g_a=fake_grad,
            g_b=fake_grad,
            n_valid_a=4,
            n_valid_b=4,
            bundle=bundle,
        )

    # After running the self loop, the same cross pair must succeed.
    # We exercise the public path so the caches get populated naturally.
    results = analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"a": batch, "b": batch},
        cross_pairs=[("a", "b")],
        n_hutchinson=2,
        micro_batch_size=2,
        sink=None,
    )
    rows = results[0].rows
    pair_rows = [r for r in rows if r.batch_a == "a" and r.batch_b == "b"]
    # All six observables should be emitted for the cross pair.
    assert {r.observable for r in pair_rows} == set(ALL_OBSERVABLES)


def test_parquet_sink_streaming_flush(tmp_path: Path) -> None:
    bundle = _build_lm_bundle()
    batch = make_tiny_lm_batch(batch_size=4, seed=0)
    out = tmp_path / "stream.parquet"
    sink = ParquetSink(out, flush_every=4)
    analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"v1": batch, "v2": batch},
        n_hutchinson=4,
        micro_batch_size=2,
        sink=sink,
    )
    table = pq.read_table(out)
    assert table.num_rows == 12
