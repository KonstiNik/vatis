"""Integration test for the HF loader path.

Uses ``SimpleStories/SimpleStories-1.25M`` (Llama architecture, ~1.25M params,
already cached on the dev box). This is the tiniest published causal LM I
could find — Pythia's smallest is 14M, which is also fine but slower to load.

Gated behind ``-m integration`` because it touches the HF hub cache.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from vatis import analyze
from vatis.models.hf import load_hf_model

pytestmark = pytest.mark.integration


_MODEL = "SimpleStories/SimpleStories-1.25M"


def _make_batch(seq_len: int = 16, batch_size: int = 2) -> dict[str, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(0)
    # Use a vocab range smaller than the model's actual vocab so we don't
    # accidentally hit special tokens.
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), generator=g)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[:, :-1] = input_ids[:, 1:]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def test_load_hf_model_smoke() -> None:
    bundle = load_hf_model(_MODEL, dtype="fp32", device="cpu")
    assert bundle.model is not None
    assert len(bundle.params) > 0
    n = sum(p.numel() for p in bundle.params)
    assert 100_000 < n < 5_000_000, f"unexpected param count {n}"


def test_analyze_real_hf_model_runs(tmp_path: Path) -> None:
    bundle = load_hf_model(_MODEL, dtype="fp32", device="cpu")
    batch = _make_batch(seq_len=16, batch_size=2)
    out = tmp_path / "results.parquet"
    results = analyze(
        model=bundle,
        revisions=["main"],
        eval_batches={"val": batch},
        n_hutchinson=4,
        chi_net_method="per_sequence_cv",  # cheaper for B=2
        micro_batch_size=2,
        sink=str(out),
    )
    assert len(results) == 1
    rows = results[0].rows
    assert len(rows) == 6  # all observables
    # All values are finite.
    for r in rows:
        assert r.value == r.value
        assert -1e30 < r.value < 1e30
    table = pq.read_table(out)
    assert table.num_rows == 6


def test_analyze_real_hf_model_both_methods(tmp_path: Path) -> None:
    """Both estimators should produce CLOSE chi_net values on a real model.

    We use a moderate n_hutchinson and assert agreement within 30% relative
    error — this is loose because (a) the test runs on CPU with a small batch,
    so variance is high, and (b) we're not trying to verify exact convergence
    here, just that both code paths execute on a real LM.
    """
    bundle = load_hf_model(_MODEL, dtype="fp32", device="cpu")
    batch = _make_batch(seq_len=16, batch_size=2)

    r_h = analyze(
        model=bundle,
        revisions=["main"],
        eval_batches={"val": batch},
        n_hutchinson=64,
        chi_net_method="hutchinson",
        micro_batch_size=2,
        sink=None,
        seed=42,
    )
    r_cv = analyze(
        model=bundle,
        revisions=["main"],
        eval_batches={"val": batch},
        n_hutchinson=16,
        chi_net_method="per_sequence_cv",
        micro_batch_size=2,
        sink=None,
        seed=42,
    )
    cn_h = next(r.value for r in r_h[0].rows if r.observable == "chi_net")
    cn_cv = next(r.value for r in r_cv[0].rows if r.observable == "chi_net")
    rel = abs(cn_h - cn_cv) / max(1e-12, abs(cn_h))
    assert rel < 0.30, f"hutch={cn_h} vs cv={cn_cv} (rel err {rel:.3f})"
