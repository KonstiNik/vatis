"""End-to-end unit tests for the single-GPU Analyzer + ParquetSink path."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
import torch.nn.functional as F

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
from vatis.analyzer import ALL_OBSERVABLES, Analyzer, _should_offload_cross_grads
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


def test_analyzer_chi_loss_uses_intersection_of_vmask_and_labels() -> None:
    """Regression test for the valid_mask_fn / chi_loss disagreement bug.

    When the bundle's ``valid_mask_fn`` disagrees with ``labels != -100``
    the chi_loss numerator and denominator must agree on the *intersection*
    of the two masks. Before the fix, the numerator used ``labels != -100``
    (via ``chi_loss_cross_entropy_unnormalized``'s internal mask) while the
    denominator used ``vmask.sum()``, producing wrong chi_loss whenever the
    two masks differed.

    Reproducer: an MLP bundle with all-True ``valid_mask_fn`` (the standard
    ``mlp_valid_mask`` from the toy fixtures) plus a batch where 3 of 8
    labels are ``-100``. The two masks disagree by construction:
    vmask claims 8 valid, labels say 5 valid. The fix takes the intersection
    (5) for both numerator and denominator.
    """
    torch.manual_seed(0)
    model = TinyMLP(in_dim=8, hidden=16, n_classes=4).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    x, y = make_tiny_mlp_batch(batch_size=8, in_dim=8, n_classes=4, seed=0)
    # Mark 3 of 8 samples as ignored.
    y_padded = y.clone()
    for i in (1, 3, 5):
        y_padded[i] = -100
    n_valid_truth = int((y_padded != -100).sum().item())
    assert n_valid_truth == 5  # sanity

    # mlp_loss does not pass ignore_index; we need a loss_fn that does so the
    # delta_loss path can run on the padded batch without crashing on -100.
    def loss_with_ignore(
        logits: torch.Tensor, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return F.cross_entropy(logits, batch[1], reduction="mean", ignore_index=-100)

    bundle = ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=lambda m, b: m(b[0]),
        loss_fn=loss_with_ignore,
        # mlp_valid_mask returns all-True; this is the disagreement source.
        valid_mask_fn=mlp_valid_mask,
        identifier="mlp@step0",
    )

    results = analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"clf": (x, y_padded)},
        n_hutchinson=4,
        micro_batch_size=8,
        sink=None,
    )
    rows = results[0].rows
    cl = next(r.value for r in rows if r.observable == "chi_loss")
    n_valid_reported = next(r.n_valid_a for r in rows if r.observable == "chi_loss")

    # vatis must use the intersection: 5 valid positions, not 8.
    assert n_valid_reported == n_valid_truth, (
        f"n_valid should be the intersection (5), got {n_valid_reported}"
    )

    # Manual chi_loss matching the user's valid_mask_fn intersected with
    # labels != -100: sum of (softmax-onehot)^2 over the 5 valid positions,
    # divided by 5^2.
    with torch.no_grad():
        logits = model(x)
    probs = F.softmax(logits.to(dtype=torch.float32), dim=-1)
    safe_y = y_padded.clone()
    safe_y[safe_y == -100] = 0  # avoid one_hot crash; masked out below
    onehot = F.one_hot(safe_y, num_classes=4).to(dtype=torch.float32)
    diff = probs - onehot
    mask_f = (y_padded != -100).to(dtype=torch.float32).unsqueeze(-1)
    masked_sq_sum = float(((diff * mask_f) ** 2).sum())
    expected_chi_loss = masked_sq_sum / (n_valid_truth**2)

    assert cl == pytest.approx(expected_chi_loss, rel=1e-5, abs=1e-7), (
        f"chi_loss = {cl}, expected {expected_chi_loss} "
        f"(numerator sum = {masked_sq_sum}, n_valid = {n_valid_truth})"
    )


def test_analyzer_invalid_observable_raises() -> None:
    with pytest.raises(ValueError, match="unknown observables"):
        Analyzer(
            eval_batches={"v": torch.zeros(2, 3)},
            observables=["chi_bogus"],
        )


def test_analyzer_threads_revision_through_self_pair_rows() -> None:
    """Self-pair rows should carry the revision label, not an empty string.

    Latent bug discovered while writing ``examples/pythia_sweep.py``: the
    self-pair rows used to hardcode ``revision=""`` instead of threading
    through the value the analyzer received. The fix passes ``revision``
    from ``_run_one_checkpoint`` into ``_compute_self_pair`` and onward
    into ``_emit_rows``.
    """
    bundle = _build_lm_bundle()
    batch = make_tiny_lm_batch(batch_size=2, seed=0)
    results = analyze(
        model=bundle,
        revisions=["my_revision_label"],
        eval_batches={"v": batch},
        n_hutchinson=2,
        micro_batch_size=2,
        sink=None,
    )
    rows = results[0].rows
    assert all(r.revision == "my_revision_label" for r in rows), (
        f"expected all rows to carry revision='my_revision_label', got "
        f"{sorted({r.revision for r in rows})}"
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


def test_cross_grad_storage_gpu_cpu_equivalent() -> None:
    """Cross-pair observables must match whether the cached gradients are held
    on-device (``"gpu"``) or offloaded to host RAM (``"cpu"``).

    The offload is a value-preserving device copy plus a device-agnostic
    ``_dot_fp64``, so forcing either storage on the same seeded run must agree.
    (On a CPU model both keep the gradient on the CPU, so this guards the
    plumbing — the resolver honouring the override, the ``.to('cpu')`` cache
    write, and the dot — rather than an actual cross-device transfer.)
    """
    batch_a = make_tiny_lm_batch(batch_size=4, seed=0)
    batch_b = make_tiny_lm_batch(batch_size=4, seed=1)

    def run(storage: str) -> dict[tuple[str, str, str], float]:
        results = analyze(
            model=_build_lm_bundle(),  # fresh model, fixed seed → identical weights
            revisions=["step0"],
            eval_batches={"a": batch_a, "b": batch_b},
            cross_pairs=[("a", "b")],
            cross_grad_storage=storage,
            n_hutchinson=8,
            micro_batch_size=2,
            seed=0,
            sink=None,
        )
        return {(r.batch_a, r.batch_b, r.observable): r.value for r in results[0].rows}

    gpu = run("gpu")
    cpu = run("cpu")
    assert gpu.keys() == cpu.keys()
    for key in gpu:
        assert gpu[key] == pytest.approx(cpu[key], rel=1e-12, abs=1e-12), key
    # And the cross pair was actually emitted (not silently skipped).
    assert ("a", "b", "delta_loss") in gpu


def test_invalid_cross_grad_storage_raises() -> None:
    with pytest.raises(ValueError, match=r"cross_grad_storage must be"):
        Analyzer(eval_batches={"v": make_tiny_lm_batch(batch_size=2)}, cross_grad_storage="disk")


def test_should_offload_cross_grads_heuristic() -> None:
    """The auto offload decision: large models offload, small stay on-GPU."""
    gib93 = 99_900_000_000  # ~93 GiB H100
    # 7B bf16, two cross batches → one self-pair (~weights + 8·P) plus the other
    # batch's fp32 gradient (4·P) overflows the card → offload.
    assert _should_offload_cross_grads(
        n_params=7_300_000_000,
        weights_bytes=14_600_000_000,
        n_cross_batches=2,
        total_device_bytes=gib93,
    )
    # A 160M model on the same card fits comfortably → stay on-GPU.
    assert not _should_offload_cross_grads(
        n_params=160_000_000,
        weights_bytes=320_000_000,
        n_cross_batches=2,
        total_device_bytes=gib93,
    )
    # Fewer than two cross batches → nothing to offload, regardless of size.
    assert not _should_offload_cross_grads(
        n_params=7_300_000_000,
        weights_bytes=14_600_000_000,
        n_cross_batches=1,
        total_device_bytes=gib93,
    )
    # Boundary: with P=1e9, weights=2e9, n=2 the footprint is weights + 12·P =
    # 14e9 bytes, compared against 0.85·total. The decision must flip right
    # around total = 14e9 / 0.85 ≈ 16.47e9 — this catches a wrong factor that
    # the 45× large-vs-small gap above would not.
    common = dict(n_params=1_000_000_000, weights_bytes=2_000_000_000, n_cross_batches=2)
    assert _should_offload_cross_grads(**common, total_device_bytes=16_000_000_000)  # 14 > 13.6
    assert not _should_offload_cross_grads(
        **common, total_device_bytes=17_000_000_000
    )  # 14 < 14.45


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real GPU→CPU offload needs CUDA")
def test_cross_grad_storage_offload_equivalent_on_cuda() -> None:
    """On a CUDA device, the cross observables must be identical whether the
    cached gradients stay on the GPU (``"gpu"``) or are offloaded to host RAM
    (``"cpu"``) — a genuine device-boundary copy plus a CPU-side dot. The
    CPU-model test above cannot exercise this (both paths keep the gradient on
    the CPU); this is the one that actually validates the offload.
    """
    batch_a = make_tiny_lm_batch(batch_size=4, seed=0)
    batch_b = make_tiny_lm_batch(batch_size=4, seed=1)

    def run(storage: str) -> dict[tuple[str, str, str], float]:
        torch.manual_seed(0)  # identical weights each call
        model = TinyTransformer().eval().cuda()
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
        results = analyze(
            model=bundle,
            revisions=["step0"],
            eval_batches={"a": batch_a, "b": batch_b},
            cross_pairs=[("a", "b")],
            cross_grad_storage=storage,
            n_hutchinson=8,
            micro_batch_size=2,
            seed=0,
            device="cuda",
            sink=None,
        )
        return {(r.batch_a, r.batch_b, r.observable): r.value for r in results[0].rows}

    gpu = run("gpu")
    cpu = run("cpu")
    assert gpu.keys() == cpu.keys()
    for key in gpu:
        assert gpu[key] == pytest.approx(cpu[key], rel=1e-6, abs=1e-6), key
