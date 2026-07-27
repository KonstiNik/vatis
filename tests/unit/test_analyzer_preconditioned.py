"""Pipeline-level tests for the preconditioned ruler (WP0 component 03).

Where ``test_chi_pos_p.py`` tests the math against closed-form answers on fp64
toys, this file tests the *plumbing*: what the analyzer emits, what the parquet
contains, that raw rows are untouched, that cross pairs get their P counterparts,
and the pipeline half of **T-identity** (``P = I`` reproduces the raw columns
through the whole ``analyze()`` -> parquet path on a synthetic bundle, atol
1e-6).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from tests.fixtures.preconditioned_toys import all_valid_mask, ce_loss, tuple_forward
from tests.fixtures.synthetic_bundle import (
    BundleTinyLM,
    bundle_metadata,
    handwritten_optim_state,
    make_batch_a,
    tiny_lm_factory,
    write_checkpoint,
    write_run_files,
)
from tests.fixtures.tiny_transformer import (
    TinyMLP,
    make_tiny_mlp_batch,
)
from vatis import analyze
from vatis.analyzer import ALL_OBSERVABLES, P_OBSERVABLES, Analyzer
from vatis.core.preconditioner import RAW_DESCRIPTOR, Preconditioner
from vatis.models.hf import ModelBundle
from vatis.models.local_bundle import load_local_bundle

BETA1, BETA2, EPS = 0.9, 0.95, 1e-8


def _mlp_bundle(seed: int = 0) -> ModelBundle:
    torch.manual_seed(seed)
    model = TinyMLP().eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=tuple_forward,
        loss_fn=ce_loss,
        valid_mask_fn=all_valid_mask,
        identifier="mlp@step0",
    )


def _by_observable(rows: list, preconditioner: str | None = None) -> dict[str, float]:
    return {
        r.observable: r.value
        for r in rows
        if preconditioner is None or r.preconditioner == preconditioner
    }


# ============================================================ default is unchanged


def test_no_preconditioner_emits_only_raw_rows_tagged_none() -> None:
    """Backwards compatibility: an un-configured run is exactly the old run,
    with the new descriptor column reading "none" on every row."""
    bundle = _mlp_bundle()
    batch = make_tiny_mlp_batch(batch_size=6, seed=0)
    rows = analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"val": batch},
        n_hutchinson=4,
        micro_batch_size=3,
        sink=None,
    )[0].rows
    assert {r.observable for r in rows} == set(ALL_OBSERVABLES)
    assert {r.preconditioner for r in rows} == {RAW_DESCRIPTOR}


def test_preconditioner_does_not_move_any_raw_row() -> None:
    """Every raw row must be bit-identical with and without the P ruler."""
    batch = make_tiny_mlp_batch(batch_size=6, seed=0)

    def run(precond: object | None) -> dict[str, float]:
        bundle = _mlp_bundle()
        n_params = sum(p.numel() for p in bundle.params)
        rows = analyze(
            model=bundle,
            revisions=["step0"],
            eval_batches={"val": batch},
            preconditioner=(
                None
                if precond is None
                else Preconditioner.from_diagonal(
                    torch.rand(n_params, generator=torch.Generator().manual_seed(5)) * 3.0 + 0.1
                )
            ),
            n_hutchinson=4,
            micro_batch_size=3,
            sink=None,
        )[0].rows
        return _by_observable(rows, RAW_DESCRIPTOR)

    without = run(None)
    with_p = run(object())
    assert with_p == without, (
        "raw rows changed when the preconditioned ruler was switched on: "
        f"{ {k: (with_p[k], without[k]) for k in without if with_p[k] != without[k]} }"
    )


def test_requesting_p_observables_without_a_preconditioner_raises() -> None:
    with pytest.raises(ValueError, match="belong to the preconditioned ruler"):
        Analyzer(eval_batches={"v": make_tiny_mlp_batch()}, observables=["chi_pos_P"])


def test_unknown_preconditioner_shorthand_raises() -> None:
    with pytest.raises(ValueError, match="not a recognized shorthand"):
        Analyzer(eval_batches={"v": make_tiny_mlp_batch()}, preconditioner="adamw")


def test_misaligned_preconditioner_is_refused() -> None:
    """A P of the wrong length would silently weight the wrong coordinates."""
    bundle = _mlp_bundle()
    n_params = sum(p.numel() for p in bundle.params)
    with pytest.raises(ValueError, match="must be aligned with the canonical"):
        analyze(
            model=bundle,
            revisions=["step0"],
            eval_batches={"val": make_tiny_mlp_batch(batch_size=4, seed=0)},
            preconditioner=Preconditioner.identity(n_params - 1),
            n_hutchinson=2,
            sink=None,
        )


# ============================================================ T-identity (pipeline)


def test_identity_preconditioner_matches_raw_rows_in_memory() -> None:
    """T-identity through the analyzer, self and cross pairs."""
    bundle = _mlp_bundle()
    n_params = sum(p.numel() for p in bundle.params)
    batch_a = make_tiny_mlp_batch(batch_size=6, seed=0)
    batch_b = make_tiny_mlp_batch(batch_size=6, seed=1)
    rows = analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"a": batch_a, "b": batch_b},
        cross_pairs=[("a", "b")],
        preconditioner=Preconditioner.identity(n_params),
        n_hutchinson=8,
        micro_batch_size=3,
        sink=None,
    )[0].rows
    pairs = {(r.batch_a, r.batch_b) for r in rows}
    assert pairs == {("a", "a"), ("b", "b"), ("a", "b")}
    for pair in sorted(pairs):
        subset = [r for r in rows if (r.batch_a, r.batch_b) == pair]
        raw = _by_observable(subset, RAW_DESCRIPTOR)
        p_rows = _by_observable(subset, "identity")
        assert set(p_rows) == set(P_OBSERVABLES)
        for p_name, raw_name in (
            ("delta_loss_P", "delta_loss"),
            ("chi_net_P", "chi_net"),
            ("chi_pos_P", "chi_pos"),
        ):
            assert abs(p_rows[p_name] - raw[raw_name]) <= 1e-6, (
                f"{pair}: {p_name}={p_rows[p_name]!r} != {raw_name}={raw[raw_name]!r} "
                f"(abs {abs(p_rows[p_name] - raw[raw_name]):.3e})"
            )


def _write_identity_bundle(run_dir: Path, steps: tuple[int, ...] = (10, 20)) -> BundleTinyLM:
    model = BundleTinyLM(seed=0)
    order = write_run_files(
        run_dir, model, metadata=bundle_metadata(model, betas=(BETA1, BETA2), eps=EPS)
    )
    for step in steps:
        write_checkpoint(
            run_dir,
            step,
            model=model,
            optim_state=handwritten_optim_state(
                model,
                order,
                step=step,
                betas=(BETA1, BETA2),
                eps=EPS,
                exp_avg_sq_constants={n: 0.1 * (i + 1) for i, n in enumerate(order)},
            ),
            batch_a=make_batch_a(seed=step),
        )
    return model


def test_identity_preconditioner_matches_raw_columns_through_parquet(tmp_path: Path) -> None:
    """T-identity, full pipeline on a synthetic bundle, atol 1e-6.

    Goes through ``load_local_bundle`` -> ``Analyzer.run(source.model_specs())``
    -> ``ParquetSink``, i.e. the same path a real run takes, and compares the
    ``delta_loss_P`` / ``chi_net_P`` / ``chi_pos_P`` *rows of the parquet file*
    against the raw rows of the same ``(checkpoint, batch)``.
    """
    run_dir = tmp_path / "run"
    _write_identity_bundle(run_dir)
    source = load_local_bundle(run_dir, tiny_lm_factory, dtype="fp32")
    out = tmp_path / "results.parquet"
    analyzer = Analyzer(
        eval_batches={"batch_a": source.batch_a_spec()},
        preconditioner="identity",
        n_hutchinson=6,
        micro_batch_size=1,
        chi_net_method="per_sequence_cv",
        sink=str(out),
        seed=11,
        dtype="fp32",
    )
    analyzer.run(source.model_specs())

    table = pq.read_table(out)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    assert "preconditioner" in columns
    assert set(columns["preconditioner"]) == {RAW_DESCRIPTOR, "identity"}
    n_checkpoints = len(source)
    assert table.num_rows == n_checkpoints * (len(ALL_OBSERVABLES) + len(P_OBSERVABLES))

    per_key: dict[tuple[str, str], float] = {}
    for ckpt, obs, value in zip(
        columns["checkpoint_id"], columns["observable"], columns["value"], strict=True
    ):
        per_key[(ckpt, obs)] = value
    worst = 0.0
    for ckpt in set(columns["checkpoint_id"]):
        for p_name, raw_name in (
            ("delta_loss_P", "delta_loss"),
            ("chi_net_P", "chi_net"),
            ("chi_pos_P", "chi_pos"),
        ):
            deviation = abs(per_key[(ckpt, p_name)] - per_key[(ckpt, raw_name)])
            worst = max(worst, deviation)
            assert deviation <= 1e-6, (
                f"{ckpt}: {p_name}={per_key[(ckpt, p_name)]!r} != "
                f"{raw_name}={per_key[(ckpt, raw_name)]!r} (abs {deviation:.3e})"
            )
    assert worst == 0.0 or worst < 1e-6  # informational: identity P is exact


# ==================================================== the adamw source, end to end


def test_adamw_preconditioner_through_the_pipeline(tmp_path: Path) -> None:
    """The ``adamw`` P source drives real P rows, tagged "adamw", that differ
    from the raw ones (so the P is not being ignored)."""
    run_dir = tmp_path / "run"
    _write_identity_bundle(run_dir, steps=(10,))
    source = load_local_bundle(run_dir, tiny_lm_factory, dtype="fp32")
    analyzer = Analyzer(
        eval_batches={"batch_a": source.batch_a_spec()},
        preconditioner=source.adamw_preconditioner(),
        n_hutchinson=6,
        micro_batch_size=1,
        chi_net_method="per_sequence_cv",
        sink=None,
        seed=3,
        dtype="fp32",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = analyzer.run(source.model_specs())[0].rows
    raw = _by_observable(rows, RAW_DESCRIPTOR)
    p_rows = _by_observable(rows, "adamw")
    assert set(p_rows) == set(P_OBSERVABLES)
    for value in p_rows.values():
        assert value == value and abs(value) < 1e30
    # exp_avg_sq is 0.1*(i+1) here, so P is far from I and the two rulers must
    # give visibly different numbers.
    assert abs(p_rows["chi_net_P"] - raw["chi_net"]) > 1e-6 * abs(raw["chi_net"])
    assert abs(p_rows["delta_loss_P"] - raw["delta_loss"]) > 1e-9


def test_step_zero_checkpoint_emits_raw_rows_only(tmp_path: Path) -> None:
    """The step-0 policy, at pipeline level: the step-0 checkpoint gets raw rows
    and NO P rows; later checkpoints get both. An absent row cannot be mistaken
    for a measured one, which a 1/eps- or identity-valued row could be."""
    run_dir = tmp_path / "run"
    model = BundleTinyLM(seed=0)
    order = write_run_files(
        run_dir, model, metadata=bundle_metadata(model, betas=(BETA1, BETA2), eps=EPS)
    )
    empty_groups = [
        {
            "lr": 1e-3,
            "betas": (BETA1, BETA2),
            "eps": EPS,
            "weight_decay": 0.1,
            "amsgrad": False,
            "params": list(range(len(order))),
        }
    ]
    write_checkpoint(
        run_dir,
        0,
        model=model,
        optim_state={"state": {}, "param_groups": empty_groups},
        batch_a=make_batch_a(seed=0),
    )
    write_checkpoint(
        run_dir,
        10,
        model=model,
        optim_state=handwritten_optim_state(model, order, step=10, betas=(BETA1, BETA2), eps=EPS),
        batch_a=make_batch_a(seed=10),
    )

    source = load_local_bundle(run_dir, tiny_lm_factory, dtype="fp32")
    analyzer = Analyzer(
        eval_batches={"batch_a": source.batch_a_spec()},
        preconditioner=source.adamw_preconditioner(),
        n_hutchinson=4,
        micro_batch_size=1,
        chi_net_method="per_sequence_cv",
        sink=None,
        seed=1,
        dtype="fp32",
    )
    with pytest.warns(UserWarning, match="SKIPPED"):
        results = analyzer.run(source.model_specs())

    by_step = {r.checkpoint_id: r.rows for r in results}
    step0 = next(rows for cid, rows in by_step.items() if cid.endswith("step0"))
    step10 = next(rows for cid, rows in by_step.items() if cid.endswith("step10"))
    assert {r.observable for r in step0} == set(ALL_OBSERVABLES)
    assert all(r.preconditioner == RAW_DESCRIPTOR for r in step0)
    assert {r.observable for r in step10} == set(ALL_OBSERVABLES) | set(P_OBSERVABLES)
    assert {r.preconditioner for r in step10} == {RAW_DESCRIPTOR, "adamw"}


def test_preconditioner_factory_returning_garbage_raises(tmp_path: Path) -> None:
    bundle = _mlp_bundle()
    with pytest.raises(TypeError, match="expected a Preconditioner or None"):
        analyze(
            model=bundle,
            revisions=["step0"],
            eval_batches={"val": make_tiny_mlp_batch(batch_size=4, seed=0)},
            preconditioner=lambda _ckpt: "not a preconditioner",  # type: ignore[arg-type,return-value]
            n_hutchinson=2,
            sink=None,
        )
    del tmp_path


def test_cross_pair_p_symmetry_and_geometric_mean() -> None:
    """``delta_loss_P(A, B) == delta_loss_P(B, A)`` and ``chi_net_P`` cross is the
    geometric mean of the two self values — the P analogues of the invariants the
    raw cross path already guarantees."""
    bundle = _mlp_bundle()
    n_params = sum(p.numel() for p in bundle.params)
    gp = torch.Generator().manual_seed(8)
    precond = Preconditioner.from_diagonal(torch.rand(n_params, generator=gp) * 2.0 + 0.05)
    batch_a = make_tiny_mlp_batch(batch_size=6, seed=0)
    batch_b = make_tiny_mlp_batch(batch_size=6, seed=1)

    def run(pairs: list[tuple[str, str]]) -> list:
        return analyze(
            model=bundle,
            revisions=["step0"],
            eval_batches={"a": batch_a, "b": batch_b},
            cross_pairs=pairs,
            preconditioner=precond,
            n_hutchinson=8,
            micro_batch_size=6,
            seed=2,
            sink=None,
        )[0].rows

    forward = run([("a", "b")])
    backward = run([("b", "a")])
    fwd_cross = _by_observable(
        [r for r in forward if (r.batch_a, r.batch_b) == ("a", "b")], "explicit"
    )
    bwd_cross = _by_observable(
        [r for r in backward if (r.batch_a, r.batch_b) == ("b", "a")], "explicit"
    )
    assert fwd_cross["delta_loss_P"] == pytest.approx(bwd_cross["delta_loss_P"], rel=1e-12)
    assert fwd_cross["chi_net_P"] == pytest.approx(bwd_cross["chi_net_P"], rel=1e-12)

    self_a = _by_observable(
        [r for r in forward if (r.batch_a, r.batch_b) == ("a", "a")], "explicit"
    )
    self_b = _by_observable(
        [r for r in forward if (r.batch_a, r.batch_b) == ("b", "b")], "explicit"
    )
    expected = (self_a["chi_net_P"] * self_b["chi_net_P"]) ** 0.5
    assert fwd_cross["chi_net_P"] == pytest.approx(expected, rel=1e-12)
