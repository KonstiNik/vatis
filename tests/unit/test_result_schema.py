"""The result-file column set: what is written, where it is null, and old files.

Eight columns were appended after ``preconditioner``:

    chi_net_method                  str      the chi_net estimator that ran
    n_ranks                         int32    number of processes (DDP world size)
    preconditioner_eps              float64  eps of an AdamW P        (P rows only)
    preconditioner_beta2            float64  beta2 of an AdamW P      (P rows only)
    preconditioner_capped_fraction  float64  share of P at its ceiling (P rows only)
    value_probe_stderr                    float64  probe standard error of ``value``
    micro_batch_size                int32    sequences per backward of the run
    hutchinson_distribution         str      distribution the probes were drawn from

This file pins the names, types and order, which rows carry which of them (the
four settings of the run -- estimator, rank count, micro-batch size, probe
distribution -- on every row; the three P columns on the P rows), and that
files written with an earlier column set stay usable. What ``value_probe_stderr``
*is* has its own file, ``test_value_probe_stderr.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import vatis.analyzer as analyzer_mod
from tests.fixtures.preconditioned_toys import Fp64MLP, all_valid_mask, ce_loss, tuple_forward
from tests.fixtures.synthetic_bundle import tiny_lm_factory
from tests.unit.test_analyzer_preconditioned import BETA2, EPS, _write_identity_bundle
from vatis import analyze
from vatis.analyzer import ALL_OBSERVABLES, P_CAPPED_OBSERVABLES, P_OBSERVABLES, Analyzer
from vatis.core.chi_net import select_chi_net_method
from vatis.core.preconditioner import CEILING_RTOL, RAW_DESCRIPTOR, Preconditioner
from vatis.models.hf import ModelBundle
from vatis.models.local_bundle import load_local_bundle
from vatis.sinks import RESULT_COLUMNS, ParquetSink, ResultRow, read_results

# Written out here on purpose, not derived from the sink: this *is* the contract.
ORIGINAL_COLUMNS = [
    ("checkpoint_id", pa.string()),
    ("revision", pa.string()),
    ("batch_a", pa.string()),
    ("batch_b", pa.string()),
    ("n_valid_a", pa.int64()),
    ("n_valid_b", pa.int64()),
    ("observable", pa.string()),
    ("value", pa.float64()),
    ("n_hutchinson", pa.int32()),
    ("hutchinson_seed", pa.int64()),
    ("wallclock_s", pa.float64()),
]
PRECONDITIONER_COLUMN = [("preconditioner", pa.string())]
NEW_COLUMNS = [
    ("chi_net_method", pa.string()),
    ("n_ranks", pa.int32()),
    ("preconditioner_eps", pa.float64()),
    ("preconditioner_beta2", pa.float64()),
    ("preconditioner_capped_fraction", pa.float64()),
    ("value_probe_stderr", pa.float64()),
    ("micro_batch_size", pa.int32()),
    ("hutchinson_distribution", pa.string()),
]
CURRENT_COLUMNS = ORIGINAL_COLUMNS + PRECONDITIONER_COLUMN + NEW_COLUMNS
P_PROVENANCE = ("preconditioner_eps", "preconditioner_beta2", "preconditioner_capped_fraction")

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ the column set


def test_new_columns_are_appended_after_the_existing_ones(tmp_path: Path) -> None:
    assert list(RESULT_COLUMNS) == [name for name, _ in CURRENT_COLUMNS]
    path = tmp_path / "r.parquet"
    sink = ParquetSink(path)
    sink.write_rows([_row("chi_net")])
    sink.close()
    schema = pq.read_table(path).schema
    assert [(f.name, f.type) for f in schema] == CURRENT_COLUMNS
    # ... and the dataclass the other sinks receive has the same fields in the same order.
    assert list(_row("chi_net").to_dict()) == [name for name, _ in CURRENT_COLUMNS]


def test_a_row_that_sets_none_of_the_new_fields_writes_nulls(tmp_path: Path) -> None:
    """``ResultRow`` built the way it was before the columns existed still works."""
    path = tmp_path / "r.parquet"
    sink = ParquetSink(path, flush_every=1)
    sink.write_rows([_row("chi_loss")])
    sink.close()
    (row,) = pq.read_table(path).to_pylist()
    assert all(row[name] is None for name, _ in NEW_COLUMNS), row
    assert row["preconditioner"] == "none"


def _row(observable: str, **fields: Any) -> ResultRow:
    return ResultRow(
        checkpoint_id="c",
        revision="r",
        batch_a="A",
        batch_b="A",
        n_valid_a=3,
        n_valid_b=3,
        observable=observable,
        value=1.5,
        n_hutchinson=4,
        hutchinson_seed=11,
        wallclock_s=0.25,
        **fields,
    )


# ------------------------------------------------------- what the analyzer writes


def _mlp_bundle() -> ModelBundle:
    model = Fp64MLP(in_dim=6, hidden=8, n_out=5, seed=0)
    return ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=tuple_forward,
        loss_fn=ce_loss,
        valid_mask_fn=all_valid_mask,
        identifier="mlp@step7",
    )


def _mlp_batch(n: int, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, 6, generator=gen, dtype=torch.float64),
        torch.randint(0, 5, (n,), generator=gen),
    )


def _second_moment(bundle: ModelBundle, zero_fraction: float) -> torch.Tensor:
    n_params = sum(p.numel() for p in bundle.params)
    gen = torch.Generator().manual_seed(5)
    exp_avg_sq = torch.rand(n_params, generator=gen, dtype=torch.float64) + 0.1
    exp_avg_sq[torch.rand(n_params, generator=gen) < zero_fraction] = 0.0
    return exp_avg_sq


@pytest.mark.parametrize("method", ["hutchinson", "per_sequence_cv"])
def test_adamw_rows_carry_eps_beta2_and_the_capped_fraction(tmp_path: Path, method: str) -> None:
    """Through ``analyze()`` and the parquet file, self pairs and a cross pair.

    The fraction is checked against a count made here from the second moment
    (``v == 0``), not against ``Preconditioner.count_capped``.
    """
    bundle = _mlp_bundle()
    exp_avg_sq = _second_moment(bundle, zero_fraction=0.3)
    n_zero = int((exp_avg_sq == 0).sum())
    assert 0 < n_zero < exp_avg_sq.numel()
    precond = Preconditioner.adamw(exp_avg_sq, step=7, beta2=0.98, eps=1e-3)
    out = tmp_path / "results.parquet"
    analyze(
        model=bundle,
        revisions=["step7"],
        eval_batches={"a": _mlp_batch(4, 1), "b": _mlp_batch(6, 2)},
        cross_pairs=[("a", "b")],
        preconditioner=precond,
        chi_net_method=method,
        n_hutchinson=3,
        micro_batch_size=2,
        sink=str(out),
        dtype="fp32",
    )
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == 3 * len(ALL_OBSERVABLES + P_OBSERVABLES + P_CAPPED_OBSERVABLES)
    for row in rows:
        assert row["chi_net_method"] == method, row
        assert row["n_ranks"] == 1, row
        assert row["micro_batch_size"] == 2, row
        assert row["hutchinson_distribution"] == "rademacher", row
        if row["observable"] in P_OBSERVABLES + P_CAPPED_OBSERVABLES:
            assert row["preconditioner"] == "adamw"
            assert row["preconditioner_eps"] == 1e-3
            assert row["preconditioner_beta2"] == 0.98
            assert row["preconditioner_capped_fraction"] == n_zero / exp_avg_sq.numel()
        else:
            # Raw rows -- chi_loss among them -- were not computed with a P.
            assert row["preconditioner"] == RAW_DESCRIPTOR
            assert all(row[name] is None for name in P_PROVENANCE), row


def test_adamw_from_a_bundle_records_the_saved_hyperparameters(tmp_path: Path) -> None:
    """The bundle path: eps / beta2 are the ones in the saved ``param_groups``;
    an override is what gets recorded when one is given."""
    run_dir = tmp_path / "run"
    _write_identity_bundle(run_dir, steps=(10,))
    source = load_local_bundle(run_dir, tiny_lm_factory, dtype="fp32")

    def p_rows(preconditioner: Any) -> list[ResultRow]:
        analyzer = Analyzer(
            eval_batches={"batch_a": source.batch_a_spec()},
            preconditioner=preconditioner,
            n_hutchinson=2,
            chi_net_method="hutchinson",
            sink=None,
            dtype="fp32",
        )
        rows = analyzer.run(source.model_specs())[0].rows
        return [r for r in rows if r.preconditioner != RAW_DESCRIPTOR]

    saved = p_rows(source.adamw_preconditioner())
    assert saved and {r.preconditioner for r in saved} == {"adamw"}
    assert {r.preconditioner_eps for r in saved} == {EPS}
    assert {r.preconditioner_beta2 for r in saved} == {BETA2}
    # Every second moment of this fixture is positive: a ceiling, nobody at it.
    assert {r.preconditioner_capped_fraction for r in saved} == {0.0}

    overridden = p_rows(source.adamw_preconditioner(eps=1e-5))
    assert {r.preconditioner for r in overridden} == {"adamw(eps=1e-05)"}
    assert {r.preconditioner_eps for r in overridden} == {1e-5}
    assert {r.preconditioner_beta2 for r in overridden} == {BETA2}


def test_capped_fraction_uses_the_ceiling_criterion_of_the_capped_rows() -> None:
    """Entries within ``CEILING_RTOL`` below the ceiling count, entries below that do not."""
    ceiling = 8.0
    diag = torch.tensor(
        [ceiling, ceiling * (1 - 0.5 * CEILING_RTOL), ceiling * (1 - 4 * CEILING_RTOL), 1.0, 2.0],
        dtype=torch.float64,
    )
    precond = Preconditioner.from_diagonal(diag, ceiling=ceiling)
    assert precond.count_capped() == 2
    assert precond.capped_fraction() == 2 / 5
    assert precond.to(dtype=torch.float32).capped_fraction() == 2 / 5
    assert Preconditioner.from_diagonal(diag).capped_fraction() is None
    # Provenance survives a move / cast; a P that is not AdamW's has none.
    adamw = Preconditioner.adamw(torch.tensor([0.0, 1.0, 4.0]), step=3, beta2=0.9, eps=0.25)
    moved = adamw.to(dtype=torch.float64)
    assert moved is not adamw
    assert (moved.eps, moved.beta2, moved.ceiling) == (0.25, 0.9, 4.0)
    assert moved.capped_fraction() == 1 / 3
    assert (precond.eps, precond.beta2) == (None, None)
    with pytest.raises(ValueError, match="beta2"):
        Preconditioner(diag=diag, descriptor="x", beta2=1.0)
    with pytest.raises(ValueError, match="eps"):
        Preconditioner(diag=diag, descriptor="x", eps=-1.0)


def test_a_p_that_is_not_adamw_writes_nulls_where_it_has_nothing_to_say() -> None:
    bundle = _mlp_bundle()
    n_params = sum(p.numel() for p in bundle.params)
    common: dict[str, Any] = {
        "model": bundle,
        "revisions": ["step7"],
        "eval_batches": {"a": _mlp_batch(4)},
        "n_hutchinson": 2,
        "sink": None,
        "dtype": "fp32",
    }

    def p_rows(preconditioner: Any) -> list[ResultRow]:
        rows = analyze(preconditioner=preconditioner, **common)[0].rows
        return [r for r in rows if r.preconditioner != RAW_DESCRIPTOR]

    for row in p_rows("identity"):
        assert (row.preconditioner_eps, row.preconditioner_beta2) == (None, None)
        assert row.preconditioner_capped_fraction is None
    # An explicit P with a ceiling has a capped fraction but no eps / beta2.
    diag = torch.full((n_params,), 2.0, dtype=torch.float64)
    diag[: n_params // 4] = 10.0
    for row in p_rows(Preconditioner.from_diagonal(diag, ceiling=10.0)):
        assert (row.preconditioner_eps, row.preconditioner_beta2) == (None, None)
        assert row.preconditioner_capped_fraction == (n_params // 4) / n_params


@pytest.mark.parametrize(
    ("method", "distribution", "micro"),
    [
        ("hutchinson", "gaussian", 3),
        ("per_sequence_cv", "gaussian", 1),
        ("hutchinson", "rademacher", 8),
    ],
)
def test_micro_batch_size_and_probe_distribution_are_on_every_row(
    tmp_path: Path, method: str, distribution: str, micro: int
) -> None:
    """The two settings the sampled values and their standard error depend on.

    Like ``chi_net_method`` and ``n_ranks`` they describe the run and are written
    on every row of it: the sampled rows, the rows no probe enters (``chi_loss``,
    ``delta_loss``), the P rows, and the cross pair, through the parquet file.

    ``micro_batch_size`` is the setting the run was made with, also where it
    exceeds a batch (``micro = 8`` on batches of 4 and 6: each batch is then one
    micro-batch, and the column still reads 8). The distribution is what the
    estimator reports having drawn from, checked here against a property of
    the numbers themselves: Gaussian and Rademacher probes give different
    ``chi_net`` for the same seed.
    """
    bundle = _mlp_bundle()
    n_params = sum(p.numel() for p in bundle.params)

    def run(dist: str, sink: str | None) -> list[ResultRow]:
        return analyze(
            model=bundle,
            revisions=["step7"],
            eval_batches={"a": _mlp_batch(4, 1), "b": _mlp_batch(6, 2)},
            cross_pairs=[("a", "b")],
            preconditioner=Preconditioner.identity(n_params),
            chi_net_method=method,
            hutchinson_distribution=dist,
            n_hutchinson=3,
            micro_batch_size=micro,
            sink=sink,
            dtype="fp32",
        )[0].rows

    out = tmp_path / "results.parquet"
    in_memory = run(distribution, str(out))
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == len(in_memory) == 3 * len(ALL_OBSERVABLES + P_OBSERVABLES)
    assert {(r["batch_a"], r["batch_b"]) for r in rows} == {("a", "a"), ("b", "b"), ("a", "b")}
    assert {r["observable"] for r in rows} >= {"chi_loss", "delta_loss", "chi_net", "chi_pos_P"}
    for row in rows:
        assert row["micro_batch_size"] == micro, row
        assert row["hutchinson_distribution"] == distribution, row
    assert {(r.micro_batch_size, r.hutchinson_distribution) for r in in_memory} == {
        (micro, distribution)
    }

    other = "rademacher" if distribution == "gaussian" else "gaussian"
    values = {(r.batch_a, r.batch_b, r.observable): r.value for r in in_memory}
    for r in run(other, None):
        assert r.hutchinson_distribution == other
        same_row = values[(r.batch_a, r.batch_b, r.observable)]
        if r.observable in ("chi_loss", "delta_loss"):
            assert r.value == same_row, r.observable
        if r.observable == "chi_net":
            assert r.value != same_row, "the distribution did not change the probes"


@pytest.mark.parametrize("batch_size", [4, 40])
def test_the_estimator_that_ran_is_recorded_when_it_was_selected_automatically(
    batch_size: int,
) -> None:
    """``chi_net_method=None``: the column holds what the selection picked for this
    batch, not the request. Compared with the selector rather than a literal, so
    the test says the same thing whichever rule the selector implements."""
    rows = analyze(
        model=_mlp_bundle(),
        revisions=["step7"],
        eval_batches={"a": _mlp_batch(batch_size)},
        chi_net_method=None,
        n_hutchinson=2,
        micro_batch_size=8,
        sink=None,
        dtype="fp32",
    )[0].rows
    picked = select_chi_net_method(batch_size, None)
    assert picked in ("hutchinson", "per_sequence_cv")
    assert {r.chi_net_method for r in rows} == {picked}


def test_a_cross_pair_of_two_estimators_names_both(monkeypatch: pytest.MonkeyPatch) -> None:
    """The automatic selection is per batch. If it picks differently for the two
    batches of a pair, the pair's rows name both estimators, in pair order."""
    monkeypatch.setattr(
        analyzer_mod,
        "select_chi_net_method",
        lambda b_total, requested: "per_sequence_cv" if b_total <= 4 else "hutchinson",
    )
    rows = analyze(
        model=_mlp_bundle(),
        revisions=["step7"],
        eval_batches={"small": _mlp_batch(4, 1), "large": _mlp_batch(6, 2)},
        cross_pairs=[("small", "large"), ("large", "small")],
        n_hutchinson=2,
        micro_batch_size=2,
        sink=None,
        dtype="fp32",
    )[0].rows
    by_pair: dict[tuple[str, str], set[str | None]] = {}
    for r in rows:
        by_pair.setdefault((r.batch_a, r.batch_b), set()).add(r.chi_net_method)
    assert by_pair == {
        ("small", "small"): {"per_sequence_cv"},
        ("large", "large"): {"hutchinson"},
        ("small", "large"): {"per_sequence_cv+hutchinson"},
        ("large", "small"): {"hutchinson+per_sequence_cv"},
    }
    # The other settings are the analyzer's own and the same for both batches.
    assert {(r.micro_batch_size, r.hutchinson_distribution) for r in rows} == {(2, "rademacher")}
    # Two independent estimates still combine: the pair has a standard error.
    cross = [r for r in rows if r.batch_a != r.batch_b and r.observable == "chi_net"]
    assert cross and all(
        r.value_probe_stderr is not None and r.value_probe_stderr > 0 for r in cross
    )


def test_n_ranks_is_the_world_size_the_run_saw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analyzer_mod, "get_world_size", lambda: 3)
    rows = analyze(
        model=_mlp_bundle(),
        revisions=["step7"],
        eval_batches={"a": _mlp_batch(6)},
        n_hutchinson=2,
        sink=None,
        dtype="fp32",
    )[0].rows
    assert rows and {r.n_ranks for r in rows} == {3}


# ------------------------------------------------- files with an earlier column set


def _write_old_file(path: Path, columns: list[tuple[str, pa.DataType]], n_rows: int = 3) -> None:
    """A result file as an earlier version wrote it: exactly ``columns``, nothing else."""
    data: dict[str, list[Any]] = {
        "checkpoint_id": [f"toy@step{i}" for i in range(n_rows)],
        "revision": [f"step{i}" for i in range(n_rows)],
        "batch_a": ["val"] * n_rows,
        "batch_b": ["val"] * n_rows,
        "n_valid_a": [60] * n_rows,
        "n_valid_b": [60] * n_rows,
        "observable": ["chi_net"] * n_rows,
        "value": [float(i) + 0.5 for i in range(n_rows)],
        "n_hutchinson": [8] * n_rows,
        "hutchinson_seed": [1000 + i for i in range(n_rows)],
        "wallclock_s": [0.1] * n_rows,
        "preconditioner": ["none", "adamw", "none"][:n_rows],
        "chi_net_method": ["hutchinson"] * n_rows,
        "n_ranks": [2] * n_rows,
        "preconditioner_eps": [None, 1e-8, None][:n_rows],
        "preconditioner_beta2": [None, 0.95, None][:n_rows],
        "preconditioner_capped_fraction": [None, 0.25, None][:n_rows],
        "value_probe_stderr": [0.125] * n_rows,
    }
    schema = pa.schema(columns)
    pq.write_table(pa.Table.from_pydict({n: data[n] for n in schema.names}, schema=schema), path)


OLD_GENERATIONS = {
    "before_preconditioner": ORIGINAL_COLUMNS,
    "before_provenance": ORIGINAL_COLUMNS + PRECONDITIONER_COLUMN,
    # Never released or merged: the column set of the first version of the change
    # that added the provenance columns, before the last two were appended. It is
    # here because nothing special is needed for it -- ``read_results`` fills
    # whatever columns a file lacks -- not because such files are known to exist.
    "before_run_settings": ORIGINAL_COLUMNS + PRECONDITIONER_COLUMN + NEW_COLUMNS[:6],
}


@pytest.mark.parametrize("generation", sorted(OLD_GENERATIONS))
def test_an_old_file_still_loads_as_it_always_did(tmp_path: Path, generation: str) -> None:
    """Nothing about reading an old file by column name changes: plain pyarrow
    returns the columns the file has, and a reader that selects ``observable`` /
    ``value`` (every script under ``examples/``) is unaffected."""
    columns = OLD_GENERATIONS[generation]
    path = tmp_path / "old.parquet"
    _write_old_file(path, columns)
    table = pq.read_table(path)
    assert table.column_names == [name for name, _ in columns]
    assert [r["value"] for r in table.to_pylist() if r["observable"] == "chi_net"] == [
        0.5,
        1.5,
        2.5,
    ]


@pytest.mark.parametrize("generation", sorted(OLD_GENERATIONS))
def test_read_results_returns_an_old_file_in_the_current_columns(
    tmp_path: Path, generation: str
) -> None:
    columns = OLD_GENERATIONS[generation]
    path = tmp_path / "old.parquet"
    _write_old_file(path, columns)
    before = path.read_bytes()
    table = read_results(path)
    assert [(f.name, f.type) for f in table.schema] == CURRENT_COLUMNS
    rows = table.to_pylist()
    # What the file holds comes back unchanged ...
    assert rows == [
        {**old, **{name: rows[i][name] for name in rows[i] if name not in old}}
        for i, old in enumerate(pq.read_table(path).to_pylist())
    ]
    # ... the columns it predates are null: not recorded is not a value ...
    in_file = {name for name, _ in columns}
    predated = [name for name, _ in NEW_COLUMNS if name not in in_file]
    assert {"micro_batch_size", "hutchinson_distribution"} <= set(predated)
    for row in rows:
        assert all(row[name] is None for name in predated), row
    # ... except ``preconditioner`` in a file from before that column, which can
    # only hold raw rows (the P rows and the column were introduced together).
    if generation == "before_preconditioner":
        assert {row["preconditioner"] for row in rows} == {"none"}
    else:
        assert [row["preconditioner"] for row in rows] == ["none", "adamw", "none"]
    assert path.read_bytes() == before, "read_results must not touch the file"


def test_read_results_concatenates_old_and_new_files(tmp_path: Path) -> None:
    oldest, old, new = (tmp_path / name for name in ("a.parquet", "b.parquet", "c.parquet"))
    _write_old_file(oldest, OLD_GENERATIONS["before_preconditioner"], n_rows=2)
    _write_old_file(old, OLD_GENERATIONS["before_provenance"], n_rows=3)
    sink = ParquetSink(new)
    sink.write_rows(
        [
            _row(
                "chi_net",
                chi_net_method="hutchinson",
                n_ranks=2,
                value_probe_stderr=0.125,
                micro_batch_size=4,
                hutchinson_distribution="gaussian",
            )
        ]
    )
    sink.close()

    # Without help the files do not go together ...
    with pytest.raises(pa.ArrowInvalid):
        pa.concat_tables([pq.read_table(p) for p in (oldest, old, new)])
    # ... with read_results they do, in the order given.
    table = read_results([oldest, old, new])
    assert [(f.name, f.type) for f in table.schema] == CURRENT_COLUMNS
    assert table.num_rows == 6
    assert table.column("chi_net_method").to_pylist() == [None] * 5 + ["hutchinson"]
    assert table.column("n_ranks").to_pylist() == [None] * 5 + [2]
    assert table.column("value_probe_stderr").to_pylist() == [None] * 5 + [0.125]
    assert table.column("micro_batch_size").to_pylist() == [None] * 5 + [4]
    assert table.column("hutchinson_distribution").to_pylist() == [None] * 5 + ["gaussian"]
    assert table.column("preconditioner").to_pylist() == ["none"] * 3 + ["adamw", "none", "none"]


@pytest.mark.parametrize("flush_every", [None, 1])
def test_writing_to_the_path_of_an_old_file_replaces_it(
    tmp_path: Path, flush_every: int | None
) -> None:
    """The sink never appends to an existing file (``test_parquet_sink.py`` pins
    that), so an old file at the target path is replaced as a whole and no file
    ever mixes two column sets."""
    path = tmp_path / "results.parquet"
    _write_old_file(path, OLD_GENERATIONS["before_provenance"])
    sink = ParquetSink(path, flush_every=flush_every)
    sink.write_rows([_row("chi_net", chi_net_method="hutchinson", n_ranks=1)])
    sink.close()
    table = pq.read_table(path)
    assert [(f.name, f.type) for f in table.schema] == CURRENT_COLUMNS
    assert table.num_rows == 1


def test_read_results_refuses_a_file_that_is_not_a_result_file(tmp_path: Path) -> None:
    path = tmp_path / "other.parquet"
    pq.write_table(pa.table({"observable": ["chi_net"], "value": [1.0]}), path)
    with pytest.raises(ValueError, match="not a vatis result file"):
        read_results(path)
    with pytest.raises(ValueError, match="at least one path"):
        read_results([])


def test_read_results_keeps_columns_it_does_not_know(tmp_path: Path) -> None:
    """A file from a *newer* version: its extra column is kept, after the known ones."""
    path = tmp_path / "newer.parquet"
    schema = pa.schema([*CURRENT_COLUMNS, ("some_future_column", pa.int64())])
    row = {**_row("chi_net").to_dict(), "some_future_column": 9}
    pq.write_table(pa.Table.from_pylist([row], schema=schema), path)
    table = read_results(path)
    assert table.column_names == [*RESULT_COLUMNS, "some_future_column"]
    assert table.column("some_future_column").to_pylist() == [9]


@pytest.mark.parametrize(
    "relative_path",
    ["examples/pythia_sweep/results.parquet", "examples/output/results.parquet"],
)
def test_the_result_files_shipped_with_the_examples_stay_readable(relative_path: str) -> None:
    """Real files written by earlier versions, as committed.

    When this was written the first had the original eleven columns and the
    second those plus ``preconditioner``. The test does not depend on that: for
    whatever columns a shipped file has, ``read_results`` must hand them back
    unchanged and fill in the rest as documented.
    """
    path = _REPO_ROOT / relative_path
    plain = pq.read_table(path)
    table = read_results(path)
    assert table.column_names == list(RESULT_COLUMNS)
    assert table.num_rows == plain.num_rows > 0
    for name, _ in CURRENT_COLUMNS:
        got = table.column(name).to_pylist()
        if name in plain.column_names:
            assert got == plain.column(name).to_pylist(), name
        elif name == "preconditioner":
            assert set(got) == {"none"}
        else:
            assert table.column(name).null_count == table.num_rows, name
