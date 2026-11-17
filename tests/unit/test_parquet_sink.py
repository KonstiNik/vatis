"""``ParquetSink`` behaviour, including the documented truncation-on-reopen footgun.

``CLAUDE.md`` §Known issues records that a second ``ParquetSink(path)`` overwrites
the first run's rows. Nothing pinned it, so a future "fix" (or an accidental
change to append semantics) would be invisible. This file pins the current
behaviour so that a change to it is a deliberate, visible act.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from vatis.sinks.base import ResultRow
from vatis.sinks.parquet import ParquetSink


def _row(observable: str, value: float = 1.0) -> ResultRow:
    return ResultRow(
        checkpoint_id="c",
        revision="r",
        batch_a="A",
        batch_b="A",
        n_valid_a=1,
        n_valid_b=1,
        observable=observable,
        value=value,
        n_hutchinson=1,
        hutchinson_seed=1,
        wallclock_s=0.0,
        preconditioner="none",
    )


def test_buffered_sink_writes_every_row_on_close(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    sink = ParquetSink(path)
    sink.write_rows([_row("chi_loss"), _row("chi_net")])
    sink.close()
    table = pq.read_table(path).to_pydict()
    assert table["observable"] == ["chi_loss", "chi_net"]


def test_flush_every_streams_and_still_writes_everything(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    sink = ParquetSink(path, flush_every=1)
    sink.write_rows([_row("a")])
    sink.write_rows([_row("b")])
    sink.close()
    assert pq.read_table(path).to_pydict()["observable"] == ["a", "b"]


def test_reopening_the_same_path_truncates(tmp_path: Path) -> None:
    """DOCUMENTED FOOTGUN, pinned: a second sink discards the first run's rows.

    ``analyze(..., sink="results.parquet")`` called twice in a loop keeps only the
    last call's rows. The deployment examples sidestep it with a single
    ``analyze()`` over many revisions. If this test starts failing because the
    sink learned to append, that is the fix landing — update ``CLAUDE.md``
    §Known issues in the same commit.
    """
    path = tmp_path / "r.parquet"
    first = ParquetSink(path)
    first.write_rows([_row("chi_loss")])
    first.close()
    assert pq.read_table(path).num_rows == 1

    second = ParquetSink(path)
    second.write_rows([_row("chi_net")])
    second.close()
    table = pq.read_table(path).to_pydict()
    assert table["observable"] == ["chi_net"], (
        "ParquetSink is documented to truncate on re-open; it appended instead, "
        "so CLAUDE.md §Known issues is now stale"
    )


def test_reopening_truncates_on_the_streaming_path_too(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    first = ParquetSink(path, flush_every=1)
    first.write_rows([_row("a")])
    first.write_rows([_row("b")])
    first.close()
    second = ParquetSink(path, flush_every=1)
    second.write_rows([_row("c")])
    second.close()
    assert pq.read_table(path).to_pydict()["observable"] == ["c"]


def test_empty_sink_writes_nothing_and_does_not_create_the_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "r.parquet"
    sink = ParquetSink(path)
    sink.write_rows([])
    sink.close()
    assert not path.exists()
