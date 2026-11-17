"""Parquet long-format sink.

We accumulate rows in memory and flush in batches. The default is to write
one parquet file per close(); for very long sweeps users can pass
``flush_every`` to write incremental files.

The schema mirrors :class:`vatis.sinks.base.ResultRow`.

Files written before a column existed
=====================================

The column set has only ever grown at the end (``preconditioner`` first, then
``chi_net_method`` ... ``hutchinson_distribution``), and a file is always written in one
piece with the column set of the version that wrote it: the sink never appends
to an existing file (re-opening a path replaces the file), so no file mixes two
column sets. An older file therefore simply lacks the later columns. Reading it
with ``pyarrow`` / ``pandas`` works as before and code that selects columns by
name is unaffected; what fails is asking such a file for a column it does not
have, or concatenating it with a newer file without schema promotion.
:func:`read_results` covers both: it returns any vatis result file -- or several,
concatenated -- in the current column set, with the columns a file predates
filled in as documented there.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vatis.sinks.base import ResultRow, ResultSink

_SCHEMA = pa.schema(
    [
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
        # Which kernel the row was measured on: "none" for the raw columns,
        # the preconditioner's descriptor ("adamw", "identity", ...) for the
        # delta_loss_P / chi_net_P / chi_pos_P rows. Appended last so existing
        # readers that select by name are unaffected.
        ("preconditioner", pa.string()),
        # Provenance and sampling error, appended after ``preconditioner`` for
        # the same reason; null wherever a column does not apply to a row. See
        # ``ResultRow`` for what each one records.
        ("chi_net_method", pa.string()),
        ("n_ranks", pa.int32()),
        ("preconditioner_eps", pa.float64()),
        ("preconditioner_beta2", pa.float64()),
        ("preconditioner_capped_fraction", pa.float64()),
        ("value_probe_stderr", pa.float64()),
        # Two more settings of the run, on every row like ``n_hutchinson``.
        ("micro_batch_size", pa.int32()),
        ("hutchinson_distribution", pa.string()),
    ]
)

#: Column names of a result file written by this version, in file order.
RESULT_COLUMNS: tuple[str, ...] = tuple(_SCHEMA.names)

# The columns every vatis result file has had since the first release. A file
# that lacks one of these is not a vatis result file; ``read_results`` refuses it
# rather than inventing the column.
_ORIGINAL_COLUMNS: tuple[str, ...] = (
    "checkpoint_id",
    "revision",
    "batch_a",
    "batch_b",
    "n_valid_a",
    "n_valid_b",
    "observable",
    "value",
    "n_hutchinson",
    "hutchinson_seed",
    "wallclock_s",
)


class ParquetSink(ResultSink):
    """Long-format parquet sink.

    Args:
        path: output filepath. Must not be a directory.
        flush_every: if set to N>0, flush a chunk every N rows. The first
            flush creates the file with the schema; subsequent flushes append
            via ``ParquetWriter`` semantics. If ``None``, all rows accumulate
            in memory and are written on ``close()``.
    """

    def __init__(self, path: str | os.PathLike[str], *, flush_every: int | None = None) -> None:
        self.path = Path(path)
        self.flush_every = flush_every
        self._buffer: list[ResultRow] = []
        self._writer: pq.ParquetWriter | None = None

    def write_rows(self, rows: list[ResultRow]) -> None:
        if not rows:
            return
        self._buffer.extend(rows)
        if self.flush_every is not None and len(self._buffer) >= self.flush_every:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        table = _rows_to_table(self._buffer)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, schema=_SCHEMA)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        if self._writer is not None:
            # Finish streaming write.
            self._flush_buffer()
            self._writer.close()
            self._writer = None
            return
        # All-in-memory write at close time.
        if self._buffer:
            table = _rows_to_table(self._buffer)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, self.path)
            self._buffer.clear()


def _rows_to_table(rows: list[ResultRow]) -> pa.Table:
    cols: dict[str, list[object]] = {f.name: [] for f in _SCHEMA}
    for r in rows:
        d = r.to_dict()
        for k in cols:
            cols[k].append(d.get(k))
    return pa.Table.from_pydict(cols, schema=_SCHEMA)


def read_results(
    paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> pa.Table:
    """Read one or more vatis result files into the **current** column set.

    A file written by an older version lacks the columns added since. This
    returns it with those columns present, so old and new files can be read by
    the same code and concatenated:

    - ``preconditioner`` is filled with ``"none"``. That is a fact about the
      file, not a guess: the preconditioned rows and this column arrived
      together, so a file without it holds raw rows only, and ``"none"`` is what
      raw rows carry.
    - every later column (``chi_net_method``, ``n_ranks``,
      ``preconditioner_eps``, ``preconditioner_beta2``,
      ``preconditioner_capped_fraction``, ``value_probe_stderr``,
      ``micro_batch_size``, ``hutchinson_distribution``) is filled with
      **null**: the file does not record them, and nothing is inferred. In
      particular a null ``value_probe_stderr`` in an old file means "not recorded",
      not "exact".

    Columns come back in the order of :data:`RESULT_COLUMNS` and in the types
    of the current schema; columns this version does not know (a file written
    by a newer version) are kept, after the known ones. Several paths are
    concatenated in the order given. The files are not modified.

    Raises:
        ValueError: a file lacks one of the columns every vatis result file has
            had since the first release -- it is not a result file, and no
            column is invented for it.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    tables = [_read_one_result_file(Path(p)) for p in paths]
    if not tables:
        raise ValueError("read_results needs at least one path")
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote_options="default")


def _read_one_result_file(path: Path) -> pa.Table:
    table = pq.read_table(path)
    missing_original = [name for name in _ORIGINAL_COLUMNS if name not in table.column_names]
    if missing_original:
        raise ValueError(
            f"{path} is not a vatis result file: it lacks the column(s) {missing_original}, "
            "which every result file has had since the first release"
        )
    columns: dict[str, pa.Array | pa.ChunkedArray] = {}
    for schema_field in _SCHEMA:
        name = schema_field.name
        if name in table.column_names:
            columns[name] = table.column(name).cast(schema_field.type)
        elif name == "preconditioner":
            columns[name] = pa.array(["none"] * table.num_rows, type=schema_field.type)
        else:
            columns[name] = pa.nulls(table.num_rows, type=schema_field.type)
    for name in table.column_names:
        if name not in columns:
            columns[name] = table.column(name)
    return pa.table(columns)
