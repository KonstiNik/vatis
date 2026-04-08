"""Parquet long-format sink.

We accumulate rows in memory and flush in batches. The default is to write
one parquet file per close(); for very long sweeps users can pass
``flush_every`` to write incremental files.

The schema mirrors :class:`vatis.sinks.base.ResultRow`.
"""

from __future__ import annotations

import os
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
    ]
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
