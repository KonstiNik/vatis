"""Optional TensorBoard result sink."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vatis.sinks.base import ResultRow, ResultSink


class TensorBoardSink(ResultSink):
    """Writes each row as a tensorboard scalar.

    Tag layout: ``{batch_a}/{observable}`` (or ``{batch_a}_x_{batch_b}/...``).
    The step is interpreted from ``checkpoint_id`` if it's a parsable int,
    otherwise the row index.
    """

    def __init__(self, log_dir: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TensorBoardSink requires tensorboard. "
                "Install with `uv pip install vatis[tensorboard]`."
            ) from exc

        # SummaryWriter is part of torch.utils.tensorboard which doesn't ship
        # type stubs. We treat it as ``Any`` so the analyzer's type checking
        # stays strict everywhere else.
        from torch.utils.tensorboard import SummaryWriter as _SW

        _writer_factory: Any = _SW
        self._writer: Any = _writer_factory(log_dir=str(log_dir))
        self._row_idx = 0

    def write_rows(self, rows: list[ResultRow]) -> None:
        for r in rows:
            tag = (
                f"{r.batch_a}/{r.observable}"
                if r.batch_a == r.batch_b
                else f"{r.batch_a}_x_{r.batch_b}/{r.observable}"
            )
            step = _coerce_step(r.checkpoint_id, fallback=self._row_idx)
            self._writer.add_scalar(tag, r.value, global_step=step)
            self._row_idx += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def _coerce_step(ckpt_id: str, *, fallback: int) -> int:
    """Best-effort: parse a leading int from a checkpoint id like ``step12345``."""
    digits = "".join(c for c in ckpt_id if c.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            return fallback
    return fallback
