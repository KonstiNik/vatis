"""ResultSink ABC + the long-format ResultRow dataclass.

The canonical schema is one row per ``(checkpoint_id, batch_a, batch_b,
observable, value)``. This format scales to arbitrary numbers of eval batches
and pair combinations without schema churn, and is trivial to load into
pandas and pivot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ResultRow:
    """One observation row.

    Field names match the parquet schema in ``CLAUDE.md``.

    ``preconditioner`` is a *descriptor*, not a value: it names which ruler the
    row was measured with. Raw (un-preconditioned) rows carry
    :data:`~vatis.core.preconditioner.RAW_DESCRIPTOR` (``"none"``); rows of the
    preconditioned ruler carry the preconditioner's descriptor (``"adamw"``,
    ``"identity"``, ``"explicit"``, ...). The P observables are additional
    ``observable`` *values* (``delta_loss_P``, ``chi_net_P``, ``chi_pos_P``) in
    the same long format — the schema grows by one descriptor column, not by one
    column per observable, and no raw row's value changes.
    """

    checkpoint_id: str
    revision: str
    batch_a: str
    batch_b: str
    n_valid_a: int
    n_valid_b: int
    observable: str
    value: float
    n_hutchinson: int | None = None
    hutchinson_seed: int | None = None
    wallclock_s: float | None = None
    preconditioner: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResultSink(ABC):
    """Abstract result sink. Implementations must be append-friendly."""

    @abstractmethod
    def write_rows(self, rows: list[ResultRow]) -> None:
        """Append a batch of rows to the sink."""

    def write_row(self, row: ResultRow) -> None:
        """Convenience: append a single row."""
        self.write_rows([row])

    def close(self) -> None:  # noqa: B027 — intentional empty default
        """Flush and close. Default is a no-op; override if needed."""

    def __enter__(self) -> ResultSink:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
