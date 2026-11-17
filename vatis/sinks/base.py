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

    ``preconditioner`` is a *descriptor*, not a value: it names which kernel the
    row was measured on. Raw (un-preconditioned) rows carry
    :data:`~vatis.core.preconditioner.RAW_DESCRIPTOR` (``"none"``); the
    preconditioned observables carry the preconditioner's descriptor (``"adamw"``,
    ``"identity"``, ``"explicit"``, ...). The P observables are additional
    ``observable`` *values* (``delta_loss_P``, ``chi_net_P``, ``chi_pos_P``) in
    the same long format — the schema grows by one descriptor column, not by one
    column per observable, and no raw row's value changes.

    The fields after ``preconditioner`` say how a value was obtained and how far
    to trust it; each is ``None`` (a null in the file) where it does not apply:

    - ``chi_net_method``: the ``chi_net`` estimator the run used for this row's
      batch (``"hutchinson"`` / ``"per_sequence_cv"``) -- the one actually used,
      whatever selected it. A cross pair whose two batches were measured with
      different estimators carries both, ``"<method of batch_a>+<method of
      batch_b>"``.
    - ``n_ranks``: the number of processes (DDP world size) the run used, 1 for
      a single process. Every rank draws its own probe stream, so a multi-rank
      ``chi_net`` depends on it.
    - ``preconditioner_eps`` / ``preconditioner_beta2``: on the rows computed
      with a ``P = 1/(sqrt(v_hat) + eps)`` (the ``adamw`` source), the ``eps``
      and ``beta2`` that ``P`` was built with.
    - ``preconditioner_capped_fraction``: on the rows computed with a ``P`` that
      has a ceiling (``1/eps`` for ``adamw``), the fraction of the coordinates
      of ``P`` that sit at it.
    - ``value_stderr``: the standard error of ``value`` under the draw of the
      Hutchinson probes, on the rows whose value is sampled (``chi_net``,
      ``chi_net_normalized``, ``chi_pos`` and their ``_P`` / ``_P_capped``
      counterparts); ``None`` on ``chi_loss`` / ``delta_loss`` rows, which are
      not sampled, and with a single probe. Definition and limits:
      ``docs/SPEC.md`` -> "Standard error of the sampled values".
    - ``micro_batch_size``: the ``micro_batch_size`` the run was made with
      (sequences per backward). One probe spans a micro-batch, so the sampled
      values and their standard error are specific to it.
    - ``hutchinson_distribution``: the distribution the probes were drawn from
      (``"rademacher"`` / ``"gaussian"``), as the estimator that ran reports it.

    ``chi_net_method``, ``n_ranks``, ``micro_batch_size`` and
    ``hutchinson_distribution`` describe the run, like ``n_hutchinson``, and are
    written on every row -- also on the ``chi_loss`` / ``delta_loss`` rows,
    whose values no probe enters.
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
    chi_net_method: str | None = None
    n_ranks: int | None = None
    preconditioner_eps: float | None = None
    preconditioner_beta2: float | None = None
    preconditioner_capped_fraction: float | None = None
    value_stderr: float | None = None
    micro_batch_size: int | None = None
    hutchinson_distribution: str | None = None

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
