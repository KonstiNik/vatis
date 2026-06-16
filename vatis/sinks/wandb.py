"""Optional W&B result sink. Lazy-imports wandb so the dep is truly optional."""

from __future__ import annotations

from typing import Any

from vatis.sinks.base import ResultRow, ResultSink


class WandbSink(ResultSink):
    """Logs each row as a W&B metric.

    The metric name is ``f"{batch_a}/{observable}"`` for self pairs and
    ``f"{batch_a}_x_{batch_b}/{observable}"`` for cross pairs. ``checkpoint_id``
    is logged as ``"checkpoint"`` so wandb's x-axis selector can pick it.
    """

    def __init__(
        self,
        *,
        project: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        entity: str | None = None,
    ) -> None:
        try:
            import wandb  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "WandbSink requires wandb. Install with `uv pip install vatis[wandb]`."
            ) from exc
        import wandb

        self._wandb = wandb
        self._run: Any = wandb.init(
            project=project,
            name=run_name,
            config=config,
            entity=entity,
            reinit="finish_previous",
        )

    def write_rows(self, rows: list[ResultRow]) -> None:
        for r in rows:
            metric = (
                f"{r.batch_a}/{r.observable}"
                if r.batch_a == r.batch_b
                else f"{r.batch_a}_x_{r.batch_b}/{r.observable}"
            )
            self._wandb.log(
                {metric: r.value, "checkpoint": r.checkpoint_id},
                commit=False,
            )

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
