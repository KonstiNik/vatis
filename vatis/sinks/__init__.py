"""Result sinks: parquet (canonical), wandb, tensorboard."""

from vatis.sinks.base import ResultRow, ResultSink
from vatis.sinks.parquet import ParquetSink

__all__ = ["ParquetSink", "ResultRow", "ResultSink"]
