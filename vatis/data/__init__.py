"""Data plumbing: EvalBatchSpec, micro-batch splitting, mask handling."""

from vatis.data.batches import EvalBatchSpec
from vatis.data.collate import iter_micro_batches

__all__ = ["EvalBatchSpec", "iter_micro_batches"]
