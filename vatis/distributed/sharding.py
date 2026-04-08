"""Per-rank batch sharding.

Each rank gets a contiguous slice of the batch dim. If the batch isn't
divisible by world size the last few ranks get one fewer sample. The
all-reduce-sum semantics handle this naturally because we accumulate
token-weighted sums, not means.
"""

from __future__ import annotations

from typing import Any

from vatis.data.collate import Batch, batch_size, slice_batch


def shard_batch(batch: Batch, rank: int, world_size: int) -> tuple[Any, int, int]:
    """Return ``(local_batch, local_size, global_size)`` for the given rank.

    The slice for rank r is ``[r * base + min(r, rem), (r+1) * base + min(r+1, rem))``
    where ``base, rem = divmod(global_size, world_size)`` — this is the
    standard "spread the remainder over the first ``rem`` ranks" rule.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} out of range [0, {world_size})")

    n = batch_size(batch)
    base, rem = divmod(n, world_size)
    start = rank * base + min(rank, rem)
    extra = 1 if rank < rem else 0
    stop = start + base + extra
    local = slice_batch(batch, start, stop)
    return local, stop - start, n
