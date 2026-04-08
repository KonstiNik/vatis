"""Distributed-mode (DDP) sanity tests.

We don't spin up a real process group here — those tests live in
``tests/integration``. Instead we test the standalone helpers
(``shard_batch``, the ddp wrappers in single-process mode) which is enough
to catch the common bugs (off-by-one sharding, wrong rank weighting).
"""

from __future__ import annotations

import torch

from vatis.data.collate import batch_size
from vatis.distributed.ddp import (
    all_reduce_sum_scalar,
    barrier,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_rank,
)
from vatis.distributed.sharding import shard_batch


def test_single_process_world_size_is_one() -> None:
    assert not is_distributed()
    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main_rank()
    barrier()  # no-op
    assert all_reduce_sum_scalar(3.5, torch.device("cpu")) == 3.5


def test_shard_batch_tensor_even_split() -> None:
    batch = torch.arange(8).reshape(8, 1)
    s0, n0, n_global = shard_batch(batch, rank=0, world_size=2)
    s1, n1, _ = shard_batch(batch, rank=1, world_size=2)
    assert n_global == 8
    assert n0 == 4
    assert n1 == 4
    assert torch.equal(s0, batch[:4])
    assert torch.equal(s1, batch[4:])


def test_shard_batch_remainder_distributed_to_first_ranks() -> None:
    batch = torch.arange(10).reshape(10, 1)
    sizes = [shard_batch(batch, rank=r, world_size=3)[1] for r in range(3)]
    assert sizes == [4, 3, 3]
    # The shard contents should partition the batch in order.
    starts = []
    for r in range(3):
        local, _, _ = shard_batch(batch, rank=r, world_size=3)
        starts.append(int(local[0, 0].item()))
    assert starts == [0, 4, 7]


def test_shard_batch_dict() -> None:
    batch = {
        "input_ids": torch.arange(6).reshape(6, 1),
        "attention_mask": torch.ones(6, 1, dtype=torch.long),
    }
    local, n_local, n_global = shard_batch(batch, rank=0, world_size=2)
    assert isinstance(local, dict)
    assert n_global == 6
    assert n_local == 3
    assert batch_size(local) == 3
    assert torch.equal(local["input_ids"], torch.arange(3).reshape(3, 1))
