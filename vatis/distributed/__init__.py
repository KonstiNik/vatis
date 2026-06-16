"""DDP plumbing: process-group init, sharding, all-reduce."""

from vatis.distributed.ddp import (
    all_reduce_sum,
    barrier,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_rank,
    shutdown_distributed,
)

__all__ = [
    "all_reduce_sum",
    "barrier",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_distributed",
    "is_main_rank",
    "shutdown_distributed",
]
