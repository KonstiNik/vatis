"""DDP utilities. All wrappers degrade gracefully when distributed isn't init.

We deliberately avoid wrapping the model in ``DistributedDataParallel`` —
vatis only does eval/inference + gradient computation, never an optimizer
step. The ranks compute partial accumulators on disjoint shards of the
batch, then we all-reduce-sum once per ``(checkpoint, eval-batch)`` call.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed(backend: str | None = None) -> bool:
    """Initialize the default process group if torchrun launched us.

    Returns:
        True if the process group is now initialized (or was already), False
        if there's nothing to do (single-process mode).
    """
    if dist.is_available() and dist.is_initialized():
        return True
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return True


def shutdown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return int(dist.get_rank())


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return int(dist.get_world_size())


def is_main_rank() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce-sum a tensor across ranks. No-op single-process.

    The input is mutated in place AND returned for chaining convenience.
    """
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_sum_scalar(value: float | int, device: torch.device) -> float:
    """Convenience: scalar all-reduce-sum returning a Python float."""
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    all_reduce_sum(t)
    return float(t.item())
