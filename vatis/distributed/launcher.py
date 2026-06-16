"""``torchrun`` wrapper for the vatis CLI.

For ``num_gpus > 1`` the CLI re-execs itself under ``torchrun`` with the
right ``--nproc_per_node`` and forwards the original arguments. The worker
side (``vatis.distributed._worker``) is what actually runs the analysis.

We do this as a subprocess.exec rather than importing torch.distributed.run
because that lets us keep the parent process small and the worker side
clean.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Sequence


def launch_torchrun(num_gpus: int, args: Sequence[str]) -> int:
    """Re-launch the vatis CLI under torchrun and return its exit code."""
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if num_gpus == 1:
        # No need for torchrun; run directly. The CLI handles single-GPU mode.
        return _run_direct(args)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--nnodes=1",
        "--standalone",
        "-m",
        "vatis._worker",
        *args,
    ]
    print(f"[vatis] launching: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.call(cmd, env=os.environ.copy())


def _run_direct(args: Sequence[str]) -> int:
    cmd = [sys.executable, "-m", "vatis._worker", *args]
    return subprocess.call(cmd, env=os.environ.copy())
