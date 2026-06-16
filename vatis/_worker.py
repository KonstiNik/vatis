"""Worker entry point launched under torchrun.

This module is invoked by ``vatis.distributed.launcher`` as
``python -m vatis._worker ...``. It mirrors the CLI argument shape so the
launcher can pass the original argv unchanged. On rank > 0 the sink is
disabled (only rank 0 writes); the analyzer's :func:`is_main_rank` check
already enforces this.
"""

from __future__ import annotations

import sys

from vatis.cli import build_parser, run_from_args
from vatis.distributed.ddp import init_distributed, shutdown_distributed


def main() -> int:
    init_distributed()
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    try:
        run_from_args(args)
    finally:
        shutdown_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
