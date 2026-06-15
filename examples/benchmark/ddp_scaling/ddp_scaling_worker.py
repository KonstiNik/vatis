"""torchrun worker for the multi-GPU DDP scaling benchmark.

Launched once per (config, world_size) by ``ddp_scaling.sbatch`` as::

    torchrun --nproc_per_node=W --nnodes=1 --standalone \\
        examples/benchmark/ddp_scaling_worker.py --batch-file ... --batch-size B ...

Each rank loads the full model on ``cuda:LOCAL_RANK`` (vatis DDP replicates the
model and shards the *eval batch* along its first dim — it parallelizes over
data, not model size). The given ``--batch-size`` is the TOTAL across ranks;
the analyzer shards it internally. Rank 0 writes a JSON record with timing +
observables. A config that the memory pre-check refuses (per_sequence_cv too
big for one rank) is recorded with ``status="refused: ..."`` rather than
crashing the whole sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vatis import analyze  # noqa: E402
from vatis.distributed.ddp import (  # noqa: E402
    get_rank,
    get_world_size,
    init_distributed,
    shutdown_distributed,
)
from vatis.models.hf import load_hf_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--batch-file", required=True, help="torch.save'd dict with input_ids/...")
    ap.add_argument("--batch-size", type=int, required=True, help="TOTAL sequences across ranks")
    ap.add_argument("--seq-len", type=int, default=None, help="truncate sequences to this length")
    ap.add_argument("--n-hutchinson", type=int, default=32)
    ap.add_argument("--method", default="hutchinson")
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    init_distributed()
    rank = get_rank()
    world = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    full = torch.load(args.batch_file, map_location="cpu", weights_only=True)
    if args.batch_size > full["input_ids"].shape[0]:
        raise SystemExit(
            f"requested batch-size {args.batch_size} > pre-built batch "
            f"{full['input_ids'].shape[0]}; rebuild with a bigger batch."
        )
    sl = args.seq_len or full["input_ids"].shape[1]
    batch = {k: v[: args.batch_size, :sl].clone() for k, v in full.items()}
    seq_len = int(batch["input_ids"].shape[1])

    bundle = load_hf_model(args.model, revision=args.revision, dtype=args.dtype, device=device)
    n_params = sum(p.numel() for p in bundle.params)

    # Warmup (lazy CUDA init / kernel autotune) on a tiny slice so it doesn't
    # bias the timed run. Cheap and never hits the memory pre-check.
    warm = {k: v[: max(1, world)].clone() for k, v in full.items()}
    try:
        analyze(
            model=bundle,
            revisions=[args.revision],
            eval_batches={"warm": warm},
            chi_net_method="hutchinson",
            n_hutchinson=2,
            micro_batch_size=args.micro_batch_size,
            seed=args.seed,
            device=device,
            dtype=args.dtype,
            sink=None,
        )
    except Exception:  # noqa: BLE001 — warmup failures are non-fatal
        pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(local_rank)

    status = "ok"
    obs: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        results = analyze(
            model=bundle,
            revisions=[args.revision],
            eval_batches={"eval": batch},
            chi_net_method=args.method,
            n_hutchinson=args.n_hutchinson,
            micro_batch_size=args.micro_batch_size,
            seed=args.seed,
            device=device,
            dtype=args.dtype,
            sink=None,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        obs = {r.observable: r.value for r in results[0].rows}
    except Exception as e:  # noqa: BLE001 — record refusals/OOM instead of crashing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        status = f"refused: {type(e).__name__}: {str(e)[:140]}"

    peak_mb = (
        torch.cuda.max_memory_allocated(local_rank) / 1e6 if torch.cuda.is_available() else 0.0
    )

    if rank == 0:
        rec = {
            "tag": args.tag,
            "world_size": world,
            "batch_total": args.batch_size,
            "batch_per_rank": args.batch_size // world if world else args.batch_size,
            "method": args.method,
            "n_hutchinson": args.n_hutchinson,
            "seq_len": seq_len,
            "dtype": args.dtype,
            "n_params": n_params,
            "wall_s": wall,
            "peak_mb_rank0": peak_mb,
            "status": status,
            "observables": obs,
        }
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out = Path(args.out_dir) / f"{args.tag}_w{world}_b{args.batch_size}.json"
        out.write_text(json.dumps(rec, indent=2))
        print(
            f"[rank0] {args.tag} W={world} B={args.batch_size} "
            f"wall={wall:.2f}s peak={peak_mb:.0f}MB status={status}",
            flush=True,
        )

    shutdown_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
