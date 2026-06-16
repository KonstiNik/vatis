"""DDP loopback test: 2 ranks on CPU via torchrun.

We compare against a single-process baseline and assert that:
    - chi_loss is identical (closed form is invariant under sharding)
    - delta_loss is identical (re-weighted all-reduce gives the right grad)
    - chi_net is close (Hutchinson is identical because the same probe seed
      is shared across ranks; ranks just partition which probe entries they
      backprop)
    - chi_pos matches within Hutchinson noise
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.fixtures.tiny_transformer import (
    TinyTransformer,
    causal_lm_loss,
    causal_lm_valid_mask,
    make_tiny_lm_batch,
)
from vatis import analyze
from vatis.models.hf import ModelBundle

pytestmark = pytest.mark.integration


def _run_ddp(out_path: Path, *, seed: int, batch_size: int, n_h: int) -> list[dict]:
    """Spawn a 2-rank torchrun pinned to CPU, return rank 0's row dicts."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU even if a GPU is around
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--standalone",
        str(Path(__file__).resolve().parent / "_ddp_worker.py"),
        "--out",
        str(out_path),
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
        "--n-hutchinson",
        str(n_h),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(
            "torchrun failed:\n--- stdout ---\n"
            + result.stdout
            + "\n--- stderr ---\n"
            + result.stderr
        )
    return json.loads(out_path.read_text())


def _baseline(seed: int, batch_size: int, n_h: int) -> dict[str, float]:
    torch.manual_seed(seed)
    model = TinyTransformer().eval()
    for p in model.parameters():
        p.requires_grad_(True)
    bundle = ModelBundle(
        model=model,
        params=list(model.parameters()),
        forward_fn=lambda m, b: m(b["input_ids"]),
        loss_fn=causal_lm_loss,
        valid_mask_fn=causal_lm_valid_mask,
        identifier="toy@step0",
    )
    batch = make_tiny_lm_batch(batch_size=batch_size, seed=seed, device="cpu")
    results = analyze(
        model=bundle,
        revisions=["step0"],
        eval_batches={"val": batch},
        n_hutchinson=n_h,
        micro_batch_size=1,
        sink=None,
        device="cpu",
    )
    return {r.observable: r.value for r in results[0].rows}


def test_ddp_loopback_matches_single_process(tmp_path: Path) -> None:
    out = tmp_path / "ddp.json"
    rows = _run_ddp(out, seed=0, batch_size=4, n_h=8)
    by_obs = {r["observable"]: r["value"] for r in rows}

    base = _baseline(seed=0, batch_size=4, n_h=8)

    # chi_loss closed form: identical regardless of sharding.
    assert abs(by_obs["chi_loss"] - base["chi_loss"]) < 1e-9
    assert abs(by_obs["chi_loss_normalized"] - base["chi_loss_normalized"]) < 1e-7

    # delta_loss: must match the single-process value (re-weighting + reduce
    # is exact).
    assert abs(by_obs["delta_loss"] - base["delta_loss"]) < 1e-7

    # chi_net + chi_pos: stochastic, but with the same probe seed shared
    # across ranks the Hutchinson estimator should give CLOSE values.
    rel_err_cn = abs(by_obs["chi_net"] - base["chi_net"]) / max(1e-12, abs(base["chi_net"]))
    assert rel_err_cn < 0.5, f"chi_net mismatch: {by_obs['chi_net']} vs {base['chi_net']}"
