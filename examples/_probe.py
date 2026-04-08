"""One-shot probe for the deployment example sizing.

Loads a small published causal LM, runs a single ``analyze()`` call with
both ``chi_net_method`` values at modest hyperparameters, and prints
wallclock + peak GPU memory + observable values. Output is consumed by
hand to populate ``examples/BENCHMARK.md``.

Run with::

    .venv/bin/python examples/_probe.py
"""

from __future__ import annotations

import os

# Must come before importing torch / transformers.
os.environ.setdefault("HF_HOME", "/data/knikolaou/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/knikolaou/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/data/knikolaou/huggingface")

import time

import torch

from vatis import analyze
from vatis.models.hf import load_hf_model

MODEL = "EleutherAI/pythia-14m"
REVISION = "step3000"  # one of the published Pythia checkpoint tags
BATCH_SIZE = 8
SEQ_LEN = 64
N_HUTCHINSON = 16
SEED = 0


def make_synthetic_batch(vocab_size: int, batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[:, :-1] = input_ids[:, 1:]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, model={MODEL}, revision={REVISION}")
    print(f"batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, n_hutchinson={N_HUTCHINSON}")

    # ---- model load ----
    t0 = time.perf_counter()
    bundle = load_hf_model(MODEL, revision=REVISION, dtype="fp32", device=device)
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in bundle.params)
    print(f"model load wallclock: {load_s:.1f}s, n_params: {n_params:,}")

    # Build a synthetic batch (same vocab as the model).
    vocab_size = bundle.model.config.vocab_size
    batch = make_synthetic_batch(vocab_size, BATCH_SIZE, SEQ_LEN)

    # ---- one analyze call per chi_net_method ----
    for method in ("hutchinson", "per_sequence_cv"):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        results = analyze(
            model=bundle,
            revisions=[REVISION],
            eval_batches={"probe": batch},
            chi_net_method=method,
            n_hutchinson=N_HUTCHINSON,
            micro_batch_size=BATCH_SIZE,
            seed=SEED,
            device=device,
            dtype="fp32",
            sink=None,
        )
        wall_s = time.perf_counter() - t0
        peak_mb = (
            torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        )
        rows = results[0].rows
        by_obs = {r.observable: r.value for r in rows}
        print(f"\n--- method={method} ---")
        print(f"wallclock: {wall_s:.2f}s, peak GPU mem: {peak_mb:.1f} MB")
        for o in (
            "chi_loss_normalized",
            "chi_net_normalized",
            "delta_loss",
            "chi_pos",
        ):
            print(f"  {o}: {by_obs[o]:.6e}")


if __name__ == "__main__":
    main()
