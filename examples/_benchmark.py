"""Benchmark sweep for the deployment example sizing.

Scans (chi_net_method, n_hutchinson, batch_size) on a single pythia-14m
revision and prints a markdown table that's pasted into BENCHMARK.md.
Single GPU, single revision — checkpoint loading is amortized across
all configurations so the per-call timings reflect compute only.

Run with::

    HF_HOME=/data/knikolaou/huggingface .venv/bin/python examples/_benchmark.py
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/data/knikolaou/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/knikolaou/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/data/knikolaou/huggingface")

import time

import torch

from vatis import analyze
from vatis.models.hf import load_hf_model

MODEL = "EleutherAI/pythia-14m"
REVISION = "step3000"
SEQ_LEN = 64
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


def run_one(
    bundle, batch, method: str, n_hutchinson: int, micro_batch_size: int
) -> tuple[float, float, dict[str, float]]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = analyze(
        model=bundle,
        revisions=[REVISION],
        eval_batches={"probe": batch},
        chi_net_method=method,
        n_hutchinson=n_hutchinson,
        micro_batch_size=micro_batch_size,
        seed=SEED,
        device=next(bundle.model.parameters()).device,
        dtype="fp32",
        sink=None,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    by_obs = {r.observable: r.value for r in results[0].rows}
    return wall, peak_mb, by_obs


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    bundle = load_hf_model(MODEL, revision=REVISION, dtype="fp32", device=device)
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in bundle.params)
    print(f"# benchmark: {MODEL}@{REVISION}")
    print(f"# device: {device}, n_params: {n_params:,}, model_load_s: {load_s:.1f}")
    print()

    vocab = bundle.model.config.vocab_size

    # CUDA warmup so the first row isn't biased.
    warm_batch = make_synthetic_batch(vocab, 4, SEQ_LEN)
    _ = run_one(bundle, warm_batch, "hutchinson", 4, 4)

    print(
        "| method | n_h | B | wall_s | peak_mb | chi_loss | chi_net | delta_loss | chi_pos |"
    )
    print("|---|---|---|---|---|---|---|---|---|")

    for batch_size in (4, 8, 16):
        batch = make_synthetic_batch(vocab, batch_size, SEQ_LEN)
        for method in ("hutchinson", "per_sequence_cv"):
            for n_h in (8, 32, 128):
                wall, peak, obs = run_one(bundle, batch, method, n_h, batch_size)
                print(
                    f"| {method} | {n_h} | {batch_size} | {wall:.2f} | {peak:.0f} | "
                    f"{obs['chi_loss_normalized']:.4e} | "
                    f"{obs['chi_net_normalized']:.4e} | "
                    f"{obs['delta_loss']:.4e} | "
                    f"{obs['chi_pos']:.4e} |"
                )


if __name__ == "__main__":
    main()
