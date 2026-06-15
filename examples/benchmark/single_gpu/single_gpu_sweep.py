"""Single-GPU benchmark sweep over (chi_net_method, n_hutchinson, batch_size).

Loads one checkpoint once (so checkpoint loading is amortized) and times the
analyzer across a grid of configs. Synthetic random-token batches are used on
purpose: this measures wallclock + peak memory only, not observable *values*
(see CLAUDE.md "Real text, not random integers" — that rule is for runs that
report observables, not for shape/timing benchmarks).

Prints a markdown table and writes the same table to ``--out``.

Run (single GPU)::

    HF_HOME=/data/horse/ws/koni010i-dpo_sft_transition/hf_cache \\
        .venv/bin/python examples/benchmark/single_gpu_sweep.py \\
        --model EleutherAI/pythia-160m --revision step3000
"""

from __future__ import annotations

import argparse
import os

# Default the HF cache to the workspace scratch (NOT a home dir, NOT someone
# else's cache). Overridable by exporting HF_HOME before launch.
_DEFAULT_HF_HOME = "/data/horse/ws/koni010i-dpo_sft_transition/hf_cache"
os.environ.setdefault("HF_HOME", _DEFAULT_HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
os.environ.setdefault("HF_DATASETS_CACHE", os.environ["HF_HOME"])

import time

import torch

from vatis import analyze
from vatis.models.hf import load_hf_model

SEED = 0


def make_synthetic_batch(vocab_size: int, batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[:, :-1] = input_ids[:, 1:]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def run_one(
    bundle, batch, revision: str, method: str, n_hutchinson: int, micro_batch_size: int
) -> tuple[float, float, dict[str, float]]:
    if torch.cuda.is_available():
        # Release cached blocks from earlier configs so the per_sequence_cv
        # memory pre-check sees the true free memory, not what the allocator
        # is holding onto.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = analyze(
        model=bundle,
        revisions=[revision],
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-160m")
    parser.add_argument("--revision", default="step3000")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--n-hutchinson", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--methods", nargs="+", default=["hutchinson", "per_sequence_cv"])
    parser.add_argument("--out", default=None, help="markdown output path (default: derived from model)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results_{args.model.split('/')[-1]}.md",
    )

    t0 = time.perf_counter()
    bundle = load_hf_model(args.model, revision=args.revision, dtype="fp32", device=device)
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in bundle.params)
    vocab = bundle.model.config.vocab_size

    header = [
        f"# single-GPU benchmark: {args.model}@{args.revision}",
        "",
        f"- device: `{device}`"
        + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""),
        f"- n_params: {n_params:,}",
        f"- seq_len: {args.seq_len}, dtype: fp32, seed: {SEED}",
        f"- model_load_s: {load_s:.1f}",
        "",
        "| method | n_h | B | wall_s | peak_mb | chi_loss | chi_net | delta_loss | chi_pos |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines = list(header)
    for line in header:
        print(line)

    # CUDA warmup so the first measured row isn't biased by lazy init.
    warm = make_synthetic_batch(vocab, 4, args.seq_len)
    _ = run_one(bundle, warm, args.revision, "hutchinson", 4, 4)

    for batch_size in args.batch_sizes:
        batch = make_synthetic_batch(vocab, batch_size, args.seq_len)
        for method in args.methods:
            for n_h in args.n_hutchinson:
                try:
                    wall, peak, obs = run_one(
                        bundle, batch, args.revision, method, n_h, batch_size
                    )
                    row = (
                        f"| {method} | {n_h} | {batch_size} | {wall:.2f} | {peak:.0f} | "
                        f"{obs['chi_loss_normalized']:.4e} | {obs['chi_net_normalized']:.4e} | "
                        f"{obs['delta_loss']:.4e} | {obs['chi_pos']:.4e} |"
                    )
                except Exception as e:  # noqa: BLE001 — record + continue the sweep
                    row = (
                        f"| {method} | {n_h} | {batch_size} | — | — | "
                        f"refused: {type(e).__name__} | | | |"
                    )
                print(row, flush=True)
                lines.append(row)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n# wrote {out_path}")


if __name__ == "__main__":
    main()
