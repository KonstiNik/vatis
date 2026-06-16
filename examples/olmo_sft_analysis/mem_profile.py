"""GPU memory profiler for the OLMo SFT analysis (diagnostic).

Runs the **real vatis code paths** (no replica) on tiny synthetic batches and
reports peak GPU memory for each step, across a small `(B, S)` grid, so we can
see exactly where the memory goes and how it scales with batch/sequence:

- model weights (resident after load),
- the analyzer's ``delta_loss`` gradient accumulation (forward + backward +
  the fp32 flat gradient vector),
- ``chi_net`` via ``hutchinson`` and ``per_sequence_cv``.

It also prints the actual gradient-tuple dtype/size — the number that decides
whether the param-bound floor is ~58 GB (bf16 grads) or ~73 GB (fp32 grads).

Synthetic random tokens on purpose: this measures memory only, not observable
values (see CLAUDE.md). Run on a GPU node (offline, model from cache):

    .venv/bin/python examples/olmo_sft_analysis/mem_profile.py
    # or via SLURM:
    examples/benchmark/submit.sh examples/olmo_sft_analysis/mem_profile.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from run_config import configure_hf_cache  # noqa: E402

configure_hf_cache()

import torch  # noqa: E402

from vatis.analyzer import _add_grads_into_flat, _zero_param_vector  # noqa: E402
from vatis.core.chi_net.hutchinson import HutchinsonEstimator  # noqa: E402
from vatis.core.chi_net.per_seq_cv import PerSequenceControlVariateEstimator  # noqa: E402
from vatis.data.collate import iter_micro_batches  # noqa: E402
from vatis.models.hf import load_hf_model  # noqa: E402

_GB = 1e9
SAMPLES_PATH = _HERE.parent / "samples.json"


def gpu_now_gb() -> float:
    return torch.cuda.memory_allocated() / _GB


def reset_peak() -> None:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_gb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / _GB


def make_batch(vocab: int, b: int, s: int, device: torch.device) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab, (b, s), generator=g).to(device)
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def run_delta_loss(bundle, batch, micro_bs: int) -> tuple[float, str, float]:
    """Replicate the analyzer's delta_loss gradient accumulation. Returns
    (peak_gb, grad_dtype, grad_tuple_gb)."""
    device = next(bundle.model.parameters()).device
    params = bundle.params
    reset_peak()
    flat = _zero_param_vector(params, device)
    grad_dtype = "?"
    grad_bytes = 0
    n_micros = max(1, (batch["input_ids"].shape[0] + micro_bs - 1) // micro_bs)
    for _start, _stop, micro in iter_micro_batches(batch, micro_bs):
        logits = bundle.forward_fn(bundle.model, micro)
        loss = bundle.loss_fn(logits, micro)
        grads = torch.autograd.grad(loss / n_micros, params, retain_graph=False, allow_unused=True)
        if grad_dtype == "?":
            present = [g for g in grads if g is not None]
            grad_dtype = str(present[0].dtype) if present else "none"
            grad_bytes = sum(g.numel() * g.element_size() for g in present)
        _add_grads_into_flat(flat, grads, params)
        del logits, loss, grads
    pk = peak_gb()
    del flat
    return pk, grad_dtype, grad_bytes / _GB


def run_chi_net(bundle, batch, micro_bs: int, method: str, n_h: int) -> float:
    device = next(bundle.model.parameters()).device
    if method == "hutchinson":
        est = HutchinsonEstimator(n_hutchinson=n_h)
    else:
        est = PerSequenceControlVariateEstimator(n_hutchinson=n_h)
    gen = torch.Generator(device=device).manual_seed(0)
    reset_peak()
    est.compute(
        bundle.model,
        batch,
        bundle.forward_fn,
        loss_fn=bundle.loss_fn,
        valid_mask_fn=bundle.valid_mask_fn,
        params=bundle.params,
        micro_batch_size=micro_bs,
        generator=gen,
    )
    return peak_gb()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="default: model from samples.json")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 10])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[64, 256, 512])
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--n-hutchinson", type=int, default=8)
    parser.add_argument("--methods", nargs="+", default=["hutchinson", "per_sequence_cv"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device — run on a GPU node.")
    device = torch.device("cuda")
    model_name = args.model or json.loads(SAMPLES_PATH.read_text())["meta"]["model"]

    total_gb = torch.cuda.get_device_properties(0).total_memory / _GB
    print(f"# mem_profile — {model_name}")
    print(f"device: {torch.cuda.get_device_name(0)}  total={total_gb:.1f} GB")
    print(f"micro_batch_size={args.micro_batch_size}, n_hutchinson={args.n_hutchinson}\n")

    reset_peak()
    bundle = load_hf_model(model_name, dtype="bf16", device=device)
    p = sum(x.numel() for x in bundle.params)
    weights_gb = gpu_now_gb()
    print(f"n_params={p:,}")
    print(f"weights resident after load: {weights_gb:.1f} GB (bf16 theory {p * 2 / _GB:.1f})")
    print(f"flat_grad fp32 (theory):     {p * 4 / _GB:.1f} GB\n")

    vocab = bundle.model.config.vocab_size
    header = "| B | S | delta_loss peak | grad dtype | grad tuple | hutch peak | pseqcv peak |"
    print(header)
    print("|---|---|---|---|---|---|---|")
    rows = []
    for b in args.batch_sizes:
        for s in args.seq_lens:
            batch = make_batch(vocab, b, s, device)
            try:
                d_pk, gdt, g_gb = run_delta_loss(bundle, batch, args.micro_batch_size)
                d_str = f"{d_pk:.1f} GB"
            except torch.OutOfMemoryError:
                d_pk, gdt, g_gb, d_str = float("nan"), "OOM", float("nan"), "OOM"
                torch.cuda.empty_cache()
            peaks = {}
            for m in args.methods:
                try:
                    peaks[m] = (
                        f"{run_chi_net(bundle, batch, args.micro_batch_size, m, args.n_hutchinson):.1f} GB"
                    )
                except torch.OutOfMemoryError:
                    peaks[m] = "OOM"
                    torch.cuda.empty_cache()
            row = (
                f"| {b} | {s} | {d_str} | {gdt} | {g_gb:.1f} GB | "
                f"{peaks.get('hutchinson', '-')} | {peaks.get('per_sequence_cv', '-')} |"
            )
            print(row, flush=True)
            rows.append(row)

    out = _HERE.parent / "mem_profile.md"
    with open(out, "w") as f:
        f.write(f"# mem_profile — {model_name}\n\n")
        f.write(f"- device: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)\n")
        f.write(
            f"- n_params: {p:,}; weights {weights_gb:.1f} GB; flat_grad fp32 {p * 4 / _GB:.1f} GB\n"
        )
        f.write(f"- micro_batch_size={args.micro_batch_size}, n_hutchinson={args.n_hutchinson}\n\n")
        f.write(header + "\n|---|---|---|---|---|---|---|\n")
        f.write("\n".join(rows) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
