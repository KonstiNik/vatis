"""Accuracy sweep: chi_net estimator variance vs n_hutchinson (multi-seed).

For a fixed (model, batch), estimate chi_net at a grid of n_hutchinson with many
random seeds, for both methods. The estimator is unbiased, so its accuracy is
governed by variance, which should fall like ~1/sqrt(n_h). per_sequence_cv's
control variate should give LOWER variance than plain hutchinson at the same
n_h. Writes per-(method, n_h, seed) chi_net values to JSON for plotting.

    .venv/bin/python examples/benchmark/accuracy/accuracy_sweep.py \\
        --model EleutherAI/pythia-160m --revision step3000

Set ``HF_HOME`` (in your shell or the repo-root ``.env``) to relocate the
model cache; see ``run_config.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from run_config import configure_hf_cache  # noqa: E402

configure_hf_cache()

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from vatis import analyze  # noqa: E402
from vatis.models.hf import load_hf_model  # noqa: E402


def make_synthetic_batch(vocab: int, b: int, s: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(1234)
    ids = torch.randint(0, vocab, (b, s), generator=g)
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def chi_net_of(bundle, batch, method: str, n_h: int, seed: int, device) -> float:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    res = analyze(
        model=bundle,
        revisions=["r"],
        eval_batches={"b": batch},
        chi_net_method=method,
        n_hutchinson=n_h,
        micro_batch_size=batch["input_ids"].shape[0],
        seed=seed,
        device=device,
        dtype="fp32",
        sink=None,
    )
    return next(r.value for r in res[0].rows if r.observable == "chi_net")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--revision", default="step3000")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument(
        "--batch-file",
        default=None,
        help="real-text batch .pt, sliced to B×seq_len. Default: synthetic random ids.",
    )
    ap.add_argument("--n-h-grid", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--ref-n-h", type=int, default=4096, help="high-n reference (proxy truth)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"device: {device} ({dev_name})")
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "accuracy.json"

    bundle = load_hf_model(args.model, revision=args.revision, dtype="fp32", device=device)
    if args.batch_file:
        full = torch.load(args.batch_file, map_location="cpu", weights_only=True)
        batch = {k: v[: args.batch_size, : args.seq_len].clone() for k, v in full.items()}
        print(f"real-text batch {args.batch_file} sliced to B={args.batch_size}, S={args.seq_len}")
    else:
        batch = make_synthetic_batch(bundle.model.config.vocab_size, args.batch_size, args.seq_len)
        print(f"synthetic random-token batch B={args.batch_size}, S={args.seq_len}")

    # Report the regime: chi_pos near 1 -> loss direction in the bulk (CV should
    # help); near 0 -> tail (CV is ~no-op). This is the quantity that actually
    # predicts whether per_sequence_cv beats hutchinson.
    reg = analyze(
        model=bundle,
        revisions=["r"],
        eval_batches={"b": batch},
        chi_net_method="hutchinson",
        n_hutchinson=256,
        micro_batch_size=args.batch_size,
        seed=0,
        device=device,
        dtype="fp32",
        sink=None,
    )
    regime = {r.observable: r.value for r in reg[0].rows}
    print(
        "regime:",
        {
            k: f"{regime[k]:.4e}"
            for k in ("chi_loss_normalized", "chi_net_normalized", "delta_loss", "chi_pos")
        },
    )

    # High-n reference (low-variance proxy for the true trace).
    ref = chi_net_of(bundle, batch, "hutchinson", args.ref_n_h, seed=99991, device=device)
    print(f"reference chi_net (n_h={args.ref_n_h}): {ref:.6e}")

    records = []
    total_runs = 2 * len(args.n_h_grid) * args.seeds
    pbar = tqdm(total=total_runs, desc="accuracy sweep", unit="run")
    for method in ("hutchinson", "per_sequence_cv"):
        for n_h in args.n_h_grid:
            for seed in range(args.seeds):
                v = chi_net_of(bundle, batch, method, n_h, seed=seed, device=device)
                records.append({"method": method, "n_h": n_h, "seed": seed, "chi_net": v})
                pbar.update(1)
            vals = [r["chi_net"] for r in records if r["method"] == method and r["n_h"] == n_h]
            mean = sum(vals) / len(vals)
            std = (sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)) ** 0.5
            pbar.set_postfix_str(f"{method} n_h={n_h} rel_std={std / mean:.2%}")
            tqdm.write(f"{method:16s} n_h={n_h:4d}  mean={mean:.4e}  rel_std={std / mean:.3%}")
    pbar.close()

    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "revision": args.revision,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "n_params": sum(p.numel() for p in bundle.params),
                "ref_n_h": args.ref_n_h,
                "ref_chi_net": ref,
                "chi_pos": regime["chi_pos"],
                "batch_file": args.batch_file,
                "seeds": args.seeds,
                "records": records,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
