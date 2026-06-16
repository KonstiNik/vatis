"""Post-process the DDP scaling JSON records → correctness verdict + tables + plot.

Reads every ``*.json`` written by ``ddp_scaling_worker.py`` and emits:

  1. Correctness: for each config family, compare the W>1 runs against the W=1
     baseline. chi_loss / chi_loss_normalized / delta_loss must match tightly
     (exact under sharding); chi_net / chi_pos within Hutchinson noise.
  2. Strong scaling: fixed total batch → speedup T1/TW and efficiency.
  3. Weak scaling: fixed batch-per-GPU → wallclock should stay ~flat while
     total data (and throughput) grows ~linearly.

No GPU, no vatis import — just reads the JSON contract (like analyze_results.py).

    .venv/bin/python examples/benchmark/ddp_scaling_analyze.py \\
        --in-dir examples/benchmark/ddp_scaling_out \\
        --out examples/benchmark/DDP_SCALING_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# tolerances
TIGHT_REL = 1e-3  # chi_loss / delta_loss are exact under sharding (bf16 slack)
NOISE_REL = 0.10  # chi_net / chi_pos are Hutchinson estimates


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(1e-30, abs(b))


def load_records(in_dir: Path) -> list[dict]:
    recs = [json.loads(p.read_text()) for p in sorted(in_dir.glob("*.json"))]
    return recs


def correctness_section(recs: list[dict]) -> list[str]:
    out = ["## Correctness (multi-GPU vs single-GPU)", ""]
    tags = sorted({r["tag"] for r in recs})
    all_ok = True
    for tag in tags:
        fam = [r for r in recs if r["tag"] == tag and r["status"] == "ok"]
        base = next((r for r in fam if r["world_size"] == 1), None)
        if base is None:
            out.append(f"- `{tag}`: no W=1 baseline (skipped)")
            continue
        # Correctness is "same batch, sharded across W -> same result". That
        # only holds when the total batch is FIXED across W (strong scaling).
        # Weak scaling deliberately grows the batch with W, so its observables
        # SHOULD differ — comparing them is meaningless, not a failure.
        batch_totals = sorted({r["batch_total"] for r in fam})
        if len(batch_totals) > 1:
            out.append(
                f"- `{tag}`: total batch varies across W ({batch_totals}) — not a "
                f"same-batch comparison (weak-scaling family); correctness check skipped."
            )
            continue
        b = base["observables"]
        for r in sorted(fam, key=lambda x: x["world_size"]):
            if r["world_size"] == 1:
                continue
            o = r["observables"]
            tight = {
                k: _rel(o[k], b[k])
                for k in ("chi_loss_normalized", "delta_loss")
                if k in o and k in b
            }
            noise = {
                k: _rel(o[k], b[k])
                for k in ("chi_net_normalized", "chi_pos")
                if k in o and k in b
            }
            ok = all(v <= TIGHT_REL for v in tight.values()) and all(
                v <= NOISE_REL for v in noise.values()
            )
            all_ok = all_ok and ok
            verdict = "PASS" if ok else "FAIL"
            tight_s = ", ".join(f"{k}={v:.1e}" for k, v in tight.items())
            noise_s = ", ".join(f"{k}={v:.1e}" for k, v in noise.items())
            out.append(
                f"- `{tag}` W={r['world_size']} vs W=1: **{verdict}** "
                f"(exact: {tight_s} ≤ {TIGHT_REL:.0e}; noise: {noise_s} ≤ {NOISE_REL:.0%})"
            )
    out.append("")
    out.append(f"**Overall: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}**")
    out.append("")
    return out


def strong_section(recs: list[dict]) -> list[str]:
    fam = sorted(
        [r for r in recs if r["tag"] == "strong" and r["status"] == "ok"],
        key=lambda x: x["world_size"],
    )
    if not fam:
        return []
    base = next((r for r in fam if r["world_size"] == 1), fam[0])
    t1 = base["wall_s"]
    out = [
        "## Strong scaling (fixed total batch)",
        "",
        f"Total batch B={base['batch_total']}, S={base['seq_len']}, "
        f"n_h={base['n_hutchinson']}, method={base['method']}, dtype={base['dtype']}.",
        "",
        "| GPUs | B/GPU | wall_s | speedup | efficiency |",
        "|---|---|---|---|---|",
    ]
    for r in fam:
        w = r["world_size"]
        sp = t1 / r["wall_s"]
        out.append(
            f"| {w} | {r['batch_per_rank']} | {r['wall_s']:.2f} | {sp:.2f}× | {sp / w:.0%} |"
        )
    out.append("")
    return out


def weak_section(recs: list[dict]) -> list[str]:
    fam = sorted(
        [r for r in recs if r["tag"] == "weak" and r["status"] == "ok"],
        key=lambda x: x["world_size"],
    )
    if not fam:
        return []
    out = [
        "## Weak scaling (fixed batch-per-GPU — the 'too much data for one GPU' case)",
        "",
        f"Batch-per-GPU fixed at {fam[0]['batch_per_rank']}, "
        f"S={fam[0]['seq_len']}, n_h={fam[0]['n_hutchinson']}, method={fam[0]['method']}.",
        "Ideal: wallclock flat while total data + throughput grow ~linearly.",
        "",
        "| GPUs | total B | wall_s | throughput (seq/s) |",
        "|---|---|---|---|",
    ]
    for r in fam:
        thru = r["batch_total"] / r["wall_s"]
        out.append(f"| {r['world_size']} | {r['batch_total']} | {r['wall_s']:.2f} | {thru:.2f} |")
    out.append("")
    return out


def memory_section(recs: list[dict]) -> list[str]:
    fam = sorted(
        [r for r in recs if r["tag"] == "memrescue"], key=lambda x: x["world_size"]
    )
    if not fam:
        return []
    out = [
        "## Memory rescue (per_sequence_cv: refused on 1 GPU, runs on N)",
        "",
        "per_sequence_cv caches B×n_params×4 bytes of per-sample gradients per rank. "
        "Sharding the batch across N GPUs cuts that by N — a config the single-GPU "
        "memory pre-check refuses can run multi-GPU.",
        "",
        "| GPUs | total B | B/GPU | status | wall_s | peak_mb |",
        "|---|---|---|---|---|---|",
    ]
    for r in fam:
        st = "ok" if r["status"] == "ok" else "refused"
        out.append(
            f"| {r['world_size']} | {r['batch_total']} | {r['batch_per_rank']} | "
            f"{st} | {r['wall_s']:.2f} | {r['peak_mb_rank0']:.0f} |"
        )
    out.append("")
    return out


def make_plot(recs: list[dict], out_png: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    # All three panels use the strong-scaling family (fixed total batch, the
    # only setting where multi-GPU == single-GPU is a meaningful comparison).
    strong = sorted(
        [r for r in recs if r["tag"] == "strong" and r["status"] == "ok"],
        key=lambda x: x["world_size"],
    )
    if not strong:
        return False
    ws = [r["world_size"] for r in strong]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # Panel 1: speedup.
    t1 = next((r["wall_s"] for r in strong if r["world_size"] == 1), strong[0]["wall_s"])
    axes[0].plot(ws, [t1 / r["wall_s"] for r in strong], "o-", label="measured")
    axes[0].plot(ws, ws, "k--", alpha=0.5, label="ideal (linear)")
    axes[0].set_xlabel("GPUs")
    axes[0].set_ylabel("speedup (T1 / TW)")
    axes[0].set_title("Strong scaling (fixed total batch)")
    axes[0].legend()

    # Panel 2: correctness — observables vs GPUs, normalized to the W=1 value.
    # Same batch sharded across W must give the same result: chi_loss and
    # delta_loss are exact (sit on 1.0); chi_net and chi_pos are Hutchinson
    # estimates so they sit within the noise band of 1.0.
    base = next((r for r in strong if r["world_size"] == 1), strong[0])["observables"]
    series = [
        ("chi_loss_normalized", "o", "exact"),
        ("delta_loss", "^", "exact"),
        ("chi_net_normalized", "s", "Hutchinson"),
        ("chi_pos", "D", "Hutchinson"),
    ]
    for key, mk, kind in series:
        if key not in base or base[key] == 0:
            continue
        axes[1].plot(ws, [r["observables"][key] / base[key] for r in strong],
                     marker=mk, ls="-", label=f"{key} ({kind})")
    axes[1].axhline(1.0, color="k", ls="--", alpha=0.5)
    axes[1].set_xlabel("GPUs")
    axes[1].set_ylabel("value / value(W=1)")
    axes[1].set_title("Correctness: observables vs GPUs\n(=1 → identical to single-GPU)")
    axes[1].legend(fontsize=7)

    # Panel 3: per-GPU peak memory — flat, because the model + gradient are
    # replicated on every rank (DDP shards data, not memory).
    mem = [r["peak_mb_rank0"] / 1024.0 for r in strong]
    axes[2].plot(ws, mem, "o-", color="tab:purple")
    axes[2].set_ylim(0, max(mem) * 1.35)
    axes[2].set_xlabel("GPUs")
    axes[2].set_ylabel("peak memory / GPU (GiB)")
    axes[2].set_title("Per-GPU memory (flat → model replicated;\nDDP shards data, not memory)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    recs = load_records(in_dir)
    if not recs:
        raise SystemExit(f"no JSON records found in {in_dir}")

    n_params = recs[0]["n_params"]
    lines = [
        "# Multi-GPU DDP scaling — vatis",
        "",
        f"Model: {n_params:,} params. "
        "Generated by `examples/benchmark/ddp_scaling/ddp_scaling_analyze.py`.",
        "vatis DDP replicates the model on every rank and shards the eval batch "
        "along its first dim (data-parallel over eval data, not model size).",
        "",
    ]
    lines += correctness_section(recs)
    lines += strong_section(recs)
    lines += weak_section(recs)
    lines += memory_section(recs)

    out_png = Path(args.out).with_suffix(".png")
    if make_plot(recs, out_png):
        lines += [f"![scaling]({out_png.name})", ""]

    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()
