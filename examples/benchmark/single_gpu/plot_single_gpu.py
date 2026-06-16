"""Plot the single-GPU sweep results from a results_*.md table.

Reads the markdown table written by ``single_gpu_sweep.py`` and produces a
two-panel figure: wall time vs n_hutchinson (lines per method×B) and peak
memory vs batch size (lines per method). Pure pandas-free parsing + matplotlib,
no GPU, no vatis import — consumes the results table as the contract.

    .venv/bin/python examples/benchmark/plot_single_gpu.py \\
        --results examples/benchmark/results_pythia-160m.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def parse_table(md_path: Path) -> list[dict]:
    rows: list[dict] = []
    cols = ["method", "n_h", "B", "wall_s", "peak_mb"]
    for line in md_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5 or cells[0] in ("method", "---") or cells[0].startswith("---"):
            continue
        if cells[3] in ("", "—"):  # refused config, no timing
            continue
        try:
            rows.append(
                {
                    "method": cells[0],
                    "n_h": int(cells[1]),
                    "B": int(cells[2]),
                    "wall_s": float(cells[3]),
                    "peak_mb": float(cells[4]),
                }
            )
        except ValueError:
            continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    md = Path(args.results)
    rows = parse_table(md)
    if not rows:
        raise SystemExit(f"no data rows parsed from {md}")
    out = Path(args.out) if args.out else md.with_name(md.stem + "_plot.png")

    methods = sorted({r["method"] for r in rows})
    batches = sorted({r["B"] for r in rows})
    cmap = {b: c for b, c in zip(batches, ["tab:blue", "tab:orange", "tab:green", "tab:red"])}
    style = {"hutchinson": "-", "per_sequence_cv": "--"}

    fig, (axt, axm) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: wall_s vs n_h. color = B, linestyle = method.
    for method in methods:
        for B in batches:
            pts = sorted(
                (r for r in rows if r["method"] == method and r["B"] == B),
                key=lambda r: r["n_h"],
            )
            if not pts:
                continue
            axt.plot(
                [p["n_h"] for p in pts],
                [p["wall_s"] for p in pts],
                style.get(method, "-"),
                color=cmap[B],
                marker="o",
            )

    # Slope-1 reference y = c * n_h. On log-log this is an exponent-1 power
    # law; the data is truly *linear* in n_h only if it runs PARALLEL to this.
    # Anchor c to the cheapest measured point so the guide overlays the data.
    n_hs = sorted({r["n_h"] for r in rows})
    anchor = min(rows, key=lambda r: r["wall_s"])
    c = anchor["wall_s"] / anchor["n_h"]
    axt.plot(n_hs, [c * n for n in n_hs], ":", color="black", lw=1.6, alpha=0.7)

    axt.set_xscale("log", base=2)
    axt.set_yscale("log")
    axt.set_xlabel("n_hutchinson")
    axt.set_ylabel("wall time (s)")
    axt.set_title("Wall time vs n_hutchinson")

    # Two legends so solid/dashed (method) and color (B) are both explicit.
    method_handles = [
        Line2D([], [], color="k", ls=style.get(m, "-"), marker="o", label=m) for m in methods
    ]
    method_handles.append(
        Line2D([], [], color="k", ls=":", lw=1.6, alpha=0.7, label="slope 1 (∝ n_h)")
    )
    color_handles = [
        Line2D([], [], color=cmap[b], ls="-", marker="o", label=f"B={b}") for b in batches
    ]
    leg1 = axt.legend(handles=method_handles, fontsize=8, loc="upper left")
    axt.add_artist(leg1)
    axt.legend(handles=color_handles, fontsize=8, loc="lower right")

    # Right: peak_mb vs B, line per method (avg over n_h, which barely matters)
    for method in methods:
        by_b: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            if r["method"] == method:
                by_b[r["B"]].append(r["peak_mb"])
        bs = sorted(by_b)
        axm.plot(
            bs,
            [sum(by_b[b]) / len(by_b[b]) for b in bs],
            style.get(method, "-"),
            marker="s",
            label=method,
        )
    axm.set_xlabel("batch size B")
    axm.set_ylabel("peak memory (MB)")
    axm.set_title("Peak memory vs B\n(hutchinson flat; per_seq_cv ∝ B)")
    axm.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
