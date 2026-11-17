"""Which update rule reproduces the step an AdamW run took? Step 2 of 2: tables + figure.

Reads ``results.parquet`` and ``run_summary.json`` as written by
``adamw_vs_sgd_reconstruction.py`` and writes ``table.md`` and
``reconstruction.png`` next to them. It imports no vatis code: the parquet file
is the interface. What the columns mean is explained in
``adamw_vs_sgd_reconstruction/README.md``.

    python examples/adamw_vs_sgd_reconstruction_evaluate.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — no display backend

import matplotlib.pyplot as plt  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DEFAULT_DIR = Path(__file__).parent / "adamw_vs_sgd_reconstruction"
UPDATE, HELD_OUT = "update_batch", "held_out"
BATCHES = {UPDATE: "update batch", HELD_OUT: "held-out batch"}
PARTS = {"grad": "gradient", "hist": "momentum", "wd": "weight decay"}

# One colour and one marker per prediction (key, legend label, colour, marker).
SERIES = (
    ("rule", "AdamW rule", "#2a78d6", "o"),
    ("sgd", r"SGD rule: $-\eta\,c\;\nabla L_B\cdot\nabla L_A$", "#eb6834", "s"),
    ("dlp", r"$-\eta\,\delta L_P$: $-\eta\;\nabla L_B\cdot P_t\,\nabla L_A$", "#1baf7a", "^"),
)
INK, MUTED, GRID, SURFACE = "#2b2b29", "#6f6e68", "#e4e3de", "#fcfcfb"


def compare(path: Path) -> list[dict[str, Any]]:
    """One record per (step, batch); predictions and parts as ratios (see the tables)."""
    rows = {
        (int(r["revision"].removeprefix("step")), r["batch_a"], r["batch_b"], r["observable"]): r[
            "value"
        ]
        for r in pq.read_table(path).to_pylist()
    }
    records = []
    for step in sorted({key[0] for key in rows}):
        lr = rows[(step, UPDATE, UPDATE, "update_lr")]
        clip = rows[(step, UPDATE, UPDATE, "update_clip_factor")]
        # delta_loss(A, B) is the dot product of the two batches' gradients.
        norms = (
            rows[(step, UPDATE, UPDATE, "delta_loss")]
            * rows[(step, HELD_OUT, HELD_OUT, "delta_loss")]
        )
        cosine = rows[(step, UPDATE, HELD_OUT, "delta_loss")] / math.sqrt(norms)
        for batch in BATCHES:
            pair = (step, UPDATE, batch)
            truth = rows[(*pair, "loss_change_first_order_weights")]
            rule = rows[(*pair, "loss_change_first_order")]
            p_dot = rows.get((*pair, "delta_loss_P"))  # absent at step 0: no P yet
            records.append(
                {
                    "step": step,
                    "batch": batch,
                    "lr": lr,
                    "clip": clip,
                    "cosine": cosine,
                    "truth": truth,
                    # the three predictions and the realized change, each / truth
                    "rule": rule / truth,
                    "sgd": -lr * clip * rows[(*pair, "delta_loss")] / truth,
                    "dlp": None if p_dot is None else -lr * p_dot / truth,
                    "realized": rows[(*pair, "loss_change")] / truth,
                    # the three parts of the AdamW rule, each / their sum
                    **{p: rows[(*pair, f"loss_change_first_order_{p}")] / rule for p in PARTS},
                }
            )
    return records


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3g}"


def make_tables(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    update_shape, held_shape = summary["update_batch_shape"], summary["held_out_shape"]
    lines = [
        f"Run `{summary['bundle']}` ({summary['optimizer']}, {summary['n_parameters']:,} "
        f"parameters, gradient clipping `{summary['clip_grad_norm']}`), fp32 on "
        f"{summary['device']}. Update batch {update_shape[0]} × {update_shape[1]} tokens, "
        f"held-out batch {held_shape[0]} × {held_shape[1]} tokens. Symbols as in the README.",
        "",
        "**Each prediction as a ratio to the truth** (the AdamW rule as ratio − 1):",
        "",
        "| step | batch | truth | AdamW rule − 1 | SGD rule | −η·δL_P | realized / truth |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['step']} | {BATCHES[r['batch']]} | {r['truth']:+.3e} | "
            f"{r['rule'] - 1:+.1e} | {fmt(r['sgd'])} | {fmt(r['dlp'])} | {r['realized']:.3f} |"
        )
    lines += [
        "",
        "**Per step.** η and c are the rows `update_lr` and `update_clip_factor`; the cosine "
        "is between the gradients of the two batches; the last two columns are the ratio on "
        "the held-out batch divided by the ratio on the update batch (1 = one rescaling "
        "would repair the prediction on both batches):",
        "",
        "| step | η | c | cosine | SGD rule: held-out ÷ update | −η·δL_P: held-out ÷ update |",
        "|---|---|---|---|---|---|",
    ]
    by_step: dict[int, dict[str, dict[str, Any]]] = {}
    for r in records:
        by_step.setdefault(r["step"], {})[r["batch"]] = r
    for step, pair in by_step.items():
        own, other = pair[UPDATE], pair[HELD_OUT]
        spread = [None if own[key] is None else other[key] / own[key] for key in ("sgd", "dlp")]
        lines.append(
            f"| {step} | {own['lr']:.3g} | {own['clip']:.3f} | {own['cosine']:.2f} | "
            f"{fmt(spread[0])} | {fmt(spread[1])} |"
        )
    lines += [
        "",
        "**The three parts of the AdamW rule**, as fractions of their sum (rows "
        "`loss_change_first_order_grad`, `_hist`, `_wd`):",
        "",
        "| step | batch | gradient | momentum | weight decay |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        # "+ 0.0" prints a rounded-away negative as 0.000, not -0.000
        parts = " | ".join(
            f"{round(r[p], 3) + 0.0:.3f}" if p != "wd" else f"{r[p]:.1e}" for p in PARTS
        )
        lines.append(f"| {r['step']} | {BATCHES[r['batch']]} | {parts} |")
    return "\n".join(lines) + "\n"


def make_figure(records: list[dict[str, Any]], path: Path) -> None:
    steps = sorted({r["step"] for r in records})
    position = {step: index for index, step in enumerate(steps)}
    peak = max(r["lr"] for r in records)
    warm_up_end = max(position[r["step"]] for r in records if r["lr"] < peak) + 0.5
    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.4), sharey=True, facecolor=SURFACE, layout="constrained"
    )
    for axis, (batch, label) in zip(axes, BATCHES.items(), strict=True):
        axis.set_facecolor(SURFACE)
        axis.axhline(1.0, color=MUTED, linewidth=1.0, zorder=1)
        axis.axvline(warm_up_end, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        axis.annotate(
            "end of warm-up",
            (warm_up_end, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -3),
            textcoords="offset points",
            va="top",
            color=MUTED,
            fontsize=8,
        )
        for key, _name, colour, marker in SERIES:
            points = [
                (position[r["step"]], r[key])
                for r in records
                if r["batch"] == batch and r[key] is not None
            ]
            xs, ys = [x for x, _ in points], [abs(y) for _, y in points]
            axis.plot(xs, ys, color=colour, linewidth=1.2, alpha=0.55, zorder=2)
            # A log axis cannot show a negative ratio: drawn hollow at |ratio|.
            for x, y in points:
                axis.plot(
                    [x],
                    [abs(y)],
                    marker=marker,
                    markersize=8,
                    markerfacecolor=SURFACE if y < 0 else colour,
                    markeredgecolor=colour,
                    markeredgewidth=1.6,
                    zorder=3,
                )
        axis.set_yscale("log")
        axis.set_xticks(range(len(steps)))
        axis.set_xticklabels([str(step) for step in steps], rotation=45, ha="right")
        axis.set_xlabel("training step (not to scale)", color=INK)
        axis.set_title(f"on the {label}", color=INK, fontsize=11, loc="left")
        axis.grid(True, which="major", axis="y", color=GRID, linewidth=0.8)
        axis.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
    axes[0].set_ylabel("prediction / truth  (1 = exact)", color=INK)
    handles = [
        Line2D([], [], color=colour, marker=marker, markersize=8, linewidth=1.2, label=name)
        for _key, name, colour, marker in SERIES
    ]
    hollow = {"markerfacecolor": SURFACE, "markeredgecolor": MUTED, "markeredgewidth": 1.6}
    label = "hollow: wrong sign, drawn at |ratio|"
    handles.append(
        Line2D([], [], linestyle="none", marker="s", markersize=8, label=label, **hollow)
    )
    axes[1].legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    title = "Which update rule reproduces the step an AdamW run took?"
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    records = compare(args.dir / "results.parquet")
    summary = json.loads((args.dir / "run_summary.json").read_text())
    (args.dir / "table.md").write_text(make_tables(records, summary))
    make_figure(records, args.dir / "reconstruction.png")
    print(f"wrote {args.dir / 'table.md'} and {args.dir / 'reconstruction.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
