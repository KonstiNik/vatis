"""Which update rule reproduces the step an AdamW run took? Step 2 of 2: table + figures.

Reads what ``examples/adamw_vs_sgd_reconstruction/run.py`` wrote and produces

- ``<dir>/table.md``            every prediction as a ratio to the truth, per step and
                                batch; the spread between the two batches; the parts
- ``<dir>/table_full.md``       the wide table: every prediction as ratio (4–7 digits)
                                and difference to the truth, the unclipped SGD rule,
                                the realized change, and the spread of every rule
- ``<dir>/reconstruction.png``  the ratio prediction / truth, one panel per batch
- ``<dir>/update_parts.png``    the optimizer's rule split into its gradient,
                                momentum and weight-decay parts, one panel per batch

**This script imports no vatis code** — ``pyarrow``, ``matplotlib`` and the
standard library only; the parquet is the contract.

Definitions (``A`` the step's update batch, ``B`` the batch a row is about,
``lr`` and ``clip`` the step's ``update_lr`` and ``update_clip_factor`` rows):

    truth              loss_change_first_order_weights(A, B)
                       = grad L^B . (theta_{t+1} - theta_t), two stored weight files
    optimizer's rule   loss_change_first_order(A, B)
    SGD rule           -lr * clip * delta_loss(A, B)   plain SGD on the gradient the
                       optimizer was handed: no momentum, no weight decay
    SGD rule, unclipped  -lr * delta_loss(A, B)        the same on the raw gradient
    -lr * delta_loss_P -lr * delta_loss_P(A, B)        pre-step P, no momentum, no decay
    realized           loss_change(A, B) = L_B(theta_{t+1}) - L_B(theta_t)

Each prediction is reported as a ratio to the truth, to three significant
digits. A ratio of 1 is a perfect reconstruction of the *first-order* change;
how far the first-order change itself is from the realized one is the last
column of the first table. The unclipped SGD rule is the SGD ratio divided by
``clip`` (listed per step); differences from the truth follow from the ratio and
the truth column.

    python examples/adamw_vs_sgd_reconstruction/evaluate.py \\
        --dir examples/adamw_vs_sgd_reconstruction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — no display backend

import matplotlib.pyplot as plt  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

DEFAULT_DIR = Path(__file__).parent
UPDATE_BATCH = "update_batch"
BATCHES = {UPDATE_BATCH: "the step's own update batch", "held_out": "a held-out batch"}
BATCH_SHORT = {UPDATE_BATCH: "update", "held_out": "held-out"}

# One hue and one marker per prediction, in a fixed order; the palette was
# checked for colour-vision-deficiency separation, and identity is never carried
# by colour alone (marker, legend and the table).
SERIES = {
    "rule": ("optimizer's rule (loss_change_first_order)", "#2a78d6", "o"),
    "sgd": ("SGD rule: −lr · clip · ∇L^B·∇L^A", "#eb6834", "s"),
    "dlp": ("−lr · delta_loss_P (pre-step P)", "#1baf7a", "^"),
}
# The parts of the optimizer's rule: (row suffix, legend label, colour, marker).
PARTS = (
    ("grad", "gradient part (_grad)", "#2a78d6", "o"),
    ("hist", "momentum part (_hist)", "#eb6834", "s"),
    ("wd", "weight-decay part (_wd)", "#1baf7a", "^"),
)
INK = "#2b2b29"
MUTED = "#6f6e68"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"


def step_of(revision: str) -> int:
    return int(revision.removeprefix("step"))


def load(directory: Path) -> tuple[dict[tuple[int, str, str, str], float], dict[str, Any]]:
    table = pq.read_table(directory / "results.parquet").to_pylist()
    rows = {
        (step_of(r["revision"]), r["batch_a"], r["batch_b"], r["observable"]): r["value"]
        for r in table
    }
    summary = json.loads((directory / "run_summary.json").read_text())
    return rows, summary


def compare(rows: dict[tuple[int, str, str, str], float]) -> list[dict[str, Any]]:
    steps = sorted({key[0] for key in rows})
    out = []
    for step in steps:
        own = (step, UPDATE_BATCH, UPDATE_BATCH)
        if (*own, "update_lr") not in rows:
            raise SystemExit(
                f"step {step}: no update_lr row. With different learning rates per parameter "
                "group there is no single 'same learning rate' for an SGD rule to use."
            )
        lr = rows[(*own, "update_lr")]
        clip = rows[(*own, "update_clip_factor")]
        for batch in BATCHES:
            pair = (step, UPDATE_BATCH, batch)
            if (*pair, "loss_change_first_order_weights") not in rows:
                continue
            truth = rows[(*pair, "loss_change_first_order_weights")]
            # delta_loss(A, A) is the self row; delta_loss(A, B) the cross row.
            dot = rows[(*pair, "delta_loss")]
            p_dot = rows.get((*pair, "delta_loss_P"))
            out.append(
                {
                    "step": step,
                    "batch": batch,
                    "lr": lr,
                    "clip": clip,
                    "truth": truth,
                    "rule": rows[(*pair, "loss_change_first_order")],
                    "sgd": -lr * clip * dot,
                    "sgd_unclipped": -lr * dot,
                    "dlp": None if p_dot is None else -lr * p_dot,
                    "realized": rows[(*pair, "loss_change")],
                    **{
                        part: rows[(*pair, f"loss_change_first_order_{part}")] for part, *_ in PARTS
                    },
                }
            )
    return out


def ratio(prediction: float | None, truth: float) -> str:
    return "—" if prediction is None else f"{prediction / truth:.3g}"


def make_table(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    header = (
        f"Bundle `{summary['bundle']}` ({summary['optimizer']}, rows tagged "
        f"`{', '.join(summary['update_rule'])}`), {summary['n_parameters']:,} parameters, "
        f"clipping stated as `{summary['clip_grad_norm']}`, fp32 on {summary['device_name']}. "
        f"Update batch {summary['update_batch_shape']} tokens, held-out batch "
        f"`{summary['held_out_batch']}` {summary['held_out_shape']} tokens. Steps "
        f"{summary['steps']} of the {len(summary['steps_with_stored_successor'])} that have "
        f"their successor stored. The truth is `loss_change_first_order_weights`; the "
        f"optimizer's rule is `loss_change_first_order`.\n\n"
    )
    lines = [
        "**Each prediction as a ratio to the truth** (the optimizer's rule as ratio − 1):",
        "",
        "| step | batch | truth | optimizer's rule − 1 | SGD rule | −lr·δL_P | realized |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['step']} | {BATCH_SHORT[r['batch']]} | {r['truth']:+.3g} | "
            f"{r['rule'] / r['truth'] - 1:+.0e} | {ratio(r['sgd'], r['truth'])} | "
            f"{ratio(r['dlp'], r['truth'])} | {ratio(r['realized'], r['truth'])} |"
        )
    return header + "\n".join(lines) + "\n\n" + scale_table(records) + "\n" + parts_table(records)


def scale_table(records: list[dict[str, Any]]) -> str:
    """Is a prediction off by one factor, or by a different one on each batch?

    A prediction that had the right direction and the wrong size would be off
    by the same factor on every batch, and a rescaled learning rate would repair
    it. The table gives, per step, the ratio to the truth on the held-out batch
    divided by the ratio on the update batch: 1 means "one factor". (For the
    optimizer's rule it is 1 to within the "− 1" column above.)
    """
    by_step: dict[int, dict[str, dict[str, Any]]] = {}
    for r in records:
        by_step.setdefault(r["step"], {})[r["batch"]] = r
    lines = [
        "**Per step**: the learning rate, the clip factor, and the ratio on the held-out "
        "batch divided by the ratio on the update batch (1 = off by the same factor on "
        "both, i.e. repairable by rescaling the step):",
        "",
        "| step | lr | clip | SGD rule | −lr·δL_P |",
        "|---|---|---|---|---|",
    ]
    for step, pair in sorted(by_step.items()):
        if len(pair) < 2:
            continue
        own, other = pair[UPDATE_BATCH], pair[next(b for b in pair if b != UPDATE_BATCH)]

        def spread(key: str, own: dict[str, Any] = own, other: dict[str, Any] = other) -> str:
            if own[key] is None or other[key] is None:
                return "—"
            return f"{(other[key] / other['truth']) / (own[key] / own['truth']):.3g}"

        lines.append(
            f"| {step} | {own['lr']:.3g} | {own['clip']:.3f} | {spread('sgd')} | {spread('dlp')} |"
        )
    return "\n".join(lines) + "\n"


def parts_table(records: list[dict[str, Any]]) -> str:
    """The parts of the optimizer's rule as fractions of their sum."""
    lines = [
        "**The parts of the optimizer's rule** as fractions of their sum (rows "
        "`loss_change_first_order_grad`, `_hist`, `_wd`):",
        "",
        "| step | batch | gradient | momentum | weight decay |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        # "+ 0.0" prints a rounded-away negative as 0.00, not -0.00
        grad, hist, wd = (r[part] / r["rule"] for part, *_ in PARTS)
        lines.append(
            f"| {r['step']} | {BATCH_SHORT[r['batch']]} | {round(grad, 2) + 0.0:.2f} | "
            f"{round(hist, 2) + 0.0:.2f} | {wd:.0e} |"
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------- table_full.md (the wide table)


def cell(prediction: float | None, truth: float) -> str:
    if prediction is None:
        return "—"
    ratio = prediction / truth
    # Seven decimals near one, where four significant digits would print "1".
    shown = f"{ratio:.7f}" if abs(ratio - 1.0) < 1e-3 else f"{ratio:.4g}"
    return f"{shown} ({prediction - truth:+.2e})"


def make_full_table(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """``table_full.md``: the wide table, every prediction with 4–7 digits and its difference."""
    clipped = any(r["clip"] < 1.0 for r in records)
    head = (
        "| step | batch | lr | clip factor | truth | optimizer's rule | SGD rule |"
        + (" SGD rule, unclipped |" if clipped else "")
        + " −lr · delta_loss_P | realized loss change | realized − truth |"
    )
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in records:
        lines.append(
            f"| {r['step']} | {r['batch']} | {r['lr']:.3g} | {r['clip']:.3f} | "
            f"{r['truth']:+.4e} | {cell(r['rule'], r['truth'])} | {cell(r['sgd'], r['truth'])} |"
            + (f" {cell(r['sgd_unclipped'], r['truth'])} |" if clipped else "")
            + f" {cell(r['dlp'], r['truth'])} | {r['realized']:+.4e} | "
            f"{r['realized'] - r['truth']:+.2e} |"
        )
    header = (
        f"Bundle `{summary['bundle']}` ({summary['optimizer']}, rows tagged "
        f"`{', '.join(summary['update_rule'])}`), {summary['n_parameters']:,} parameters, "
        f"clipping stated as `{summary['clip_grad_norm']}`, fp32 on {summary['device_name']}. "
        f"Update batch {summary['update_batch_shape']} tokens, held-out batch "
        f"`{summary['held_out_batch']}` {summary['held_out_shape']} tokens. Steps "
        f"{summary['steps']} of the {len(summary['steps_with_stored_successor'])} that have "
        f"their successor stored.\n\n"
        "Predictions are `ratio to the truth (prediction − truth)`; the truth is "
        "`loss_change_first_order_weights`.\n\n"
    )
    return header + "\n".join(lines) + "\n\n" + full_scale_table(records)


def full_scale_table(records: list[dict[str, Any]]) -> str:
    """Is a prediction off by one factor, or by a different one on each batch?

    A prediction that had the right direction and the wrong size would be off
    by the same factor on every batch, and a rescaled learning rate would repair
    it. The table gives, per step, the ratio to the truth on the held-out batch
    divided by the ratio on the update batch: 1 means "one factor".
    """
    by_step: dict[int, dict[str, dict[str, Any]]] = {}
    for r in records:
        by_step.setdefault(r["step"], {})[r["batch"]] = r
    lines = [
        "Ratio on the held-out batch divided by the ratio on the update batch "
        "(1 = off by the same factor on both, i.e. repairable by rescaling the step):",
        "",
        "| step | optimizer's rule | SGD rule | −lr · delta_loss_P |",
        "|---|---|---|---|",
    ]
    for step, pair in sorted(by_step.items()):
        if len(pair) < 2:
            continue
        own, other = pair[UPDATE_BATCH], pair[next(b for b in pair if b != UPDATE_BATCH)]

        def spread(key: str, own: dict[str, Any] = own, other: dict[str, Any] = other) -> str:
            if own[key] is None or other[key] is None:
                return "—"
            return f"{(other[key] / other['truth']) / (own[key] / own['truth']):.4g}"

        lines.append(f"| {step} | {spread('rule')} | {spread('sgd')} | {spread('dlp')} |")
    return "\n".join(lines) + "\n"


def make_figure(records: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    steps = sorted({r["step"] for r in records})
    position = {step: index for index, step in enumerate(steps)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, facecolor=SURFACE)
    for axis, (batch, label) in zip(axes, BATCHES.items(), strict=True):
        axis.set_facecolor(SURFACE)
        axis.axhline(1.0, color=MUTED, linewidth=1.0, zorder=1)
        subset = [r for r in records if r["batch"] == batch]
        for key, (name, colour, marker) in SERIES.items():
            points = [
                (position[r["step"]], r[key] / r["truth"]) for r in subset if r[key] is not None
            ]
            # A log axis cannot show a prediction of the wrong sign: such points
            # are drawn hollow at |ratio| and listed in the table with their sign.
            for hollow in (False, True):
                chosen = [(x, abs(y)) for x, y in points if (y < 0) == hollow]
                if not chosen:
                    continue
                axis.plot(
                    [x for x, _ in chosen],
                    [y for _, y in chosen],
                    linestyle="none",
                    marker=marker,
                    markersize=8,
                    markerfacecolor=SURFACE if hollow else colour,
                    markeredgecolor=colour,
                    markeredgewidth=1.6,
                    label=None if hollow else name,
                    zorder=3,
                )
            axis.plot(
                [x for x, _ in points],
                [abs(y) for _, y in points],
                color=colour,
                linewidth=1.2,
                alpha=0.55,
                zorder=2,
            )
        axis.set_yscale("log")
        axis.set_xticks(range(len(steps)))
        axis.set_xticklabels([str(step) for step in steps], rotation=45, ha="right")
        axis.set_xlabel("training step (not to scale)", color=INK)
        axis.set_title(f"on {label}", color=INK, fontsize=11, loc="left")
        axis.grid(True, which="major", axis="y", color=GRID, linewidth=0.8)
        axis.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
    axes[0].set_ylabel(
        "prediction / true first-order change\n(1 = exact; hollow = wrong sign)", color=INK
    )
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.suptitle(
        f"Reconstructing the first-order loss change of the step taken — "
        f"{summary['bundle']} ({summary['optimizer']}, {summary['n_parameters'] / 1e6:.0f}M parameters)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_parts_figure(records: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    """The optimizer's rule split into its parts, each as a fraction of their sum.

    The fractions are signed (on the held-out batch the momentum part often has
    the opposite sign to the sum), so the y axis is symmetric-log: logarithmic
    above 1e-3 in both directions, linear inside that band.
    """
    steps = sorted({r["step"] for r in records})
    position = {step: index for index, step in enumerate(steps)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, facecolor=SURFACE)
    for axis, (batch, label) in zip(axes, BATCHES.items(), strict=True):
        axis.set_facecolor(SURFACE)
        axis.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        axis.axhline(0.0, color=MUTED, linewidth=1.0, zorder=1)
        subset = [r for r in records if r["batch"] == batch]
        for part, name, colour, marker in PARTS:
            xs = [position[r["step"]] for r in subset]
            ys = [r[part] / r["rule"] for r in subset]
            axis.plot(
                xs,
                ys,
                color=colour,
                marker=marker,
                markersize=7,
                linewidth=1.2,
                label=name,
                zorder=3,
            )
        axis.set_yscale("symlog", linthresh=1e-3, linscale=0.5)
        axis.set_xticks(range(len(steps)))
        axis.set_xticklabels([str(step) for step in steps], rotation=45, ha="right")
        axis.set_xlabel("training step (not to scale)", color=INK)
        axis.set_title(f"on {label}", color=INK, fontsize=11, loc="left")
        axis.grid(True, which="major", axis="y", color=GRID, linewidth=0.8)
        axis.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
    axes[0].set_ylabel("part / sum of the parts\n(dashed line: 1 = the whole step)", color=INK)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.suptitle(
        f"The optimizer's rule split into its parts — {summary['bundle']} "
        f"({summary['optimizer']}, {summary['n_parameters'] / 1e6:.0f}M parameters)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    rows, summary = load(args.dir)
    records = compare(rows)
    if not records:
        raise SystemExit("no step with realized rows in the results")
    table = make_table(records, summary)
    (args.dir / "table.md").write_text(table)
    (args.dir / "table_full.md").write_text(make_full_table(records, summary))
    make_figure(records, summary, args.dir / "reconstruction.png")
    make_parts_figure(records, summary, args.dir / "update_parts.png")
    print(table)
    print(
        f"wrote {args.dir / 'table.md'}, {args.dir / 'table_full.md'}, "
        f"{args.dir / 'reconstruction.png'} and "
        f"{args.dir / 'update_parts.png'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
