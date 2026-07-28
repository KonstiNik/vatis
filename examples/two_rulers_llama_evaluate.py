"""WP0 GPU end-to-end example, phase 2: the figure and the tables.

Reads what ``examples/two_rulers_llama.py`` wrote and produces the deliverables a
human actually looks at:

- ``examples/output/two_rulers_llama.png`` — ONE figure, two panels:
  ``chi_pos`` and ``chi_pos^P`` versus training step, and versus train loss.
- the printed performance table (wall time per checkpoint, peak GPU memory,
  total runtime) and the epsilon-ceiling diagnostic.

**This script imports no vatis code** — only ``pyarrow``, ``matplotlib`` and the
standard library. That is the point of the compute/analysis split
(``CLAUDE.md`` §Research workflow): the parquet is the contract with downstream
consumers, and the only way to keep a contract honest is to consume it. The one
thing the parquet cannot carry is per-checkpoint *run* metadata (train loss, lr,
the timing table, the epsilon-ceiling measurement) — that is not an observable
and does not belong in the observable schema, so the compute phase writes it to
``run_summary.json`` alongside.

Run with::

    python examples/two_rulers_llama_evaluate.py --dir examples/output

Two things the figure has to get right, because both are real and easy to
misread:

1. **A step-0 checkpoint has raw rows but no P rows.** Pre-step semantics means
   the saved optimizer state at step 0 is empty, so no AdamW ``P`` exists there
   and vatis skips those rows by design. The P curve therefore *starts later*
   than the raw curve. It must not be drawn as a zero, and the gap is called out
   in the figure instead of being left for the reader to notice.
2. **The epsilon ceiling.** ``P = 1/(sqrt(v_hat) + eps)`` puts every coordinate
   the optimizer never moved at ``1/eps`` — unused vocabulary rows, mostly. Those
   dead directions can carry a large share of ``chi_net^P`` and therefore
   suppress ``chi_pos^P``. The convention is frozen (nothing is clipped or
   masked); the share measured for this run is printed and stamped on the figure.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — no display backend

import matplotlib.pyplot as plt  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

DEFAULT_DIR = Path(__file__).parent / "output"

# Hue carries the ruler (the comparison this figure exists for); linestyle and
# marker carry the batch pair. Two hues, so the palette stays trivially
# colourblind-safe, and identity is never colour-alone.
RULERS = {
    "chi_pos": {"color": "#2a78d6", "label": r"$\chi_{\mathrm{pos}}$ (raw)"},
    "chi_pos_P": {"color": "#eb6834", "label": r"$\chi_{\mathrm{pos}}^{P}$ (AdamW)"},
}
# (batch_a, batch_b) -> (linestyle, marker, legend label)
PAIRS: list[tuple[tuple[str, str], tuple[str, str, str]]] = [
    (("B", "B"), ("-", "o", "B self")),
    (("A1", "B"), ("--", "s", "A1 x B")),
    (("A2", "B"), (":", "^", "A2 x B")),
]

GRID_KWARGS = {"color": "#b8b7b2", "linewidth": 0.5, "alpha": 0.5}
TEXT_MUTED = "#52514e"


def load(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parquet = directory / "results.parquet"
    summary_path = directory / "run_summary.json"
    if not parquet.is_file() or not summary_path.is_file():
        raise SystemExit(
            f"missing inputs in {directory}: expected results.parquet and run_summary.json. "
            "Run examples/two_rulers_llama.py first (phase 1, the expensive one)."
        )
    rows = pq.read_table(parquet).to_pylist()
    summary = json.loads(summary_path.read_text())
    return rows, summary


def series_by_step(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[int, float]]:
    """``(observable, batch_a, batch_b) -> {step: value}``."""
    out: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in rows:
        step = step_of(row["checkpoint_id"], row["revision"])
        if step is None:
            continue
        key = (row["observable"], row["batch_a"], row["batch_b"])
        out.setdefault(key, {})[step] = float(row["value"])
    return out


def step_of(checkpoint_id: str, revision: str) -> int | None:
    """Recover the training step from the bundle's own ``step<N>`` labels."""
    for text in (revision, checkpoint_id):
        if not text:
            continue
        marker = text.rfind("step")
        if marker == -1:
            continue
        digits = text[marker + 4 :]
        if digits.isdigit():
            return int(digits)
    return None


def y_scale(values: list[float]) -> tuple[str, dict[str, float]]:
    """Log where the data allows it, symlog only when something is <= 0.

    ``chi_pos`` is signed — a cross pair measures interference and can come out
    negative — so the axis must survive a sign change. But a symlog axis spends
    real estate on its linear band and on a whole negative branch, and this
    observable spans many decades, so symlog is used only when the run actually
    produced a non-positive value. The threshold is then the 10th percentile of
    the magnitudes rather than the smallest one: a single near-zero point must
    not push nine decades of live signal into a sliver of the panel. Points below
    the threshold are still drawn, inside the linear band.
    """
    finite = [v for v in values if math.isfinite(v)]
    magnitudes = sorted(abs(v) for v in finite if v != 0.0)
    if not magnitudes:
        return "linear", {}
    if all(v > 0.0 for v in finite):
        return "log", {}
    low = magnitudes[max(0, int(0.10 * (len(magnitudes) - 1)))]
    return "symlog", {"linthresh": float(10 ** math.floor(math.log10(low)))}


def plot(
    directory: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    series = series_by_step(rows)
    loss_by_step = {
        int(c["step"]): c["train_loss"]
        for c in summary["checkpoints"]
        if c.get("train_loss") is not None
    }

    plotted: list[float] = []
    for observable in RULERS:
        for pair, _style in PAIRS:
            plotted.extend(series.get((observable, *pair), {}).values())
    scale_name, scale_kwargs = y_scale(plotted)

    fig, (ax_step, ax_loss) = plt.subplots(1, 2, figsize=(12.0, 6.2), sharey=True)

    for observable, ruler in RULERS.items():
        for pair, (linestyle, marker, pair_label) in PAIRS:
            points = series.get((observable, *pair), {})
            if not points:
                continue
            steps = sorted(points)
            values = [points[s] for s in steps]
            label = f"{ruler['label']}  {pair_label}"
            ax_step.plot(
                steps,
                values,
                color=ruler["color"],
                linestyle=linestyle,
                marker=marker,
                markersize=5,
                linewidth=1.6,
                label=label,
            )
            losses = [loss_by_step.get(s) for s in steps]
            xy = [(x, y) for x, y in zip(losses, values, strict=True) if x is not None]
            if xy:
                ax_loss.plot(
                    [p[0] for p in xy],
                    [p[1] for p in xy],
                    color=ruler["color"],
                    linestyle=linestyle,
                    marker=marker,
                    markersize=5,
                    linewidth=1.6,
                )

    for axis in (ax_step, ax_loss):
        axis.set_yscale(scale_name, **scale_kwargs)
        if scale_name == "symlog":
            axis.axhline(0.0, color="#8a8983", linewidth=0.8)
        axis.grid(True, which="major", **GRID_KWARGS)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    # Step 0 exists (raw only), so a plain log x-axis cannot show it; symlog can,
    # but its negative branch is meaningless for a step count, so clip it away.
    ax_step.set_xscale("symlog", linthresh=1)
    ax_step.set_xlim(left=-0.15)
    ax_step.set_xlabel("training step")
    ax_step.set_ylabel(
        rf"$\chi_{{\mathrm{{pos}}}}$  /  $\chi_{{\mathrm{{pos}}}}^{{P}}$  ({scale_name})"
    )
    ax_step.set_title("versus training step", fontsize=11)

    # Loss falls as training proceeds; invert so time still runs left to right.
    ax_loss.invert_xaxis()
    ax_loss.set_xlabel("train loss at the checkpoint  (later in training →)")
    ax_loss.set_title("versus train loss", fontsize=11)

    p_steps = sorted(series.get(("chi_pos_P", "B", "B"), {}))
    raw_steps = sorted(series.get(("chi_pos", "B", "B"), {}))
    missing = [s for s in raw_steps if s not in p_steps]
    if missing:
        ax_step.annotate(
            f"no $\\chi_{{pos}}^{{P}}$ at step {missing[0]}:\nempty AdamW state (pre-step)",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=8,
            color=TEXT_MUTED,
            va="bottom",
        )

    meta = summary["bundle_metadata"]
    config = meta["model_config"]
    fig.suptitle(
        f"Two rulers over a {meta['total_steps_planned']}-step AdamW run — "
        f"Llama dim={config['dim']} x {config['n_layers']} layers x {config['n_heads']} heads, "
        f"{meta['num_parameters']:,} params, {meta['dataset_id']}",
        fontsize=12,
    )
    handles, labels = ax_step.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.24, 1, 0.94))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.155),
        fontsize=8.5,
        frameon=False,
        ncols=2,  # column-major fill puts the raw ruler in one column, P in the other
    )
    fig.text(
        0.012, 0.012, caption(summary, scale_name), fontsize=7.5, color=TEXT_MUTED, va="bottom"
    )

    path = directory / "two_rulers_llama.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def caption(summary: dict[str, Any], scale_name: str = "log") -> str:
    """The figure's caption — the caveats belong here, not in a side channel."""
    lines: list[str] = []
    if summary.get("scientifically_empty"):
        lines.append(
            "RANDOM-TOKEN FALLBACK RUN: measurement batches (and possibly training) used "
            "random tokens. Every value here is scientifically empty."
        )
    axis_note = (
        "Cross pairs are signed and this run produced a non-positive value, hence the symlog "
        "y axis."
        if scale_name == "symlog"
        else "Cross pairs are signed; every value in this run came out positive, so the y axis "
        "is plain log."
    )
    lines.append(
        "Descriptive only: a few-hundred-step demo run, no claim is made about the dynamics. "
        + axis_note
    )
    lines.append(ceiling_summary(summary))
    lines.append(
        "chi_pos^P uses the frozen convention P = 1/(sqrt(v_hat)+eps) with nothing clipped, "
        "floored or masked; the ceiling coordinates above are dead directions the optimizer "
        "never moved, and they suppress chi_pos^P by inflating chi_net^P."
    )
    perf = summary["performance"]
    device = summary["config"]["device"]
    hardware = f"{perf['peak_gpu_mb']:.0f} MiB peak GPU, untuned" if perf.get("peak_gpu_mb") else ""
    lines.append(
        f"Analysis sweep: {len(perf['per_checkpoint'])} checkpoints in "
        f"{perf['total_wall_s']:.1f} s on {device}" + (f", {hardware}" if hardware else "")
    )
    return "\n".join(textwrap.fill(line, width=165) for line in lines)


def ceiling_summary(summary: dict[str, Any]) -> str:
    records = [r for r in summary.get("epsilon_ceiling", []) if "diag_capped_share" in r]
    if not records:
        return "epsilon ceiling: not measured for this run."
    shares = [r["diag_capped_share"] for r in records]
    measured = [r for r in records if "chi_net_p_ceiling_share" in r]
    text = (
        f"epsilon ceiling (P = 1/eps, eps={records[0]['eps']:g}): {span(shares, '.1%')} of the "
        f"P diagonal over {len(records)} checkpoints"
    )
    if measured:
        chi_shares = [r["chi_net_p_ceiling_share"] for r in measured]
        steps = ", ".join(str(r["step"]) for r in measured)
        plural = "s" if len(measured) > 1 else ""
        text += (
            f"; those coordinates carry {span(chi_shares, '.0%')} of chi_net^P on held-out "
            f"batch B (step{plural} {steps})"
        )
    return text + "."


def span(values: list[float], spec: str) -> str:
    """``a-b`` for a range, a single number when there is only one."""
    low, high = min(values), max(values)
    if format(low, spec) == format(high, spec):
        return format(low, spec)
    return f"{format(low, spec)}-{format(high, spec)}"


# ------------------------------------------------------------------- printing


def print_tables(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    perf = summary["performance"]
    print("=" * 78)
    print(f"performance (analysis phase, {summary['config']['device']})")
    print("=" * 78)
    print(perf["table"])

    print("\n" + "=" * 78)
    print("epsilon ceiling of the AdamW P (reported, never clipped)")
    print("=" * 78)
    print(f"{'step':>6} {'P at 1/eps':>22} {'share':>8} {'chi_net^P from ceiling':>24}")
    for record in summary.get("epsilon_ceiling", []):
        if "diag_capped_share" not in record:
            print(f"{record['step']:>6} {'(no optimizer state -- no P row)':>22}")
            continue
        share = f"{record['diag_capped_share']:.2%}"
        chi = (
            f"{record['chi_net_p_ceiling_share']:.2%}"
            if "chi_net_p_ceiling_share" in record
            else "-"
        )
        print(
            f"{record['step']:>6} {record['n_capped']:>10}/{record['n_total']:<11} "
            f"{share:>8} {chi:>24}"
        )
    top = next(
        (
            r["capped_by_param"]
            for r in summary.get("epsilon_ceiling", [])
            if r.get("capped_by_param")
        ),
        [],
    )
    if top:
        print("  ceiling coordinates by parameter (largest first, first checkpoint measured):")
        for entry in top:
            print(f"    {entry['name']:<40} {entry['n_capped']:>9}/{entry['numel']}")
    measured = [r for r in summary.get("epsilon_ceiling", []) if "chi_net_p_ceiling_share" in r]
    if measured:
        print("  never-updated vocabulary rows, and how many of them batch B actually uses:")
        for record in measured:
            for vocab in record.get("dead_vocab_rows", []):
                if not vocab["n_dead_rows"]:
                    continue  # e.g. an untied output head: every row gets gradient
                print(
                    f"    step {record['step']:>4}  {vocab['name']}: "
                    f"{vocab['n_dead_rows']}/{vocab['vocab_size']} rows dead, "
                    f"{vocab['n_dead_rows_present_in_batch_b']} of them occur in batch B "
                    f"({vocab['n_distinct_tokens_in_batch_b']} distinct tokens) -> "
                    f"chi_net^P share {record['chi_net_p_ceiling_share']:.2%}"
                )

    identity = summary["identity_recheck"]
    print("\n" + "=" * 78)
    print("acceptance checks")
    print("=" * 78)
    print(
        f"  P=I reproduces raw : {identity['pairs_compared']} pairs at "
        f"{identity['checkpoint_id']}, worst |P - raw| = "
        f"{identity['worst_abs_deviation']:.3e} <= {identity['tolerance']:.0e} -> "
        f"{'PASS' if identity['passed'] else 'FAIL'}"
    )
    finite = [r for r in rows if not math.isfinite(float(r["value"]))]
    print(f"  all values finite  : {len(rows) - len(finite)}/{len(rows)} rows finite")
    p_ckpts = {r["checkpoint_id"] for r in rows if r["observable"].endswith("_P")}
    all_ckpts = {r["checkpoint_id"] for r in rows}
    print(
        f"  P rows present     : {len(p_ckpts)}/{len(all_ckpts)} checkpoints "
        f"(missing: {sorted(all_ckpts - p_ckpts)} -- step-0 policy)"
    )

    print("\n" + "=" * 78)
    print("chi_pos vs chi_pos^P (held-out eval batch B, self pair)")
    print("=" * 78)
    series = series_by_step(rows)
    raw = series.get(("chi_pos", "B", "B"), {})
    preconditioned = series.get(("chi_pos_P", "B", "B"), {})
    losses = {int(c["step"]): c["train_loss"] for c in summary["checkpoints"]}
    print(f"{'step':>6} {'train_loss':>11} {'chi_pos':>14} {'chi_pos_P':>14} {'ratio':>10}")
    for step in sorted(raw):
        loss = losses.get(step)
        p_value = preconditioned.get(step)
        ratio = "-" if p_value in (None, 0.0) else f"{raw[step] / p_value:.2f}"
        print(
            f"{step:>6} {'-' if loss is None else f'{loss:11.4f}'} {raw[step]:>14.6e} "
            f"{'-' if p_value is None else f'{p_value:14.6e}'} {ratio:>10}"
        )
    if summary.get("scientifically_empty"):
        print("\n!! RANDOM-TOKEN FALLBACK: the numbers above are a pipeline demo, not a result.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="output directory to read")
    args = parser.parse_args(argv)
    directory = args.dir.expanduser().resolve()

    rows, summary = load(directory)
    print_tables(rows, summary)
    path = plot(directory, rows, summary)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
