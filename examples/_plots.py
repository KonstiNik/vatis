"""The common diagnostic figure of the examples: the loss and every observable over training.

One small panel for the loss, then one panel per observable, each against the
training step. Every panel shows the self pairs (solid line, filled marker) and
the cross pairs (dashed line, hollow marker); the colour follows the first
batch of the pair. A panel whose values are all positive has a log y axis; a
panel with a value at or below zero (cross pairs are signed) has a
symmetric-log axis: logarithmic in both directions, linear in a band around
zero, with a line at zero.

Used by the evaluate scripts, which read the rows with ``pyarrow``::

    rows = pq.read_table("results.parquet").to_pylist()
    plot_observables_over_training(rows, losses, "observables_over_training.png",
                                   title="...", loss_label="...")

Needs matplotlib only; it imports neither vatis nor torch.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — no display backend

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Categorical hues in a fixed order (checked for colour-vision-deficiency
# separation on this surface); identity is also carried by marker and legend.
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "^", "D")
INK, MUTED, GRID, SURFACE = "#2b2b29", "#6f6e68", "#e4e3de", "#fcfcfb"

# Panel order; observables not listed here follow in alphabetical order.
OBSERVABLE_ORDER = (
    "chi_loss",
    "chi_loss_normalized",
    "chi_net",
    "chi_net_normalized",
    "delta_loss",
    "chi_pos",
    "delta_loss_P",
    "chi_net_P",
    "chi_pos_P",
    "delta_loss_P_capped",
    "chi_net_P_capped",
)


def step_of(revision: str) -> int:
    """The training step at the end of a revision label (``step512`` -> 512)."""
    match = re.search(r"(\d+)$", revision)
    if match is None:
        raise ValueError(f"no step number at the end of revision {revision!r}")
    return int(match.group(1))


def plot_observables_over_training(
    rows: Iterable[Mapping[str, Any]],
    losses: Mapping[str, Mapping[int, float]],
    path: str | Path,
    *,
    title: str,
    loss_label: str,
    note: str = "",
    columns: int = 3,
) -> Path:
    """Draw the figure and save it to ``path``.

    Args:
        rows: result rows with at least ``revision``, ``batch_a``, ``batch_b``,
            ``observable`` and ``value`` (as ``pq.read_table(...).to_pylist()``).
        losses: ``{series name: {step: loss}}``. A name that is also a batch
            name is drawn in that batch's colour; any other name (for example
            the train loss a run recorded) is drawn in ink.
        path: the PNG to write.
        title: the figure title.
        loss_label: the y label of the loss panel, with its unit.
        note: one line of small print under the panels (sources, caveats).
        columns: panels per row.
    """
    series: dict[str, dict[tuple[str, str], dict[int, float]]] = {}
    batches: list[str] = []
    for row in rows:
        a, b = row["batch_a"], row["batch_b"]
        if a == b and a not in batches:
            batches.append(a)
        pair = series.setdefault(row["observable"], {}).setdefault((a, b), {})
        pair[step_of(row["revision"])] = float(row["value"])
    for a, _b in (pair for pairs in series.values() for pair in pairs):
        if a not in batches:
            batches.append(a)
    if len(batches) > len(COLOURS):
        raise ValueError(f"{len(batches)} batches, but only {len(COLOURS)} colours: {batches}")
    style = {name: (COLOURS[i], MARKERS[i]) for i, name in enumerate(batches)}

    observables = [o for o in OBSERVABLE_ORDER if o in series]
    observables += sorted(o for o in series if o not in OBSERVABLE_ORDER)
    n_panels = 1 + len(observables)
    n_rows = math.ceil(n_panels / columns)
    fig, axes = plt.subplots(
        n_rows,
        columns,
        figsize=(4.2 * columns, 2.9 * n_rows + 0.8),
        sharex=True,
        facecolor=SURFACE,
        squeeze=False,
    )
    flat = list(axes.ravel())
    all_steps = sorted({s for pairs in series.values() for p in pairs.values() for s in p})

    # The loss panel.
    ax = flat[0]
    for name, by_step in losses.items():
        colour, marker = style.get(name, (INK, "o"))
        steps = sorted(by_step)
        ax.plot(steps, [by_step[s] for s in steps], color=colour, marker=marker, markersize=4)
    ax.set_title("loss", loc="left", color=INK, fontsize=10)
    ax.set_ylabel(loss_label, color=INK, fontsize=9)

    # One panel per observable: self pairs solid, cross pairs dashed and hollow.
    for ax, observable in zip(flat[1:], observables, strict=False):
        values: list[float] = []
        pairs = sorted(series[observable].items(), key=lambda item: (item[0][0] != item[0][1],))
        for (a, b), by_step in pairs:
            colour, marker = style[a]
            cross = a != b
            steps = sorted(by_step)
            ys = [by_step[s] for s in steps]
            values += ys
            ax.plot(
                steps,
                ys,
                color=colour,
                marker=marker,
                markersize=4,
                linestyle="--" if cross else "-",
                markerfacecolor=SURFACE if cross else colour,
                markeredgecolor=colour,
            )
        _set_y_scale(ax, values)
        ax.set_title(observable, loc="left", color=INK, fontsize=10)

    for ax in flat[:n_panels]:
        _set_x_scale(ax, all_steps)
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    # The bottom panel of every column carries the x axis label and tick labels.
    for column in range(columns):
        filled = [i for i in range(column, n_panels, columns)]
        if not filled:
            continue
        bottom = flat[filled[-1]]
        bottom.xaxis.set_tick_params(labelbottom=True)
        bottom.set_xlabel("training step", color=INK, fontsize=9)
    for ax in flat[n_panels:]:
        ax.set_visible(False)

    handles = [
        Line2D([], [], color=style[a][0], marker=style[a][1], label=f"{a} (self pair)")
        for a in batches
        if any((a, a) in pairs for pairs in series.values())
    ]
    cross_pairs = sorted(
        {pair for pairs in series.values() for pair in pairs if pair[0] != pair[1]}
    )
    handles += [
        Line2D(
            [],
            [],
            color=style[a][0],
            marker=style[a][1],
            linestyle="--",
            markerfacecolor=SURFACE,
            label=f"{a} × {b} (cross pair)",
        )
        for a, b in cross_pairs
    ]
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 1 - 0.4 / fig.get_figheight()),
        ncols=min(len(handles), 5),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    if note:
        fig.text(0.01, 0.005, note, color=MUTED, fontsize=8, va="bottom")
    top = 1 - 0.65 / fig.get_figheight()
    bottom_margin = 0.35 / fig.get_figheight() if note else 0.0
    fig.tight_layout(rect=(0, bottom_margin, 1, top))
    out = Path(path)
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def _set_x_scale(ax: Any, steps: list[int]) -> None:
    """Log steps; a symmetric-log axis when step 0 is present (log cannot show it)."""
    if steps and steps[0] <= 0:
        ax.set_xscale("symlog", linthresh=1, linscale=0.5)
        ax.set_xlim(left=-0.2)
    else:
        ax.set_xscale("log")


def _set_y_scale(ax: Any, values: list[float]) -> None:
    """Log when every value is positive, else symmetric-log with a line at zero.

    The linear band is the power of ten below the 10th percentile of the nonzero
    magnitudes, so that one value near zero does not squeeze the rest.
    """
    finite = [v for v in values if math.isfinite(v)]
    magnitudes = sorted(abs(v) for v in finite if v != 0.0)
    if not magnitudes:
        return
    if all(v > 0.0 for v in finite):
        ax.set_yscale("log")
        return
    low = magnitudes[max(0, int(0.10 * (len(magnitudes) - 1)))]
    ax.set_yscale("symlog", linthresh=10 ** math.floor(math.log10(low)), linscale=0.6)
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
