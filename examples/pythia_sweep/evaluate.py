"""Post-process a vatis ``results.parquet`` into derived observables.

This script demonstrates the *analysis* half of a typical vatis
workflow. The compute half — running ``analyze()`` over a checkpoint
sweep — is in ``examples/pythia_sweep/run.py``; that's the expensive step
(it loads model weights and runs backward passes). This script is the
cheap step: it reads the parquet output and produces *derived*
observables that are useful for interpreting the LNA results but
aren't part of the canonical row schema.

Why a separate script? Two reasons:

1. **Iteration speed.** ``run.py`` loads 13 checkpoints and runs
   backward passes (its run time is in ``README.md``); ``evaluate.py``
   only reads files. If you want to try a different normalization, a
   different plot style, or a different derived quantity, you do it
   here without re-running compute.
2. **The parquet is the contract.** vatis's public output is the long-
   format parquet schema documented in ``docs/SPEC.md``. Anything you want
   to compute downstream of vatis lives in scripts like this one,
   reading only the parquet — no vatis imports needed.

What this script computes
=========================

For each cross pair ``(A, B)`` recorded in the parquet, it computes
the **normalized cross-batch gradient correlation**:

.. math::

    \\cos(g_A, g_B) = \\frac{\\delta L(A, B)}{\\sqrt{\\delta L(A, A) \\, \\delta L(B, B)}}

This is the cosine similarity between the parameter-space gradients
on batch A and batch B. It's bounded in ``[-1, +1]``:

- ``+1``: gradient steps on A and B point in identical directions —
  perfect transfer.
- ``0``: gradients are orthogonal — A and B are independent.
- ``-1``: gradient steps on A and B point in opposite directions —
  perfect interference; learning A directly hurts B.

The cosine similarity strips out the gradient magnitudes (which can
grow several orders of magnitude over training and dominate the raw
``delta_loss`` numbers) so the *direction* trajectory is visible.

In the committed ``results.parquet`` the raw cross ``delta_loss`` is
not monotonic: +3.25 at step1, +0.67 at step512, +2.62 at step1000,
negative from step4000 on. The cosine drops from +0.31 at step1 to
+0.012 at step512 (about 26×), rebounds to +0.043 at step1000 (3.6×;
the raw value rebounds 3.9×, because the product of the two gradient
norms grows only 1.07× between these two steps) and then stays at small
negative values. The decorrelation is dominated by the early phase
(step1 → step512); after step1000 the two gradients are close to
orthogonal.

Outputs
=======

- ``examples/pythia_sweep/cos_similarity.png`` — one line per cross pair, training
  step on x-axis (log), cos in [-1, +1] on y-axis (linear).
- ``examples/pythia_sweep/observables_over_training.png`` — the common diagnostic
  figure of the examples (``examples/_plots.py``): the loss of each batch (from
  ``run_summary.json`` next to the parquet) and every observable over training.
  Skipped, with a note, when there is no ``run_summary.json``.
- A summary table printed to stdout showing all cross pairs at all
  checkpoints with the derived correlation alongside the canonical
  ``delta_loss`` and ``chi_pos`` columns.

Usage
=====

::

    python examples/pythia_sweep/evaluate.py

Optionally pass an alternate parquet path (the figures are still written
next to this script)::

    python examples/pythia_sweep/evaluate.py path/to/other.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/, for _helpers
from _helpers import step_from_revision  # noqa: E402
from _plots import plot_observables_over_training  # noqa: E402

DEFAULT_PARQUET = Path(__file__).parent / "results.parquet"
COS_PLOT_PATH = Path(__file__).parent / "cos_similarity.png"
OVERVIEW_PATH = Path(__file__).parent / "observables_over_training.png"


def main(parquet_path: Path = DEFAULT_PARQUET) -> None:
    if not parquet_path.exists():
        sys.exit(
            f"results file not found: {parquet_path}\n"
            f"Run examples/pythia_sweep/run.py first to produce it."
        )

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()

    # Index rows by (observable, batch_a, batch_b) -> {step: value}.
    by_obs: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in rows:
        key = (row["observable"], row["batch_a"], row["batch_b"])
        by_obs.setdefault(key, {})[step_from_revision(row["revision"])] = row["value"]

    # Discover the cross pairs and the checkpoint set.
    cross_pairs: list[tuple[str, str]] = sorted({(a, b) for (_obs, a, b) in by_obs if a != b})
    if not cross_pairs:
        sys.exit(
            f"no cross pairs found in {parquet_path}.\n"
            f"Re-run examples/pythia_sweep/run.py with cross_pairs=[(a, b)] to enable this analysis."
        )
    all_steps = sorted({s for d in by_obs.values() for s in d})

    # Compute cos(g_A, g_B) per cross pair per step.
    cos_data: dict[tuple[str, str], dict[int, float]] = {}
    for a, b in cross_pairs:
        cos_data[(a, b)] = {}
        for step in all_steps:
            try:
                d_ab = by_obs[("delta_loss", a, b)][step]
                d_aa = by_obs[("delta_loss", a, a)][step]
                d_bb = by_obs[("delta_loss", b, b)][step]
            except KeyError:
                continue
            denom = d_aa * d_bb
            if denom <= 0.0:
                # delta_loss should always be >= 0 for self pairs (it's a
                # squared norm), but skip the rare zero-gradient case.
                continue
            cos_data[(a, b)][step] = d_ab / math.sqrt(denom)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a, b in cross_pairs:
        pts = sorted(cos_data[(a, b)].items())
        if not pts:
            continue
        xs = [s for s, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, marker="s", linestyle="--", label=f"{a} × {b}")
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\cos(g_A,\, g_B)$")
    ax.set_xscale("log")
    ax.set_title("normalized cross-batch gradient correlation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(COS_PLOT_PATH, dpi=120)
    plt.close(fig)
    print(f"wrote {COS_PLOT_PATH}")

    # ---- printed summary ----
    print()
    header = (
        f"{'step':>8s}  {'pair':>16s}  "
        f"{'δL(A,B)':>12s}  {'chi_pos(A,B)':>14s}  {'cos(g_A,g_B)':>14s}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for a, b in cross_pairs:
        label = f"{a} × {b}"
        for step in all_steps:
            d_ab = by_obs.get(("delta_loss", a, b), {}).get(step)
            cp_ab = by_obs.get(("chi_pos", a, b), {}).get(step)
            cos = cos_data.get((a, b), {}).get(step)
            if d_ab is None or cp_ab is None or cos is None:
                continue
            print(f"{step:>8d}  {label:>16s}  {d_ab:>+12.3e}  {cp_ab:>+14.3e}  {cos:>+14.4f}")
    print(sep)
    print()
    print(
        "The cosine similarity is the cross delta_loss with the gradient magnitudes\n"
        "divided out. What it shows for the committed run is described in\n"
        "examples/pythia_sweep/README.md; the longer notes are in details.md."
    )

    plot_overview(parquet_path, rows)


def plot_overview(parquet_path: Path, rows: list[dict[str, object]]) -> None:
    """The common diagnostic figure, if run.py recorded the losses next to the parquet."""
    summary_path = parquet_path.parent / "run_summary.json"
    if not summary_path.is_file():
        print(f"\nno {summary_path.name} next to {parquet_path}: overview figure skipped")
        return
    summary = json.loads(summary_path.read_text())
    losses: dict[str, dict[int, float]] = {}
    for revision, by_batch in summary["loss"].items():
        for batch, value in by_batch.items():
            losses.setdefault(batch, {})[step_from_revision(revision)] = value
    path = plot_observables_over_training(
        rows,
        losses,
        OVERVIEW_PATH,
        title=f"{summary['model']}: loss and every observable over training",
        loss_label="mean cross-entropy (nats per token)",
        note="Loss: mean next-token cross-entropy of each eval batch at the checkpoint "
        "(run_summary.json). prose, code: the two eval batches of run.py.",
    )
    print(f"\nwrote {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
        + " Draws cos_similarity.png and observables_over_training.png next to this script."
    )
    parser.add_argument(
        "parquet",
        nargs="?",
        type=Path,
        default=DEFAULT_PARQUET,
        help=f"result file to read (default: {DEFAULT_PARQUET.name} next to this script)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args().parquet)
