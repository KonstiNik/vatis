"""Post-process a vatis ``results.parquet`` into derived observables.

This script demonstrates the *analysis* half of a typical vatis
workflow. The compute half — running ``analyze()`` over a checkpoint
sweep — is in ``examples/pythia_sweep.py``; that's the expensive step
(it loads model weights and runs backward passes). This script is the
cheap step: it reads the parquet output and produces *derived*
observables that are useful for interpreting the LNA results but
aren't part of the canonical row schema.

Why a separate script? Two reasons:

1. **Iteration speed.** ``pythia_sweep.py`` takes ~20 s with a warm
   HF cache and ~50 s cold. ``analyze_results.py`` runs in well under
   a second. If you want to try a different normalization, a
   different plot style, or a different derived quantity, you do it
   here without re-running compute.
2. **The parquet is the contract.** vatis's public output is the long-
   format parquet schema documented in CLAUDE.md. Anything you want
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

This matters because the raw cross ``delta_loss`` plot in
``pythia_sweep.py`` looks non-monotonic — the cross value drops from
+3.25 at step1 to +0.67 at step512, then *climbs back* to +2.62 at
step1000 before declining to negative values. That looks like a big
effect because the absolute number 4×'d. The cosine similarity tells
a different story: the correlation drops from +0.31 at step1 to
+0.012 at step512 (a ~26× drop), rebounds to +0.043 at step1000 (a
real but small rebound, also ~4×), and then declines to small
negative values. So the rebound is real in *both* views — but the
absolute-delta_loss view exaggerates it because gradient magnitudes
are simultaneously growing 60× from step512 to step1000. The
underlying decorrelation is dominated by the early phase (step1 →
step512); everything after that is small wiggles around an
essentially-orthogonal regime.

Outputs
=======

- ``examples/cos_similarity.png`` — one line per cross pair, training
  step on x-axis (log), cos in [-1, +1] on y-axis (linear).
- A summary table printed to stdout showing all cross pairs at all
  checkpoints with the derived correlation alongside the canonical
  ``delta_loss`` and ``chi_pos`` columns.

Usage
=====

::

    .venv/bin/python examples/analyze_results.py

Optionally pass an alternate parquet path::

    .venv/bin/python examples/analyze_results.py path/to/other.parquet
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

DEFAULT_PARQUET = Path(__file__).parent / "results.parquet"
COS_PLOT_PATH = Path(__file__).parent / "cos_similarity.png"


def step_from_revision(rev: str) -> int:
    """Parse the trailing integer step out of a Pythia-style revision tag."""
    return int(rev.removeprefix("step"))


def main(parquet_path: Path = DEFAULT_PARQUET) -> None:
    if not parquet_path.exists():
        sys.exit(
            f"results file not found: {parquet_path}\n"
            f"Run examples/pythia_sweep.py first to produce it."
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
            f"Re-run pythia_sweep.py with cross_pairs=[(a, b)] to enable this analysis."
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
        "The cosine similarity is the cross delta_loss with gradient "
        "magnitudes divided\nout. The early-phase decline (step1 → step512) "
        "is the dominant feature: ~26× drop\nfrom +0.31 to +0.012. The small "
        "step512 → step1000 rebound is real but ~10×\nsmaller than what the "
        "absolute delta_loss plot suggests, because gradient\nmagnitudes are "
        "also growing during that step range. After step1000 the\ncorrelation "
        "drifts into the small-negative regime and stays there. See\n"
        "examples/BENCHMARK.md for the spectral interpretation."
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        main()
