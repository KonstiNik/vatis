"""Evaluate the spectral tail experiment across model scales.

Reads parquet files produced by ``spectral_tail_experiment.py`` and
generates comparison plots. No vatis imports -- only reads the parquet
contract.

For each of the three LNA quantities (chi_loss, chi_net, chi_pos) and
delta_loss, produces a figure with one subplot per model scale showing
self-pair and cross-pair trajectories side by side. Then produces a
dedicated cross-scale comparison figure overlaying the cross chi_pos
from all available models on the same axes.

Usage::

    .venv/bin/python examples/spectral_tail_evaluate.py

Reads from ``examples/spectral_tail/results_*.parquet``.
Writes plots to ``examples/spectral_tail/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from _helpers import step_from_revision

OUT_DIR = Path(__file__).parent / "spectral_tail"

# Models we expect results for, in order of scale.
MODELS = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
]

OBSERVABLES = ["chi_loss_normalized", "chi_net_normalized", "delta_loss", "chi_pos"]

YSCALES: dict[str, tuple[str, dict[str, float]]] = {
    "chi_loss_normalized": ("linear", {}),
    "chi_net_normalized": ("log", {}),
    "delta_loss": ("symlog", {"linthresh": 1.0}),
    "chi_pos": ("symlog", {"linthresh": 1e-9}),
}

YLABEL = {
    "chi_loss_normalized": r"$\tilde{\chi}_{\mathrm{loss}}$",
    "chi_net_normalized": r"$\tilde{\chi}_{\mathrm{net}}$",
    "delta_loss": r"$\delta L$",
    "chi_pos": r"$\chi_{\mathrm{pos}}$",
}


def model_tag(name: str) -> str:
    return name.split("/")[-1]


def parquet_path(name: str) -> Path:
    return OUT_DIR / f"results_{model_tag(name)}.parquet"


def load_series(
    pq_path: Path,
) -> dict[tuple[str, str, str], list[tuple[int, float]]]:
    """Load a parquet into {(observable, batch_a, batch_b): [(step, value)]}."""
    rows = pq.read_table(pq_path).to_pylist()
    series: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
    for row in rows:
        key = (row["observable"], row["batch_a"], row["batch_b"])
        series.setdefault(key, []).append(
            (step_from_revision(row["revision"]), row["value"])
        )
    for key in series:
        series[key].sort()
    return series


def cross_label(a: str, b: str) -> str:
    return f"{a} x {b}"


def main() -> None:
    # Discover available results.
    available: list[tuple[str, Path]] = []
    for model_name in MODELS:
        pq_path = parquet_path(model_name)
        if pq_path.exists():
            available.append((model_name, pq_path))
        else:
            print(f"  skipping {model_name}: {pq_path} not found")

    if not available:
        sys.exit("no result files found -- run spectral_tail_experiment.py first")

    all_data: dict[str, dict[tuple[str, str, str], list[tuple[int, float]]]] = {}
    for model_name, pq_path in available:
        all_data[model_name] = load_series(pq_path)

    n_models = len(available)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- per-observable figure: one subplot per model scale ----
    for obs_name in OBSERVABLES:
        fig, axes = plt.subplots(
            1, n_models, figsize=(7 * n_models, 5.5), squeeze=False
        )

        for col, (model_name, _pq_path) in enumerate(available):
            ax = axes[0, col]
            series = all_data[model_name]
            tag = model_tag(model_name)

            # Discover batch names from the series keys.
            batch_names = sorted(
                {a for (_o, a, b) in series if a == b}
            )
            cross_pairs = sorted(
                {(a, b) for (_o, a, b) in series if a != b}
            )

            # Self pairs.
            for bname in batch_names:
                pts = series.get((obs_name, bname, bname), [])
                if pts:
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        marker="o",
                        markersize=5,
                        linestyle="-",
                        label=bname,
                    )

            # Cross pairs.
            for a, b in cross_pairs:
                pts = series.get((obs_name, a, b), [])
                if pts:
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        marker="s",
                        markersize=5,
                        linestyle="--",
                        label=cross_label(a, b),
                    )

            if col == 0:
                ax.set_ylabel(YLABEL.get(obs_name, obs_name), fontsize=11)
            ax.set_xscale("log")
            scale_kind, scale_kwargs = YSCALES[obs_name]
            ax.set_yscale(scale_kind, **scale_kwargs)
            if scale_kind == "symlog":
                ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
            ax.set_xlabel("training step", fontsize=11)
            ax.set_title(tag, fontsize=12)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=9)

        fig.suptitle(f"{obs_name} vs training step", fontsize=14, y=1.02)
        fig.tight_layout()
        out_path = OUT_DIR / f"{obs_name}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")

    # ---- cross-scale comparison: cross chi_pos on the same axes ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_name, _pq_path in available:
        series = all_data[model_name]
        tag = model_tag(model_name)

        # Find the cross pair (there should be exactly one).
        cross_pairs = sorted({(a, b) for (_o, a, b) in series if a != b})
        for a, b in cross_pairs:
            pts = series.get(("chi_pos", a, b), [])
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="o",
                    label=tag,
                )

    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\chi_{\mathrm{pos}}$ (cross pair)")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-9)
    ax.set_title(r"cross $\chi_{\mathrm{pos}}$ across model scales")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "chi_pos_cross_scale.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")

    # ---- cross-scale comparison: cross delta_loss on the same axes ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_name, _pq_path in available:
        series = all_data[model_name]
        tag = model_tag(model_name)

        cross_pairs = sorted({(a, b) for (_o, a, b) in series if a != b})
        for a, b in cross_pairs:
            pts = series.get(("delta_loss", a, b), [])
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="o",
                    label=tag,
                )

    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\delta L$ (cross pair)")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title(r"cross $\delta L$ across model scales")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "delta_loss_cross_scale.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
