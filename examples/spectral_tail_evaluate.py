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

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from _helpers import step_from_revision

DATA_DIR = Path(__file__).parent / "spectral_tail"
FIG_DIR = DATA_DIR / "figures"

# Models we expect results for, in order of scale.
MODELS = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
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


def param_count_millions(name: str) -> float:
    """Extract approximate param count in millions from a model name."""
    tag = model_tag(name)
    # e.g. "pythia-14m" -> 14, "pythia-1b" -> 1000
    for part in tag.split("-"):
        if part.endswith("m") and part[:-1].isdigit():
            return float(part[:-1])
        if part.endswith("b") and part[:-1].isdigit():
            return float(part[:-1]) * 1000
    return 0.0


# Pythia training config: all sizes use batch_size=1024, seq_len=2048.
PYTHIA_TOKENS_PER_STEP = 1024 * 2048  # ~2.1M tokens/step


def step_to_flops(step: int, model_name: str) -> float:
    """Estimate training FLOPs at a given step: C ≈ 6 * N * D."""
    n_params = param_count_millions(model_name) * 1e6
    tokens_seen = step * PYTHIA_TOKENS_PER_STEP
    return 6 * n_params * tokens_seen


def parquet_path(name: str) -> Path:
    return DATA_DIR / f"results_{model_tag(name)}.parquet"


def load_series(
    pq_path: Path,
) -> dict[tuple[str, str, str], list[tuple[int, float]]]:
    """Load a parquet into {(observable, batch_a, batch_b): [(step, value)]}."""
    rows = pq.read_table(pq_path).to_pylist()
    series: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
    for row in rows:
        key = (row["observable"], row["batch_a"], row["batch_b"])
        series.setdefault(key, []).append((step_from_revision(row["revision"]), row["value"]))
    for key in series:
        series[key].sort()
    return series


def cross_label(a: str, b: str) -> str:
    return f"{a} x {b}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compute",
        action="store_true",
        help="Use estimated FLOPs (6ND) as x-axis instead of training step.",
    )
    args = parser.parse_args()
    use_compute = args.compute

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
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    xlabel = "estimated FLOPs (6ND)" if use_compute else "training step"
    suffix = "_compute" if use_compute else ""

    def xvals(pts: list[tuple[int, float]], model_name: str) -> list[float]:
        """Convert step-based points to the chosen x-axis."""
        if use_compute:
            return [step_to_flops(p[0], model_name) for p in pts]
        return [float(p[0]) for p in pts]

    # ---- per-observable figure: one subplot per model scale ----
    for obs_name in OBSERVABLES:
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5.5), squeeze=False)

        for col, (model_name, _pq_path) in enumerate(available):
            ax = axes[0, col]
            series = all_data[model_name]
            tag = model_tag(model_name)

            # Discover batch names from the series keys.
            batch_names = sorted({a for (_o, a, b) in series if a == b})
            cross_pairs = sorted({(a, b) for (_o, a, b) in series if a != b})

            # Self pairs.
            for bname in batch_names:
                pts = series.get((obs_name, bname, bname), [])
                if pts:
                    ax.plot(
                        xvals(pts, model_name),
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
                        xvals(pts, model_name),
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
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_title(tag, fontsize=12)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=9)

        fig.suptitle(f"{obs_name} vs {xlabel}", fontsize=14, y=1.02)
        fig.tight_layout()
        out_path = FIG_DIR / f"{obs_name}{suffix}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")

    # ---- cross-scale comparison: one figure per (observable, cross pair) ----
    # Collect all cross pairs across models.
    all_cross_pairs: set[tuple[str, str]] = set()
    for model_name, _ in available:
        series = all_data[model_name]
        all_cross_pairs |= {(a, b) for (_o, a, b) in series if a != b}
    all_cross_pairs_sorted = sorted(all_cross_pairs)

    cross_obs = [
        ("chi_pos", r"$\chi_{\mathrm{pos}}$", "symlog", {"linthresh": 1e-9}),
        ("delta_loss", r"$\delta L$", "symlog", {"linthresh": 1.0}),
    ]

    # Build colormap over model scale (log of param count).
    scale_values = [param_count_millions(name) for name, _ in available]
    norm = mcolors.LogNorm(vmin=min(scale_values), vmax=max(scale_values))
    cmap = plt.cm.cool

    for obs_name, ylabel, yscale, yscale_kw in cross_obs:
        for a, b in all_cross_pairs_sorted:
            pair_tag = f"{a}_x_{b}"
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for model_name, _pq_path in available:
                series = all_data[model_name]
                tag = model_tag(model_name)
                color = cmap(norm(param_count_millions(model_name)))
                pts = series.get((obs_name, a, b), [])
                if pts:
                    ax.plot(
                        xvals(pts, model_name),
                        [p[1] for p in pts],
                        marker="o",
                        color=color,
                        label=tag,
                    )

            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"{ylabel} ({cross_label(a, b)})")
            ax.set_xscale("log")
            ax.set_yscale(yscale, **yscale_kw)
            ax.set_title(f"cross {ylabel} across model scales ({cross_label(a, b)})")
            ax.grid(True, which="both", alpha=0.3)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = fig.colorbar(sm, ax=ax, pad=0.02, ticks=scale_values)
            cbar.ax.set_yticklabels([model_tag(name) for name, _ in available])
            cbar.set_label("model scale")

            fig.tight_layout()
            out_path = FIG_DIR / f"{obs_name}_cross_scale_{pair_tag}{suffix}.png"
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"wrote {out_path}")

    # ---- single-model comparison: all cross pairs on one axes ----
    # Use the largest available model to maximize chance of resolved semantics.
    largest_model = max(available, key=lambda x: param_count_millions(x[0]))
    lg_name, lg_path = largest_model
    lg_tag = model_tag(lg_name)
    lg_series = all_data[lg_name]

    # Define cross pairs in order: positive signal first, then controls.
    cross_pair_styles = [
        ("python", "cpp", "python x cpp (same algorithm)", "-", "o", "#1f77b4"),
        ("python", "prose", "python x prose (same algorithm)", "-", "s", "#2ca02c"),
        ("cpp", "cpp_str", "cpp x cpp_str (same language, diff algorithm)", "--", "^", "#d62728"),
        (
            "python",
            "cpp_str",
            "python x cpp_str (diff language, diff algorithm)",
            "--",
            "D",
            "#9467bd",
        ),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for batch_a, batch_b, label, ls, marker, color in cross_pair_styles:
        pts = lg_series.get(("chi_pos", batch_a, batch_b), [])
        if pts:
            ax.plot(
                xvals(pts, lg_name),
                [p[1] for p in pts],
                marker=marker,
                markersize=6,
                linestyle=ls,
                color=color,
                linewidth=2,
                label=label,
            )

    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)

    # Shade the late-training region of interest.
    if use_compute:
        ax.axvspan(
            step_to_flops(8000, lg_name),
            step_to_flops(64000, lg_name),
            alpha=0.07,
            color="orange",
            label="late-training window",
        )
    else:
        ax.axvspan(8000, 64000, alpha=0.07, color="orange", label="late-training window")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"$\chi_{\mathrm{pos}}$ (cross pair)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_title(
        f"cross $\\chi_{{\\mathrm{{pos}}}}$ — signal vs controls ({lg_tag})",
        fontsize=13,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    out_path = FIG_DIR / f"chi_pos_signal_vs_controls_{lg_tag}{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
