"""Stage 3 (analysis, CPU): plot + tabulate the OLMo SFT LNP results.

Reads ``results.parquet`` (the canonical vatis output) and ``resources.json``
(time / memory / GPU activity) and produces:

- ``spectral_position.png`` — ``chi_pos`` for the code and English self-pairs
  (the spectral position of each batch's SFT loss gradient, in [0, 1]) plus the
  cross-pair ``chi_pos(english, code)`` (the parameter-space gradient cosine,
  in [-1, 1]).
- ``lnp_components.png`` — the full decomposition: ``chi_loss_normalized``,
  ``chi_net_normalized``, ``delta_loss``, ``chi_pos`` as small multiples.
- ``gpu_activity.png`` — the DCGM activity timeline (SM / tensor / DRAM active,
  framebuffer memory) with the load vs analyze phases shaded.

It also prints a markdown observable table and a resource summary.

Like ``examples/analyze_results.py``, this script has **zero vatis imports** —
only ``pyarrow`` and ``matplotlib`` (and stdlib ``json``). The parquet schema
plus the resources JSON are the entire contract with the compute step.

Run (CPU is fine)::

    .venv/bin/python examples/olmo_sft_analysis/analyze_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

_HERE = Path(__file__).resolve()
PARQUET_PATH = _HERE.parent / "results.parquet"
RESOURCES_PATH = _HERE.parent / "resources.json"

CODE = "code"
ENGLISH = "english"
CROSS = (ENGLISH, CODE)

# (key, human label) for the fraction-valued DCGM activity channels.
_ACTIVITY_CHANNELS = [
    ("sm_active", "SM active"),
    ("tensor_active", "tensor core"),
    ("dram_active", "DRAM (mem BW)"),
    ("gr_engine_active", "GR engine"),
]


def pair_label(a: str, b: str) -> str:
    return a if a == b else f"{a}×{b}"


def load_observables(path: Path) -> dict[str, dict[str, float]]:
    """Return ``{observable: {pair_label: value}}`` from the parquet."""
    rows = pq.read_table(path).to_pylist()
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        out.setdefault(r["observable"], {})[pair_label(r["batch_a"], r["batch_b"])] = r["value"]
    return out


def plot_spectral_position(obs: dict[str, dict[str, float]], path: Path) -> None:
    chi_pos = obs.get("chi_pos", {})
    labels = [CODE, ENGLISH, pair_label(*CROSS)]
    values = [chi_pos.get(lbl) for lbl in labels]
    present = [(lbl, v) for lbl, v in zip(labels, values, strict=True) if v is not None]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = range(len(present))
    colors = ["#1f77b4", "#1f77b4", "#d62728"]
    ax.bar(list(xs), [v for _, v in present], color=colors[: len(present)])
    for x, (_, v) in zip(xs, present, strict=True):
        ax.text(x, v, f"{v:.2e}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for lbl, _ in present])
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("chi_pos (spectral position)")
    ax.set_title(
        "Spectral position of the SFT loss gradient\n"
        "self pairs ∈ [0,1]: bulk→1, tail→0  |  cross = gradient cosine ∈ [-1,1]"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def plot_lnp_components(obs: dict[str, dict[str, float]], path: Path) -> None:
    panels = ["chi_loss_normalized", "chi_net_normalized", "delta_loss", "chi_pos"]
    logy = {"chi_net_normalized"}
    labels = [CODE, ENGLISH, pair_label(*CROSS)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, name in zip(axes, panels, strict=True):
        series = obs.get(name, {})
        present = [(lbl, series[lbl]) for lbl in labels if lbl in series]
        if not present:
            ax.set_visible(False)
            continue
        xs = range(len(present))
        ax.bar(list(xs), [v for _, v in present], color="#4c72b0")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([lbl for lbl, _ in present], rotation=20, ha="right")
        ax.set_title(name)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        if name in logy and all(v > 0 for _, v in present):
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("LNP decomposition — Olmo-3-7B-Think-SFT on Dolci-Think-SFT samples")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def plot_gpu_activity(resources: dict, path: Path) -> None:
    timeline = resources.get("timeline", [])
    summary = resources.get("summary", {})
    if not timeline:
        print("no GPU timeline to plot (sampler collected no samples).")
        return
    ts = [s["t"] for s in timeline]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted = False
    for key, label in _ACTIVITY_CHANNELS:
        ys = [s.get(key) for s in timeline]
        if any(y is not None for y in ys):
            ax.plot(
                ts,
                [(y * 100.0 if y is not None else None) for y in ys],
                marker=".",
                ms=3,
                label=label,
            )
            plotted = True
    # Coarse util% fallback if no profiling channels (nvidia-smi sampler).
    if not plotted:
        ys = [s.get("gpu_util_pct") for s in timeline]
        if any(y is not None for y in ys):
            ax.plot(ts, ys, marker=".", ms=3, label="GPU util (coarse)")
            plotted = True
    ax.set_xlabel("time since tracker start (s)")
    ax.set_ylabel("engine active (%)")
    ax.set_ylim(0, 105)

    # Framebuffer memory on a secondary axis.
    fb = [s.get("fb_used_mb") for s in timeline]
    if any(v is not None for v in fb):
        ax2 = ax.twinx()
        ax2.plot(
            ts,
            [(v / 1024.0 if v is not None else None) for v in fb],
            color="gray",
            linestyle="--",
            label="FB mem (GB)",
        )
        ax2.set_ylabel("framebuffer memory (GB)")
        ax2.legend(loc="upper right")

    # Shade the named phases.
    phase_colors = {"load_model": "#ffeda0", "analyze": "#c7e9c0"}
    for p in summary.get("phases", []):
        t0, t1 = p.get("t_start"), p.get("t_end")
        if t0 is None or t1 is None:
            continue
        ax.axvspan(t0, t1, color=phase_colors.get(p["name"], "#eeeeee"), alpha=0.5, zorder=0)
        ax.text((t0 + t1) / 2, 102, p["name"], ha="center", va="top", fontsize=9)

    sampler = summary.get("sampler", "?")
    ax.set_title(f"GPU activity during analysis (sampler: {sampler})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def print_observable_table(obs: dict[str, dict[str, float]]) -> None:
    cols = [CODE, ENGLISH, pair_label(*CROSS)]
    order = [
        "chi_pos",
        "delta_loss",
        "chi_loss_normalized",
        "chi_net_normalized",
        "chi_loss",
        "chi_net",
    ]
    names = [o for o in order if o in obs] + [o for o in obs if o not in order]
    print("\n### LNP observables\n")
    print("| observable | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for name in names:
        cells = []
        for c in cols:
            v = obs[name].get(c)
            cells.append(f"{v:+.4e}" if v is not None else "—")
        print(f"| {name} | " + " | ".join(cells) + " |")


def print_resource_summary(resources: dict) -> None:
    s = resources.get("summary", {})
    if not s:
        return
    print("\n### Resources\n")
    print(
        f"- device: `{s.get('device')}`" + (f" ({s.get('gpu_name')})" if s.get("gpu_name") else "")
    )
    print(
        f"- GPU sampler: `{s.get('sampler')}` "
        f"({s.get('n_gpu_samples')} samples @ {s.get('sampler_interval_s')}s)"
    )
    print(f"- wall total: {s.get('wall_s_total', 0):.1f} s")
    print(
        f"- torch peak: {s.get('torch_peak_alloc_mb', 0):.0f} MB allocated, "
        f"{s.get('torch_peak_reserved_mb', 0):.0f} MB reserved"
    )
    gpu = s.get("gpu", {})

    def stat(key: str, scale: float = 1.0, unit: str = "") -> str:
        st = gpu.get(key)
        return (
            f"mean {st['mean'] * scale:.1f}{unit}, max {st['max'] * scale:.1f}{unit}" if st else "—"
        )

    print(f"- SM active: {stat('sm_active', 100, '%')}")
    print(f"- tensor-core active: {stat('tensor_active', 100, '%')}")
    print(f"- DRAM (mem-BW) active: {stat('dram_active', 100, '%')}")
    print(f"- power: {stat('power_w', 1, ' W')}")
    print("\n| phase | wall_s | torch peak MB | SM active mean |")
    print("|---|---|---|---|")
    for p in s.get("phases", []):
        sm = p.get("gpu", {}).get("sm_active")
        smtxt = f"{sm['mean'] * 100:.0f}%" if sm else "—"
        wall = f"{p['wall_s']:.1f}" if p.get("wall_s") is not None else "—"
        print(f"| {p['name']} | {wall} | {p.get('torch_peak_alloc_mb', 0):.0f} | {smtxt} |")


def main() -> None:
    if not PARQUET_PATH.exists():
        raise SystemExit(f"{PARQUET_PATH} not found — run olmo_sft_analyze.py first.")
    obs = load_observables(PARQUET_PATH)

    plot_spectral_position(obs, _HERE.parent / "spectral_position.png")
    plot_lnp_components(obs, _HERE.parent / "lnp_components.png")
    print_observable_table(obs)

    if RESOURCES_PATH.exists():
        resources = json.loads(RESOURCES_PATH.read_text())
        plot_gpu_activity(resources, _HERE.parent / "gpu_activity.png")
        print_resource_summary(resources)
    else:
        print(f"\n(no {RESOURCES_PATH.name} — skipping resource plots/table)")


if __name__ == "__main__":
    main()
