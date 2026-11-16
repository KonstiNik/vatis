"""Stage 3b (analysis, CPU): error-scaling + convergence figures from the sweep records.

Reads ``runs/{tag}/results.json`` and ``runs/{tag}/sweep.json`` (written by
``diagnostics.py``) and produces four figures (written next to them) answering
"how big are the error bars, and did we use enough probes / pairs?":

- ``r_dir_vs_batch.png`` — R_dir at the nested prefix snapshots
  (B = 10, 40, 160, 640, full). Deterministic (no Hutchinson noise); shows how
  many pairs are needed before the *sign* of R_dir stabilises.
- ``error_scaling.png`` — two panels. Left: probe-bootstrap relative
  std of the boost-arm chi_net estimate vs probes-per-pair n (pairs fixed —
  pure Hutchinson noise), against the 1/sqrt(n) law. Right: pair-bootstrap
  relative std of the per-pair chi_net mean vs number of pairs B at the base
  n = 8 (probe noise + pair-sampling heterogeneity), against 1/sqrt(B).
- ``r_spec_ci.png`` — R_spec point estimates with probe-bootstrap 95%
  CIs. Only chi_net carries Hutchinson noise, so the CI is the full stochastic
  error bar on the reported R_spec.
- ``dpo_margins.png`` — ECDF of the per-pair preference margin at the
  DPO checkpoint (SFT margins are identically 0 because the policy *is* the
  reference model there).
- ``ablation_nb.png`` — the (n_hutchinson x B) ablation. R_spec is
  reconstructed at every nested prefix snapshot B from the stored raw
  aggregates (``grad_*_sq_raw``, ``chi_loss_*_raw``, additive per-pair
  chi_net), so the batch-size dependence of both the estimate and its
  probe-bootstrap CI comes from one run. Three panels: R_spec vs B with 95%
  CIs; relative CI half-width vs probes-per-pair n at full B; relative CI
  half-width vs B at uniform n = 8. R_dir needs no n ablation — it contains
  no Hutchinson probes (see ``r_dir_vs_batch.png`` for its B dependence).

Zero vatis imports — only ``json``, ``numpy`` and ``matplotlib``. The two JSON
files are the entire contract with the compute step.

Run (CPU is fine)::

    .venv/bin/python examples/dpo_spectral_filter/analyze_sweep.py --tag rewardbench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve()

REVISIONS = ["sft", "dpo"]
DOMAINS = ["code", "english"]

# Validated reference palette (dataviz skill): identity = domain, style = revision.
DOMAIN_COLORS = {"code": "#2a78d6", "english": "#1baf7a"}
REV_STYLE = {"sft": {"linestyle": "--", "fillstyle": "none"}, "dpo": {"linestyle": "-"}}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

N_BOOT = 1000
RNG_SEED = 0


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#c3c2b7",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "axes.grid": True,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def _run_dir(tag: str) -> Path:
    return _HERE.parent / "runs" / (tag or "default")


def load(tag: str) -> tuple[dict, dict]:
    run_dir = _run_dir(tag)
    results = json.loads((run_dir / "results.json").read_text())
    sweep = json.loads((run_dir / "sweep.json").read_text())
    return results, sweep


def by_rev_domain(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["revision"], r["domain"]): r for r in rows}


# ---------------------------------------------------------------- figure 1


def plot_r_dir_vs_batch(results: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Per-series end-label offsets so nearby line ends don't collide.
    label_dy = {
        ("sft", "code"): -12,
        ("sft", "english"): 6,
        ("dpo", "code"): -3,
        ("dpo", "english"): 8,
    }
    for rev, domain in ((r, d) for r in REVISIONS for d in DOMAINS):
        row = by_rev_domain(results["results"]).get((rev, domain))
        if row is None:
            continue
        snaps = row["grad_snapshots"]
        xs = [s["n_pairs"] for s in snaps]
        ys = [s["r_dir"] for s in snaps]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=5,
            linewidth=2,
            color=DOMAIN_COLORS[domain],
            label=f"{domain} @ {rev}",
            **REV_STYLE[rev],
        )
        ax.annotate(
            f"{ys[-1]:+.2f}",
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(8, label_dy[(rev, domain)]),
            fontsize=8,
            color=DOMAIN_COLORS[domain],
        )
    ax.set_xscale("log")
    ax.axhline(0.0, color=INK, linewidth=0.7, alpha=0.5)
    ax.set_xlabel("number of preference pairs B (log)")
    ax.set_ylabel("R_dir = cos(grad CE, grad DPO)")
    ax.set_title("R_dir vs batch size — nested prefix snapshots (deterministic)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------- figure 2


def _boost_matrix(record: dict) -> np.ndarray:
    """(n_boost_pairs, 64) matrix of chosen-side probe values."""
    rows = [p["probes_chosen"] for p in record["per_pair_chi_net"] if p["n_probes"] > 8]
    return np.asarray(rows, dtype=np.float64)


def _pair_estimates(record: dict, n_base: int) -> np.ndarray:
    """Per-pair chi_net estimate (chosen side) at a uniform n_base probes."""
    return np.asarray(
        [np.mean(p["probes_chosen"][:n_base]) for p in record["per_pair_chi_net"]],
        dtype=np.float64,
    )


def rel_std_vs_n(probes: np.ndarray, rng: np.random.Generator) -> tuple[list[int], list[float]]:
    """Probe-bootstrap rel. std of the boost-subset chi_net sum vs probes per pair.

    Pairs are fixed; probes are resampled with replacement from each pair's 64
    draws — pure Hutchinson noise, same methodology as the R_spec CI.
    """
    n_pairs, n_probes = probes.shape
    ns, stds = [], []
    n = 1
    while n <= n_probes:
        idx = rng.integers(0, n_probes, size=(N_BOOT, n_pairs, n))
        ests = np.take_along_axis(probes[None, :, :], idx, axis=2).mean(axis=2).sum(axis=1)
        ns.append(n)
        stds.append(float(np.std(ests, ddof=1) / np.mean(ests)))
        n *= 2
    return ns, stds


def rel_std_vs_b(pair_ests: np.ndarray, rng: np.random.Generator) -> tuple[list[int], list[float]]:
    """Bootstrap rel. std of the mean per-pair chi_net estimate vs pair count B."""
    total = len(pair_ests)
    bs = [b for b in (10, 20, 40, 80, 160, 320, 640) if b < total] + [total]
    stds = []
    for b in bs:
        idx = rng.integers(0, total, size=(N_BOOT, b))
        means = pair_ests[idx].mean(axis=1)
        stds.append(float(np.std(means, ddof=1) / np.mean(means)))
    return bs, stds


def plot_error_scaling(sweep: dict, run_config: dict, path: Path) -> dict[tuple[str, str], float]:
    rng = np.random.default_rng(RNG_SEED)
    records = {(r["revision"], r["domain"]): r for r in sweep["records"]}
    n_base = int(run_config["n_hutchinson"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    anchor = None
    for rev, domain in ((r, d) for r in REVISIONS for d in DOMAINS):
        rec = records.get((rev, domain))
        if rec is None:
            continue
        ns, stds = rel_std_vs_n(_boost_matrix(rec), rng)
        ax.plot(
            ns,
            stds,
            marker="o",
            markersize=5,
            linewidth=2,
            color=DOMAIN_COLORS[domain],
            label=f"{domain} @ {rev}",
            **REV_STYLE[rev],
        )
        anchor = anchor or (ns, stds)
    if anchor is not None:
        ns, stds = anchor
        guide = [stds[0] * (ns[0] / n) ** 0.5 for n in ns]
        ax.plot(ns, guide, linestyle=":", color=MUTED, linewidth=1.5, label="1/sqrt(n)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("probes per pair n (boost arm, 20 pairs)")
    ax.set_ylabel("relative std of chi_net")
    ax.set_title("Hutchinson noise vs n (pairs fixed)")
    ax.legend(fontsize=8)

    ax = axes[1]
    final_rel_std: dict[tuple[str, str], float] = {}
    guide_bs: list[int] = []
    guide: list[float] = []
    for rev, domain in ((r, d) for r in REVISIONS for d in DOMAINS):
        rec = records.get((rev, domain))
        if rec is None:
            continue
        bs, stds = rel_std_vs_b(_pair_estimates(rec, n_base), rng)
        ax.plot(
            bs,
            stds,
            marker="o",
            markersize=5,
            linewidth=2,
            color=DOMAIN_COLORS[domain],
            label=f"{domain} @ {rev}",
            **REV_STYLE[rev],
        )
        final_rel_std[(rev, domain)] = stds[-1]
        guide_bs, guide = bs, [stds[0] * (bs[0] / b) ** 0.5 for b in bs]
    ax.plot(guide_bs, guide, linestyle=":", color=MUTED, linewidth=1.5, label="1/sqrt(B)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"number of pairs B (n = {n_base} probes/pair)")
    ax.set_ylabel("relative std of chi_net")
    ax.set_title("Pair-sampling + probe noise vs B (pair bootstrap)")
    ax.legend(fontsize=8)

    fig.suptitle("chi_net error scaling — the only Hutchinson-stochastic observable")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}")
    return final_rel_std


# ---------------------------------------------------------------- figure 3


def _probe_bootstrap_ratio(record: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap (chi_net_ce, chi_net_dpo) raw sums by resampling probes per pair.

    The chosen-side resample is shared between the CE and DPO sums, preserving
    their correlation. Returns two (N_BOOT,) arrays.
    """
    ce = np.zeros(N_BOOT)
    dpo = np.zeros(N_BOOT)
    pairs = record["per_pair_chi_net"]
    for k in sorted({p["n_probes"] for p in pairs}):
        chosen = np.asarray(
            [p["probes_chosen"] for p in pairs if p["n_probes"] == k], dtype=np.float64
        )
        rejected = np.asarray(
            [p["probes_rejected"] for p in pairs if p["n_probes"] == k], dtype=np.float64
        )
        n_pairs = chosen.shape[0]
        idx = rng.integers(0, k, size=(N_BOOT, n_pairs, k))
        ch = np.take_along_axis(chosen[None, :, :], idx, axis=2).mean(axis=2)
        rj = np.take_along_axis(rejected[None, :, :], idx, axis=2).mean(axis=2)
        ce += ch.sum(axis=1)
        dpo += (ch + rj).sum(axis=1)
    return ce, dpo


def plot_r_spec_ci(
    results: dict, sweep: dict, path: Path
) -> dict[tuple[str, str], tuple[float, float, float]]:
    rng = np.random.default_rng(RNG_SEED)
    rows = by_rev_domain(results["results"])
    records = {(r["revision"], r["domain"]): r for r in sweep["records"]}

    cis: dict[tuple[str, str], tuple[float, float, float]] = {}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(DOMAINS), dtype=np.float64)
    width = 0.18
    for i, rev in enumerate(REVISIONS):
        px, pv, err = [], [], [[], []]
        for j, domain in enumerate(DOMAINS):
            row, rec = rows.get((rev, domain)), records.get((rev, domain))
            if row is None or rec is None:
                continue
            ce_boot, dpo_boot = _probe_bootstrap_ratio(rec, rng)
            # R_spec ∝ chi_net_ce / chi_net_dpo; everything else is deterministic.
            r_spec = row["r_spec"] * (ce_boot / row["chi_net_ce"]) / (dpo_boot / row["chi_net_dpo"])
            lo, hi = np.percentile(r_spec, [2.5, 97.5])
            cis[(rev, domain)] = (row["r_spec"], float(lo), float(hi))
            px.append(xs[j] + (i - 0.5) * 2 * width)
            pv.append(row["r_spec"])
            err[0].append(row["r_spec"] - lo)
            err[1].append(hi - row["r_spec"])
        ax.errorbar(
            px,
            pv,
            yerr=err,
            fmt="o" if rev == "dpo" else "o",
            markersize=8,
            capsize=4,
            linewidth=2,
            linestyle="none",
            color=INK,
            markerfacecolor=INK if rev == "dpo" else SURFACE,
            markeredgecolor=INK,
            label=rev,
        )
        for x, v in zip(px, pv, strict=True):
            ax.annotate(
                f"{v:.4f}",
                (x, v),
                textcoords="offset points",
                xytext=(10, -3),
                fontsize=8,
                color=INK,
            )
    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle="--")
    ax.annotate(
        "R_spec = 1 (no spectral separation)",
        (0.02, 1.0),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=(0, 4),
        fontsize=8,
        color=MUTED,
    )
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(DOMAINS)
    ax.set_ylabel("R_spec = chi_pos(DPO) / chi_pos(CE)   (log)")
    ax.set_title("R_spec with probe-bootstrap 95% CI — the full stochastic error bar")
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0.02, 0.88))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}")
    return cis


# ---------------------------------------------------------------- figure 4


def plot_dpo_margins(sweep: dict, path: Path) -> None:
    records = {(r["revision"], r["domain"]): r for r in sweep["records"]}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for domain in DOMAINS:
        rec = records.get(("dpo", domain))
        if rec is None:
            continue
        margins = np.sort([p["margin"] for p in rec["per_pair_dpo"]])
        ecdf = np.arange(1, len(margins) + 1) / len(margins)
        frac_pos = float(np.mean(margins > 0))
        ax.plot(
            margins,
            ecdf,
            linewidth=2,
            color=DOMAIN_COLORS[domain],
            label=f"{domain} ({frac_pos:.0%} resolved)",
        )
    ax.axvline(0.0, color=MUTED, linewidth=1.0, linestyle="--")
    ax.set_xlabel("per-pair margin  (logp_w − logp_l) − (ref_w − ref_l), length-normalized")
    ax.set_ylabel("ECDF")
    ax.set_title(
        "Preference margins at the DPO checkpoint\n"
        "(SFT margins are identically 0: the policy is the reference model)"
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------- figure 5

_BOOT_CHUNK = 250  # bound the (chunk, n_pairs, n) index arrays to ~125 MB


def _prefix_chi_net_boot(
    rec: dict, n_pairs: int, n: int | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap (chi_net_ce, chi_net_dpo) raw sums over the first ``n_pairs`` pairs.

    Resamples ``n`` probes per pair with replacement from that pair's stored
    draws (``n = None`` keeps each pair's own draw count — the tiered design as
    run). The chosen-side resample is shared between the CE and DPO sums,
    preserving their correlation. Returns two (N_BOOT,) arrays.
    """
    ce = np.zeros(N_BOOT)
    dpo = np.zeros(N_BOOT)
    pairs = rec["per_pair_chi_net"][:n_pairs]
    for k in sorted({p["n_probes"] for p in pairs}):
        chosen = np.asarray(
            [p["probes_chosen"] for p in pairs if p["n_probes"] == k], dtype=np.float64
        )
        rejected = np.asarray(
            [p["probes_rejected"] for p in pairs if p["n_probes"] == k], dtype=np.float64
        )
        m = chosen.shape[0]
        draws = k if n is None else n
        for lo_c in range(0, N_BOOT, _BOOT_CHUNK):
            hi_c = min(lo_c + _BOOT_CHUNK, N_BOOT)
            idx = rng.integers(0, k, size=(hi_c - lo_c, m, draws))
            ch = np.take_along_axis(chosen[None, :, :], idx, axis=2).mean(axis=2)
            rj = np.take_along_axis(rejected[None, :, :], idx, axis=2).mean(axis=2)
            ce[lo_c:hi_c] += ch.sum(axis=1)
            dpo[lo_c:hi_c] += (ch + rj).sum(axis=1)
    return ce, dpo


def _r_spec_at_snapshot(rec: dict, snap: dict) -> tuple[float, float]:
    """(point estimate, deterministic prefactor) of R_spec at one prefix snapshot.

    R_spec = chi_pos_dpo / chi_pos_ce = const * chi_net_ce / chi_net_dpo with
    const = (grad_dpo_sq * chi_loss_ce) / (grad_ce_sq * chi_loss_dpo) — all raw
    prefix aggregates stored in the snapshot; chi_net sums use every stored
    probe draw. Reproduces the reported full-B r_spec to ~1e-8 relative.
    """
    pairs = rec["per_pair_chi_net"][: snap["n_pairs"]]
    ch = np.asarray([np.mean(p["probes_chosen"]) for p in pairs])
    rj = np.asarray([np.mean(p["probes_rejected"]) for p in pairs])
    const = (snap["grad_dpo_sq_raw"] * snap["chi_loss_ce_raw"]) / (
        snap["grad_ce_sq_raw"] * snap["chi_loss_dpo_raw"]
    )
    return float(const * ch.sum() / (ch + rj).sum()), float(const)


def _r_spec_ci_at(
    rec: dict, snap: dict, n: int | None, rng: np.random.Generator
) -> tuple[float, float, float]:
    """(point, lo, hi) of the probe-bootstrap 95% CI of R_spec at (snapshot B, n)."""
    point, const = _r_spec_at_snapshot(rec, snap)
    ce, dpo = _prefix_chi_net_boot(rec, snap["n_pairs"], n, rng)
    lo, hi = np.percentile(const * ce / dpo, [2.5, 97.5])
    return point, float(lo), float(hi)


ABLATION_NS = [1, 2, 4, 8, 16, 32, 64]
TABLE_NS = [2, 8, 32]


def plot_ablation_nb(
    results: dict, sweep: dict, path: Path
) -> dict[tuple[str, str], dict[tuple[int, int], float]]:
    """The (n_hutchinson x B) ablation figure. Returns rel. CI half-widths per cell,
    keyed by (n, B), for the printed grid tables."""
    rng = np.random.default_rng(RNG_SEED)
    rows = by_rev_domain(results["results"])
    records = {(r["revision"], r["domain"]): r for r in sweep["records"]}

    grids: dict[tuple[str, str], dict[tuple[int, int], float]] = {}
    full_bs: dict[str, int] = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for rev, domain in ((r, d) for r in REVISIONS for d in DOMAINS):
        row, rec = rows.get((rev, domain)), records.get((rev, domain))
        if row is None or rec is None:
            continue
        snaps = row["grad_snapshots"]
        grid: dict[tuple[int, int], float] = {}
        color = DOMAIN_COLORS[domain]
        style = REV_STYLE[rev]
        label = f"{domain} @ {rev}"

        # Panel 1: R_spec vs B, tiered-n CI (what this run's design reports at B).
        xs, ys, err = [], [], [[], []]
        for snap in snaps:
            point, lo, hi = _r_spec_ci_at(rec, snap, None, rng)
            xs.append(snap["n_pairs"])
            ys.append(point)
            err[0].append(point - lo)
            err[1].append(hi - point)
        axes[0].errorbar(
            xs,
            ys,
            yerr=err,
            marker="o",
            markersize=5,
            linewidth=2,
            capsize=3,
            color=color,
            label=label,
            **style,
        )

        # Panel 2: rel CI half-width vs n at full B.
        full = snaps[-1]
        full_bs[domain] = full["n_pairs"]
        hw_n = []
        for n in ABLATION_NS:
            point, lo, hi = _r_spec_ci_at(rec, full, n, rng)
            hw_n.append((hi - lo) / 2 / point)
        axes[1].plot(
            ABLATION_NS,
            hw_n,
            marker="o",
            markersize=5,
            linewidth=2,
            color=color,
            label=label,
            **style,
        )

        # Panel 3: rel CI half-width vs B at uniform n = 8; fill the table grid.
        hw_b = []
        for snap in snaps:
            for n in TABLE_NS:
                point, lo, hi = _r_spec_ci_at(rec, snap, n, rng)
                grid[(n, snap["n_pairs"])] = (hi - lo) / 2 / point
            hw_b.append(grid[(8, snap["n_pairs"])])
        axes[2].plot(
            [s["n_pairs"] for s in snaps],
            hw_b,
            marker="o",
            markersize=5,
            linewidth=2,
            color=color,
            label=label,
            **style,
        )
        grids[(rev, domain)] = grid

    ax = axes[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle="--")
    ax.set_xlabel("number of pairs B (prefix snapshots)")
    ax.set_ylabel("R_spec (log)")
    ax.set_title("R_spec vs B — tiered-n 95% CI")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    guide = [hw_n[ABLATION_NS.index(8)] * (8 / n) ** 0.5 for n in ABLATION_NS]
    ax.plot(ABLATION_NS, guide, linestyle=":", color=MUTED, linewidth=1.5, label="1/sqrt(n)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("probes per pair n (uniform)")
    ax.set_ylabel("rel. 95% CI half-width of R_spec")
    full_desc = ", ".join(f"{d} {b}" for d, b in full_bs.items())
    ax.set_title(f"error vs n at full B ({full_desc})")
    ax.legend(fontsize=8)

    ax = axes[2]
    bs = [s["n_pairs"] for s in snaps]
    guide = [hw_b[0] * (bs[0] / b) ** 0.5 for b in bs]
    ax.plot(bs, guide, linestyle=":", color=MUTED, linewidth=1.5, label="1/sqrt(B)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of pairs B (prefix snapshots)")
    ax.set_ylabel("rel. 95% CI half-width of R_spec")
    ax.set_title("error vs B at uniform n = 8")
    ax.legend(fontsize=8)

    fig.suptitle(
        "R_spec ablation over (n_hutchinson, B) — probe-bootstrap error, pairs fixed per prefix"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}")
    return grids


def print_ablation(
    results: dict, grids: dict[tuple[str, str], dict[tuple[int, int], float]]
) -> None:
    rows = by_rev_domain(results["results"])
    print("\n### R_spec ablation: rel. 95% CI half-width vs (n_hutchinson, B)")
    print(
        "\nBackward-pass cost per (checkpoint, domain) scales as ~ B * (2n + 3)."
        "\nError is probe-bootstrap only (pairs fixed per nested prefix), so it is"
        "\nnot monotone in B: a heavy-tail pair entering the prefix (see the pair-"
        "\nconcentration table) raises the Hutchinson variance of the chi_net sum.\n"
    )
    for (rev, domain), grid in grids.items():
        bs = sorted({b for _, b in grid})
        print(f"**{rev}/{domain}** (R_spec at full B = {rows[(rev, domain)]['r_spec']:.4f})\n")
        print("| n \\ B | " + " | ".join(str(b) for b in bs) + " |")
        print("|---" * (len(bs) + 1) + "|")
        for n in TABLE_NS:
            cells = " | ".join(f"{grid[(n, b)]:.1%}" for b in bs)
            print(f"| {n} | {cells} |")
        print()


# ---------------------------------------------------------------- main


def print_summary(
    results: dict,
    sweep: dict,
    run_config: dict,
    cis: dict[tuple[str, str], tuple[float, float, float]],
    final_rel_std: dict[tuple[str, str], float],
) -> None:
    records = {(r["revision"], r["domain"]): r for r in sweep["records"]}
    n_base = int(run_config["n_hutchinson"])
    print("\n### R_spec with probe-bootstrap 95% CI\n")
    print("| revision | domain | R_spec | 95% CI | rel half-width |")
    print("|---|---|---|---|---|")
    for rev in REVISIONS:
        for domain in DOMAINS:
            if (rev, domain) not in cis:
                continue
            v, lo, hi = cis[(rev, domain)]
            rel = (hi - lo) / 2 / v
            print(f"| {rev} | {domain} | {v:.4f} | [{lo:.4f}, {hi:.4f}] | {rel:.1%} |")

    print("\n### chi_net pair concentration + pair-bootstrap error at full B\n")
    print("| revision | domain | top-1 pair share | top-10 share | rel std (pair bootstrap) |")
    print("|---|---|---|---|---|")
    for rev in REVISIONS:
        for domain in DOMAINS:
            rec = records.get((rev, domain))
            if rec is None:
                continue
            ests = np.sort(_pair_estimates(rec, n_base))[::-1]
            total = ests.sum()
            print(
                f"| {rev} | {domain} | {ests[0] / total:.1%} | {ests[:10].sum() / total:.1%} "
                f"| {final_rel_std.get((rev, domain), float('nan')):.1%} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default="",
        help="run name matching diagnostics.py --tag: reads/writes runs/{tag}/.",
    )
    args = parser.parse_args()
    run_dir = _run_dir(args.tag)

    _style()
    results, sweep = load(args.tag)
    plot_r_dir_vs_batch(results, run_dir / "r_dir_vs_batch.png")
    final_rel_std = plot_error_scaling(sweep, results["run_config"], run_dir / "error_scaling.png")
    cis = plot_r_spec_ci(results, sweep, run_dir / "r_spec_ci.png")
    plot_dpo_margins(sweep, run_dir / "dpo_margins.png")
    grids = plot_ablation_nb(results, sweep, run_dir / "ablation_nb.png")
    print_summary(results, sweep, results["run_config"], cis, final_rel_std)
    print_ablation(results, grids)


if __name__ == "__main__":
    main()
