"""Stage 3 (analysis, CPU): tabulate + plot R_dir / R_spec across the SFT->DPO step.

Reads ``results.json`` (written by ``diagnostics.py``) and ``resources.json``
and produces:

- ``r_dir_r_spec.png`` — ``R_dir`` and ``R_spec`` at both checkpoints, per
  domain (code, english). The four-outcome table in the main README's
  "Four possible outcomes" section is the lens for reading this plot:
  ``R_spec > 1`` and ``R_dir`` stable positive together is the strongest
  support for the hypothesis.
- ``chi_pos_breakdown.png`` — ``chi_pos_ce`` vs ``chi_pos_dpo`` at both
  checkpoints (the two terms ``R_spec`` is a ratio of).

Zero vatis imports — only ``json`` and ``matplotlib``. ``results.json`` is the
entire contract with the compute step.

Run (CPU is fine)::

    .venv/bin/python examples/dpo_spectral_filter/analyze_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve()
RESULTS_PATH = _HERE.parent / "results.json"
RESOURCES_PATH = _HERE.parent / "resources.json"

REVISIONS = ["sft", "dpo"]
DOMAINS = ["code", "english"]
REV_COLORS = {"sft": "#1f77b4", "dpo": "#d62728"}


def load_results(path: Path) -> dict[str, dict[str, dict]]:
    """Return ``{domain: {revision: record}}``."""
    payload = json.loads(path.read_text())
    out: dict[str, dict[str, dict]] = {d: {} for d in DOMAINS}
    for r in payload["results"]:
        out.setdefault(r["domain"], {})[r["revision"]] = r
    return out


def plot_r_dir_r_spec(by_domain: dict[str, dict[str, dict]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.35
    xs = range(len(DOMAINS))

    ax = axes[0]
    for i, rev in enumerate(REVISIONS):
        offset = (i - 0.5) * width
        vals = [by_domain[d].get(rev, {}).get("r_dir") for d in DOMAINS]
        bx = [x + offset for x in xs]
        ax.bar(
            bx,
            [v if v is not None else 0.0 for v in vals],
            width=width,
            label=rev,
            color=REV_COLORS.get(rev),
        )
        for x, v in zip(bx, vals, strict=True):
            if v is not None:
                va = "bottom" if v >= 0 else "top"
                ax.text(x, v, f"{v:+.2f}", ha="center", va=va, fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(DOMAINS)
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("R_dir (cosine, grad CE vs grad DPO)")
    ax.set_title("R_dir — readiness diagnostic")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    for i, rev in enumerate(REVISIONS):
        offset = (i - 0.5) * width
        vals = [by_domain[d].get(rev, {}).get("r_spec") for d in DOMAINS]
        bx = [x + offset for x in xs]
        ax.bar(
            bx,
            [v if v is not None else 0.0 for v in vals],
            width=width,
            label=rev,
            color=REV_COLORS.get(rev),
        )
        for x, v in zip(bx, vals, strict=True):
            if v is not None:
                ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(DOMAINS)
    ax.axhline(1.0, color="black", linewidth=0.7, alpha=0.5, linestyle="--")
    ax.set_ylabel("R_spec = chi_pos(DPO) / chi_pos(CE)")
    ax.set_title("R_spec — spectral-tail diagnostic (1.0 = no separation)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("DPO as a spectral filter — Olmo-3-7B Instruct, SFT vs DPO checkpoint")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}")


def plot_chi_pos_breakdown(by_domain: dict[str, dict[str, dict]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(5 * len(DOMAINS), 4.5))
    if len(DOMAINS) == 1:
        axes = [axes]
    width = 0.35
    for ax, domain in zip(axes, DOMAINS, strict=True):
        xs = range(len(REVISIONS))
        ce_vals = [by_domain[domain].get(rev, {}).get("chi_pos_ce") for rev in REVISIONS]
        dpo_vals = [by_domain[domain].get(rev, {}).get("chi_pos_dpo") for rev in REVISIONS]
        ax.bar(
            [x - width / 2 for x in xs],
            [v if v is not None else 0.0 for v in ce_vals],
            width=width,
            label="chi_pos(CE)",
            color="#4c72b0",
        )
        ax.bar(
            [x + width / 2 for x in xs],
            [v if v is not None else 0.0 for v in dpo_vals],
            width=width,
            label="chi_pos(DPO)",
            color="#dd8452",
        )
        ax.set_xticks(list(xs))
        ax.set_xticklabels(REVISIONS)
        ax.set_title(domain)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("chi_pos (spectral position)")
    axes[0].legend()
    fig.suptitle("chi_pos(CE) vs chi_pos(DPO) at each checkpoint")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}")


def print_table(by_domain: dict[str, dict[str, dict]]) -> None:
    print("\n### R_dir / R_spec\n")
    header = ["domain"] + [f"{m} ({rev})" for m in ("R_dir", "R_spec") for rev in REVISIONS]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for domain in DOMAINS:
        cells = []
        for metric in ("r_dir", "r_spec"):
            for rev in REVISIONS:
                v = by_domain[domain].get(rev, {}).get(metric)
                cells.append(f"{v:+.4f}" if v is not None else "—")
        print(f"| {domain} | " + " | ".join(cells) + " |")

    print("\n### Full observables\n")
    cols = [
        "chi_loss_ce",
        "delta_loss_ce",
        "chi_net_ce",
        "chi_pos_ce",
        "chi_loss_dpo",
        "delta_loss_dpo",
        "chi_net_dpo",
        "chi_pos_dpo",
    ]
    for domain in DOMAINS:
        print(f"\n**{domain}**\n")
        print("| metric | " + " | ".join(REVISIONS) + " |")
        print("|" + "---|" * (len(REVISIONS) + 1))
        for col in cols:
            cells = [
                f"{by_domain[domain].get(rev, {}).get(col):.4e}"
                if by_domain[domain].get(rev, {}).get(col) is not None
                else "—"
                for rev in REVISIONS
            ]
            print(f"| {col} | " + " | ".join(cells) + " |")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found — run diagnostics.py first.")
    by_domain = load_results(RESULTS_PATH)

    plot_r_dir_r_spec(by_domain, _HERE.parent / "r_dir_r_spec.png")
    plot_chi_pos_breakdown(by_domain, _HERE.parent / "chi_pos_breakdown.png")
    print_table(by_domain)

    if RESOURCES_PATH.exists():
        resources = json.loads(RESOURCES_PATH.read_text())
        s = resources.get("summary", {})
        print("\n### Resources\n")
        gpu_name = f" ({s.get('gpu_name')})" if s.get("gpu_name") else ""
        print(f"- device: `{s.get('device')}`{gpu_name}")
        print(f"- wall total: {s.get('wall_s_total', 0):.1f} s")
        print(f"- torch peak: {s.get('torch_peak_alloc_mb', 0):.0f} MB allocated")
        for p in s.get("phases", []):
            wall = f"{p['wall_s']:.1f}" if p.get("wall_s") is not None else "—"
            print(f"  - {p['name']}: {wall}s, peak {p.get('torch_peak_alloc_mb', 0):.0f} MB")


if __name__ == "__main__":
    main()
