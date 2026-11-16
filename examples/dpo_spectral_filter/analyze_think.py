"""Offline metrics for the Think SFT-trajectory runs (no GPU, no vatis import).

Reads ``runs/{tag}/results.json`` + ``sweep.json`` written by ``diagnostics.py``
(one or more ``--checkpoint`` runs, possibly from separate per-checkpoint tags)
and prints, per checkpoint, the observables beyond R_dir / R_spec:

1. **Gradient coherence** ``kappa = ||sum_i g_i||^2 / sum_i ||g_i||^2`` for CE
   and DPO (1 = pairs pull in unrelated directions, B = all identical), and the
   implied mean pairwise alignment ``rho = (kappa - 1) / (B - 1)``.
2. **Per-pair spectral position** ``chi_pos_i = dL_i / (chi_loss_i chi_net_i)``
   (median / IQR) and the median per-pair R_spec.
3. **Chosen-rejected score similarity** ``cos(s_w, s_l)`` per pair, from
   ``||s_w||^2`` (CE grad / n_valid^2), ``||s_l||^2`` (extra backward) and
   ``||s_w - s_l||^2`` (DPO grad / (beta w)^2).
4. **Preference accuracy / margin** — raw (``logp_w > logp_l``) and, if a
   base-model run is present, the implicit-reward margin against the base.
5. **One-step transfer / linearity** from the finite-difference pass: realized
   vs first-order-predicted loss change, realized R_dir, curvature along d.
6. **Spectral concentration** ``nu = E_gamma[lambda^2] / E_gamma[lambda]^2``
   along each loss gradient (1 = the gradient sits in one eigen-band).

Usage::

    .venv/bin/python examples/dpo_spectral_filter/analyze_think.py \\
        --tag think_smoke [--base-revision main] [--beta 0.1]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

_RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _q(a: np.ndarray) -> str:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "n/a"
    p25, p50, p75 = np.percentile(a, [25, 50, 75])
    return f"{p50:+.3e} [{p25:+.3e}, {p75:+.3e}]"


def load(tags: list[str]) -> list[tuple[dict, dict]]:
    """[(summary, raw)] per (checkpoint, group), across all given tags."""
    out = []
    for tag in tags:
        run = _RUNS_DIR / tag
        res = json.loads((run / "results.json").read_text())
        sweep = json.loads((run / "sweep.json").read_text())
        beta = res["run_config"]["beta"]
        for summ, raw in zip(res["results"], sweep["records"], strict=True):
            summ["_beta"] = beta
            summ["_tag"] = tag
            out.append((summ, raw))
    return out


def merge_pooled(entries: list[tuple[dict, dict]]) -> None:
    """Give each pooled record the concatenated per-pair data of its groups."""
    for summ, raw in entries:
        if summ["domain"] != "pooled":
            continue
        groups = [
            r
            for s, r in entries
            if s["_tag"] == summ["_tag"]
            and s["revision"] == summ["revision"]
            and s["domain"] in raw["groups"]
        ]
        for key in ("per_pair_ce", "per_pair_dpo", "per_pair_chi_net"):
            raw[key] = [x for g in groups for x in g[key]]


def coherence(summ: dict, raw: dict) -> dict[str, float]:
    full = summ["grad_snapshots"][-1] if "grad_snapshots" in summ else summ
    b = len(raw["per_pair_ce"])
    out = {}
    for name, key, recs in (
        ("ce", "grad_ce_sq_raw", raw["per_pair_ce"]),
        ("dpo", "grad_dpo_sq_raw", raw["per_pair_dpo"]),
    ):
        total = sum(r["grad_sq_raw"] for r in recs)
        kappa = full[key] / total
        out[f"kappa_{name}"] = kappa
        out[f"rho_{name}"] = (kappa - 1.0) / (b - 1) if b > 1 else math.nan
    return out


def per_pair_chi_pos(raw: dict) -> dict[str, np.ndarray]:
    ce, dpo, cn = raw["per_pair_ce"], raw["per_pair_dpo"], raw["per_pair_chi_net"]
    net_w = np.array([np.mean(r["probes_chosen"]) for r in cn])
    net_l = np.array([np.mean(r["probes_rejected"]) for r in cn])
    pos_ce = np.array([r["grad_sq_raw"] / r["chi_loss_raw"] for r in ce]) / net_w
    pos_dpo = np.array([r["grad_sq_raw"] / r["chi_loss_raw"] for r in dpo]) / (net_w + net_l)
    return {"chi_pos_ce": pos_ce, "chi_pos_dpo": pos_dpo, "r_spec_pair": pos_dpo / pos_ce}


def score_cosine(raw: dict, beta: float) -> dict[str, np.ndarray]:
    ce, dpo = raw["per_pair_ce"], raw["per_pair_dpo"]
    sw = np.array([r["grad_sq_raw"] / r["n_valid"] ** 2 for r in ce])
    sl = np.array([r["score_l_sq_raw"] for r in dpo])
    w = _sigmoid(-np.array([r["margin"] for r in dpo]))
    diff = np.array([r["grad_sq_raw"] for r in dpo]) / (beta * w) ** 2
    cos = (sw + sl - diff) / (2.0 * np.sqrt(sw * sl))
    # per-pair R_dir = cos(g_CE_i, g_DPO_i) = cos(s_w, s_w - s_l)
    #                = (||s_w||^2 - <s_w, s_l>) / (||s_w|| ||s_w - s_l||)
    sw_dot_sl = 0.5 * (sw + sl - diff)
    r_dir = (sw - sw_dot_sl) / np.sqrt(sw * diff)
    return {
        "cos_sw_sl": cos,
        "sl_over_sw": sl / sw,
        "diff_over_sum": diff / (sw + sl),
        "r_dir_pair": r_dir,
    }


def preference(raw: dict, base_raw: dict | None, beta: float) -> dict[str, float]:
    lw = np.array([r["logp_w"] for r in raw["per_pair_dpo"]])
    ll = np.array([r["logp_l"] for r in raw["per_pair_dpo"]])
    out = {"acc_raw": float(np.mean(lw > ll)), "gap_raw_median": float(np.median(lw - ll))}
    if base_raw is not None:
        bw = np.array([r["logp_w"] for r in base_raw["per_pair_dpo"]])
        bl = np.array([r["logp_l"] for r in base_raw["per_pair_dpo"]])
        m = beta * ((lw - bw) - (ll - bl))
        out["acc_vs_base"] = float(np.mean(m > 0))
        out["margin_vs_base_median"] = float(np.median(m))
    return out


def perturbation(raw: dict) -> list[dict]:
    pert = raw.get("perturbation")
    if not pert:
        return []
    l0 = pert["loss_zero"]
    chi_loss = {
        "ce": sum(r["chi_loss_raw"] for r in raw["per_pair_ce"]),
        "dpo": sum(r["chi_loss_raw"] for r in raw["per_pair_dpo"]),
    }
    by_key = {(r["direction"], r["rel_step"]): r for r in pert["records"]}
    rows = []
    for (d, step), r in sorted(by_key.items()):
        eps = r["eps"]
        row = {"direction": d, "rel_step": step, "frac_changed": r["frac_params_changed"]}
        for a in ("ce", "dpo"):
            act_p = r["loss_plus"][a] - l0[a]
            act_m = r["loss_minus"][a] - l0[a]
            pred_p, pred_m = r["pred"][a]["plus"], r["pred"][a]["minus"]
            # central difference cancels curvature: compare to the first-order term
            row[f"lin_{a}"] = (act_p - act_m) / (pred_p - pred_m) if pred_p != pred_m else math.nan
            row[f"curv_{a}"] = (act_p + act_m) / eps**2
            row[f"descent_{a}"] = act_m  # loss change for a descent step along d
        # Spectral concentration along this loss's own gradient. jd ~ ||J s||^2
        # with s = Delta+ - Delta- the realized (bf16-rounded) step, so normalize
        # by ||s||^2, not (2 eps)^2. ``step_cos`` = cos(s, g_d): how faithfully
        # the rounded step follows the gradient direction.
        jd = r["jd_sq_chosen"] + (r["jd_sq_rejected"] if d == "dpo" else 0.0)
        s_sq = r.get("delta_diff_sq") or (2.0 * eps) ** 2
        row["nu"] = (jd / s_sq) * chi_loss[d] / r["g_norm_raw"] ** 2
        own = r["pred"][d]
        row["step_cos"] = (own["plus"] - own["minus"]) / (r["g_norm_raw"] * math.sqrt(s_sq))
        rows.append(row)
    # realized R_dir: DPO-loss change from a CE descent step vs from a DPO step
    for step in sorted({s for _, s in by_key}):
        ce_step = next((x for x in rows if x["direction"] == "ce" and x["rel_step"] == step), None)
        dpo_step = next(
            (x for x in rows if x["direction"] == "dpo" and x["rel_step"] == step), None
        )
        if ce_step and dpo_step and dpo_step["descent_dpo"] != 0:
            ce_step["realized_r_dir"] = ce_step["descent_dpo"] / dpo_step["descent_dpo"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", action="append", required=True, help="runs/{tag}; repeatable")
    parser.add_argument(
        "--base-revision",
        default=None,
        help="revision label of the base-model run (for margin vs base)",
    )
    parser.add_argument(
        "--base-tag",
        default=None,
        help="tag of the base-model run (for margin vs base); use when several "
        "checkpoints share a revision label such as 'main'.",
    )
    args = parser.parse_args()

    entries = load(args.tag)
    merge_pooled(entries)
    # base-model records per group, matched pair-for-pair (same samples.json)
    base_by_domain = {
        s["domain"]: raw
        for s, raw in entries
        if (args.base_tag is not None and s["_tag"] == args.base_tag)
        or (args.base_tag is None and s["revision"] == args.base_revision)
    }

    for summ, raw in entries:
        beta = summ["_beta"]
        print(
            f"\n=== {summ['_tag']}: {summ['checkpoint']} @ {summ['revision']}  [{summ['domain']}]"
            f"  B={summ['n_pairs']}  ref={summ.get('ref')}"
        )
        print(
            f"R_dir={summ['r_dir']:+.4f}  R_spec={summ['r_spec']:.4e}  "
            f"chi_pos_ce={summ['chi_pos_ce']:.3e}  chi_pos_dpo={summ['chi_pos_dpo']:.3e}"
        )

        coh = coherence(summ, raw)
        print(
            f"[1] coherence  kappa_ce={coh['kappa_ce']:.3f} (rho {coh['rho_ce']:+.4f})   "
            f"kappa_dpo={coh['kappa_dpo']:.3f} (rho {coh['rho_dpo']:+.4f})"
        )

        pp = per_pair_chi_pos(raw)
        print(f"[2] per-pair chi_pos_ce  median [IQR] {_q(pp['chi_pos_ce'])}")
        print(f"    per-pair chi_pos_dpo median [IQR] {_q(pp['chi_pos_dpo'])}")
        print(
            f"    per-pair R_spec      median [IQR] {_q(pp['r_spec_pair'])}"
            f"   (batch R_spec {summ['r_spec']:.3e})"
        )

        sc = score_cosine(raw, beta)
        cos = sc["cos_sw_sl"]
        print(
            f"[3] cos(s_w, s_l)       median [IQR] {_q(cos)}  "
            f"out-of-range: {int(np.sum(np.abs(cos) > 1.0 + 1e-6))}/{cos.size}"
        )
        print(f"    per-pair R_dir       median [IQR] {_q(sc['r_dir_pair'])}")
        print(
            f"    ||s_w-s_l||^2/(||s_w||^2+||s_l||^2) median {np.median(sc['diff_over_sum']):.3f}"
            f"   ||s_l||^2/||s_w||^2 median {np.median(sc['sl_over_sw']):.3f}"
        )

        base_raw = base_by_domain.get(summ["domain"])
        pref = preference(raw, base_raw if base_raw is not raw else None, beta)
        line = f"[4] pref acc (logp_w > logp_l) {pref['acc_raw']:.3f}  median gap {pref['gap_raw_median']:+.3f}"
        if "acc_vs_base" in pref:
            line += (
                f"   acc vs base {pref['acc_vs_base']:.3f}"
                f"  median margin vs base {pref['margin_vs_base_median']:+.4f}"
            )
        print(line)

        rows = perturbation(raw)
        if rows:
            print(
                "[5/6] dir  r        changed step_cos lin_ce   lin_dpo  curv_ce     curv_dpo    nu      realized_R_dir"
            )
            for r in rows:
                print(
                    f"      {r['direction']:4s} {r['rel_step']:<8g} {r['frac_changed']:.3f}   "
                    f"{r['step_cos']:+.3f}   "
                    f"{r['lin_ce']:+.3f}   {r['lin_dpo']:+.3f}   {r['curv_ce']:+.3e}  "
                    f"{r['curv_dpo']:+.3e}  {r['nu']:7.2f}  "
                    f"{r.get('realized_r_dir', float('nan')):+.4f}"
                )


if __name__ == "__main__":
    main()
