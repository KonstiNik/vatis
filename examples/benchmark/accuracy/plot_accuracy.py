"""Plot the accuracy sweep: chi_net estimator accuracy vs n_hutchinson.

Two panels:
  (left)  relative std (std/mean) across seeds vs n_h, log-log, with a 1/sqrt(n)
          reference. Both methods should track slope -1/2; per_sequence_cv should
          sit BELOW hutchinson (lower variance at equal n_h).
  (right) chi_net estimate (mean +/- std) vs n_h with the high-n reference line —
          shows the estimates are unbiased (means hit the reference) while the
          error bars shrink with n_h.

No GPU, no vatis import — consumes accuracy.json.

    .venv/bin/python examples/benchmark/plot_accuracy.py \\
        --in examples/benchmark/accuracy.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {"hutchinson": ("-", "o", "tab:blue"), "per_sequence_cv": ("--", "s", "tab:orange")}


def stats(vals: list[float]) -> tuple[float, float]:
    mean = sum(vals) / len(vals)
    std = (sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)) ** 0.5
    return mean, std


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text())
    out = Path(args.out) if args.out else Path(args.inp).with_suffix(".png")
    ref = data["ref_chi_net"]

    # group: method -> n_h -> [chi_net]
    grp: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in data["records"]:
        grp[r["method"]][r["n_h"]].append(r["chi_net"])

    fig, (axv, axe) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: relative std vs n_h (log-log) + 1/sqrt(n) guide.
    all_nh = sorted({nh for m in grp for nh in grp[m]})
    for method, byn in grp.items():
        ls, mk, col = STYLE.get(method, ("-", "o", "k"))
        nhs = sorted(byn)
        rel = [stats(byn[nh])[1] / stats(byn[nh])[0] for nh in nhs]
        axv.plot(nhs, rel, ls, marker=mk, color=col, label=method)
    # 1/sqrt(n) reference anchored to the first hutchinson point.
    anchor_nh = all_nh[0]
    anchor_rel = stats(grp["hutchinson"][anchor_nh])[1] / stats(grp["hutchinson"][anchor_nh])[0]
    axv.plot(
        all_nh,
        [anchor_rel * (anchor_nh / nh) ** 0.5 for nh in all_nh],
        ":",
        color="black",
        lw=1.6,
        alpha=0.7,
        label="1/√n (slope -1/2)",
    )
    axv.set_xscale("log", base=2)
    axv.set_yscale("log")
    axv.set_xlabel("n_hutchinson")
    axv.set_ylabel("relative std of chi_net  (std / mean)")
    axv.set_title(f"Estimator accuracy vs n_hutchinson\n({data['seeds']} seeds per point)")
    axv.legend(fontsize=8)

    # Right: mean +/- std vs n_h with the reference line.
    for method, byn in grp.items():
        ls, mk, col = STYLE.get(method, ("-", "o", "k"))
        nhs = sorted(byn)
        means = [stats(byn[nh])[0] for nh in nhs]
        stds = [stats(byn[nh])[1] for nh in nhs]
        axe.errorbar(nhs, means, yerr=stds, fmt=mk, ls=ls, color=col, capsize=3, label=method)
    axe.axhline(ref, color="black", ls=":", lw=1.6, alpha=0.7, label=f"ref (n_h={data['ref_n_h']})")
    axe.set_xscale("log", base=2)
    axe.set_xlabel("n_hutchinson")
    axe.set_ylabel("chi_net estimate")
    axe.set_title("Unbiasedness: mean ± std vs reference\n(means hit ref; error bars shrink)")
    axe.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
