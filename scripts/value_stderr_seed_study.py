"""Seed studies behind the figures quoted for ``value_stderr``.

``docs/SPEC.md`` -> "Standard error of the sampled values" and ``CLAUDE.md``
quote measured numbers that need more seeds than a test tier should spend.
This script produces them. It is not collected by pytest and not run by CI;
everything is seeded, so a re-run on the same torch build prints the same table.

Run from the repository root, with the repository's environment::

    python scripts/value_stderr_seed_study.py propagation --seeds 4000
    python scripts/value_stderr_seed_study.py weighting --seeds 800

``propagation``
    The comparison of ``tests/unit/test_value_stderr.py::
    test_reported_stderr_matches_the_spread_over_seeds`` with more seeds, on
    the same model, batches, preconditioner and four configurations, and for
    *every* row type in every configuration (the test asserts the first-order
    rows only where ``chi_net`` is known to a few percent). Per row: the
    relative standard deviation of the value, the mean reported variance over
    the sample variance of the value, and their difference in standard errors.
    The summary lines are what the documentation quotes. About seven minutes
    at 4000 seeds on four CPU threads.

``weighting``
    A row's ``value`` and its ``value_stderr`` come from the same per-probe
    values, and for a right-skewed per-probe distribution they are positively
    correlated: a draw that comes out high also reports a large error. Weighting
    repeated measurements by ``1 / value_stderr**2`` therefore favours the low
    draws. Per configuration: the correlation of ``value`` and ``value_stderr``
    over the seeds, the plain mean of ``chi_net``, and the inverse-variance
    weighted mean. Under a minute at 800 seeds.

The helpers (model, batches, the paired comparison) are imported from the test
module so that the study and the test cannot drift apart.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import tests.unit.test_value_stderr as t  # noqa: E402

PAIRS = (("a", "a"), ("b", "b"), ("a", "b"))


def propagation(n_seeds: int) -> None:
    summary: dict[str, dict[str, float]] = {}
    for method, micro, n_h, asserted_in_test in t._STDERR_CONFIGS:
        bundle, precond, _ = t._mlp_bundle()
        values, stderrs = t._collect(
            n_seeds,
            model=bundle,
            eval_batches={"a": t._mlp_batch(5, 1), "b": t._mlp_batch(7, 2)},
            cross_pairs=[("a", "b")],
            preconditioner=precond,
            chi_net_method=method,
            n_hutchinson=n_h,
            micro_batch_size=micro,
            dtype="fp32",
        )
        print(f"=== {method}  micro_batch_size={micro}  n_hutchinson={n_h}  seeds={n_seeds}")
        for a, b in PAIRS:
            for obs in (*t.DIRECT, *t.PROPAGATED):
                first_order = a != b or obs in t.PROPAGATED
                x, s = values[(a, b, obs)], stderrs[(a, b, obs)]
                diff, diff_stderr, v_rep = t._reported_minus_empirical(x, s)
                ratio = v_rep / (diff + v_rep)
                z = diff / diff_stderr
                rel_sd = float(x.std(unbiased=True) / x.mean().abs())
                if not first_order:
                    group = "directly sampled rows"
                elif asserted_in_test:
                    group = "first-order rows, configurations the test asserts"
                else:
                    group = "first-order rows, two probes per micro-batch"
                print(
                    f"  ({a},{b}) {obs:18s} rel_sd={rel_sd:.3f}  "
                    f"reported/observed variance={ratio:.3f}  z={z:+.2f}  [{group}]"
                )
                g = summary.setdefault(
                    group,
                    {"n": 0, "lo": math.inf, "hi": 0.0, "z": 0.0, "r_lo": math.inf, "r_hi": 0.0},
                )
                g["n"] += 1
                g["lo"], g["hi"] = min(g["lo"], ratio), max(g["hi"], ratio)
                g["z"] = max(g["z"], abs(z))
                g["r_lo"], g["r_hi"] = min(g["r_lo"], rel_sd), max(g["r_hi"], rel_sd)
    print(f"SUMMARY  seeds={n_seeds}")
    for group, g in summary.items():
        print(
            f"  {group}: {int(g['n'])} comparisons, relative sd of the value "
            f"{g['r_lo']:.3f}-{g['r_hi']:.3f}, reported/observed variance "
            f"{g['lo']:.3f}-{g['hi']:.3f}, largest |z| {g['z']:.2f}"
        )


def weighting(n_seeds: int) -> None:
    bundle, _, _ = t._mlp_bundle()
    batch = t._mlp_batch(5, 1)
    print(f"chi_net of one batch of 5 samples (test MLP, Rademacher probes), seeds={n_seeds}")
    print(
        "  estimator        micro  n_h   corr(value, stderr)   plain mean   "
        "1/stderr^2-weighted mean   weighted/plain"
    )
    for method in ("hutchinson", "per_sequence_cv"):
        for micro in (5, 1):
            for n_h in (2, 4, 16):
                values, stderrs = t._collect(
                    n_seeds,
                    model=bundle,
                    eval_batches={"a": batch},
                    chi_net_method=method,
                    n_hutchinson=n_h,
                    micro_batch_size=micro,
                    observables=["chi_net"],
                    dtype="fp32",
                )
                x, s = values[("a", "a", "chi_net")], stderrs[("a", "a", "chi_net")]
                corr = float(torch.corrcoef(torch.stack([x, s]))[0, 1])
                weights = 1.0 / (s * s)
                weighted = float((weights * x).sum() / weights.sum())
                plain = float(x.mean())
                print(
                    f"  {method:16s} {micro:5d} {n_h:4d}   {corr:19.3f}   {plain:10.2f}   "
                    f"{weighted:24.2f}   {weighted / plain:14.3f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("study", choices=("propagation", "weighting"))
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    {"propagation": propagation, "weighting": weighting}[args.study](args.seeds)


if __name__ == "__main__":
    main()
