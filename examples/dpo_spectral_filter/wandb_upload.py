"""Post-run W&B upload: push a finished diagnostics run to Weights & Biases.

The Capella GPU nodes run offline, so live W&B logging during the job is
impossible; instead this script uploads a finished run's outputs
(``runs/{tag}/results.json`` + optional ``sweep.json`` / ``resources.json``)
from a login node. Zero vatis imports — the JSON files written by
``diagnostics.py`` are the entire contract, so any past run (e.g.
``--tag dolci``) can be uploaded retroactively.

One-time setup (interactive, needs your API key)::

    .venv/bin/wandb login

Then, per run::

    .venv/bin/python examples/dpo_spectral_filter/wandb_upload.py \
        --tag rewardbench --project dpo-spectral-filter

What lands in the W&B run:

- **config**: the samples' ``meta`` block + ``run_config`` (n_hutchinson,
  boost, snapshots, beta, seed, ...).
- **summary scalars**: ``{revision}/{domain}/{observable}`` for every
  observable in each results record (r_dir, r_spec, chi_pos_ce, ...).
- **grad_snapshots table**: one row per (revision, domain, n_pairs) prefix —
  the R_dir / delta_loss / chi_loss vs batch-size curves.
- **resources**: wall/peak-memory summary scalars.
- **plots**: every ``*.png`` in the run directory, as images.
- **artifact** (``--artifacts``): the raw JSON files, including the per-probe
  ``sweep`` records, so the error analysis is reproducible from W&B alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve()


def _run_dir(tag: str) -> Path:
    return _HERE.parent / "runs" / (tag or "default")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default="",
        help="run name matching diagnostics.py --tag: reads runs/{tag}/.",
    )
    parser.add_argument("--project", default="dpo-spectral-filter")
    parser.add_argument("--entity", default=None, help="W&B team/user; default = your login.")
    parser.add_argument("--run-name", default=None, help="default: the tag, or 'dolci' if none.")
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help="also upload the raw JSON files (incl. per-probe sweep records) as a W&B artifact.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="write a local offline run (sync later with `wandb sync`) — for testing.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="existing W&B run id to update in place instead of creating a new run.",
    )
    parser.add_argument(
        "--job-log",
        default=None,
        help="SLURM console log to attach to the run (shows under Files).",
    )
    args = parser.parse_args()
    run_dir = _run_dir(args.tag)

    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"{results_path} not found — run diagnostics.py first.")
    payload = json.loads(results_path.read_text())

    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name or (args.tag or "default"),
        mode="offline" if args.offline else "online",
        id=args.run_id,
        resume="allow" if args.run_id else None,
        config={"meta": payload.get("meta", {}), "run_config": payload.get("run_config", {})},
    )

    snapshot_rows: list[list] = []
    snapshot_cols = [
        "revision",
        "domain",
        "n_pairs",
        "n_valid_chosen",
        "r_dir",
        "dot_raw",
        "grad_ce_sq_raw",
        "grad_dpo_sq_raw",
        "chi_loss_ce_raw",
        "chi_loss_dpo_raw",
    ]
    # Scalars go through run.log (not just run.summary) so the Charts
    # workspace auto-creates a panel per observable; grad snapshots are
    # logged as step-wise series keyed on n_pairs so R_dir/‖g‖² vs batch
    # size render as real line charts.
    run.define_metric("n_pairs")
    run.define_metric("snap/*", step_metric="n_pairs")
    scalars: dict[str, float] = {}
    for rec in payload["results"]:
        prefix = f"{rec['revision']}/{rec['domain']}"
        for key, value in rec.items():
            if isinstance(value, (int, float)):
                scalars[f"{prefix}/{key}"] = value
        for snap in rec.get("grad_snapshots", []):
            snapshot_rows.append(
                [rec["revision"], rec["domain"]] + [snap.get(c) for c in snapshot_cols[2:]]
            )
            run.log(
                {"n_pairs": snap["n_pairs"]}
                | {
                    f"snap/{prefix}/{c}": snap[c]
                    for c in ("r_dir", "grad_ce_sq_raw", "grad_dpo_sq_raw", "dot_raw")
                    if snap.get(c) is not None
                }
            )
    run.log(scalars)
    if snapshot_rows:
        run.log({"grad_snapshots": wandb.Table(columns=snapshot_cols, data=snapshot_rows)})

    resources_path = run_dir / "resources.json"
    if resources_path.exists():
        summary = json.loads(resources_path.read_text()).get("summary", {})
        for key in ("wall_s_total", "torch_peak_alloc_mb", "torch_peak_reserved_mb"):
            if key in summary:
                run.summary[f"resources/{key}"] = summary[key]

    images = {png.stem: wandb.Image(str(png)) for png in sorted(run_dir.glob("*.png"))}
    if images:
        run.log(images)

    if args.job_log:
        job_log = Path(args.job_log)
        if not job_log.exists():
            raise SystemExit(f"--job-log {job_log} not found")
        run.save(str(job_log), base_path=str(job_log.parent), policy="now")

    if args.artifacts:
        artifact = wandb.Artifact(f"diagnostics_{args.tag or 'default'}", type="run-output")
        for stem in ("results", "sweep", "resources"):
            path = run_dir / f"{stem}.json"
            if path.exists():
                artifact.add_file(str(path))
        run.log_artifact(artifact)

    run.finish()
    print(f"uploaded: {run.name} ({run.url or 'offline — sync with `wandb sync`'})")


if __name__ == "__main__":
    main()
