#!/usr/bin/env bash
#SBATCH --job-name=dpo-spectral-filter
# Partition / account come from .env via examples/benchmark/submit.sh
# (SBATCH_PARTITION / SBATCH_ACCOUNT); override on the CLI to change per-submit.
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=188000M
#SBATCH --time=20:00:00
#SBATCH --output=examples/dpo_spectral_filter/logs/dpo_spectral_filter_%j.log

set -euo pipefail

# IMPORTANT: run prefetch.py on a LOGIN node first (it needs internet) so both
# checkpoints' weights are in the HF cache and runs/{tag}/samples.json exists.
# This job runs OFFLINE — see this example's README.
#
# Submit from the repo root via the wrapper that pulls SBATCH_* from .env:
#   examples/benchmark/submit.sh examples/dpo_spectral_filter/run.sbatch
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "node: $(hostname)"
nvidia-smi -L

# Memory profile (chunked grad passes, see diagnostics.py `grad_passes`):
# GPU peak is the DPO chunk pass — one ~28 GB fp32 flat chunk-gradient +
# bf16 model (~14 GB) + both retained seq graphs (~25 GB at S=1024) ≈ 80 GB,
# the same profile job 3688580 survived at seq_len 1024. Do NOT raise
# --seq-len past 1024 in this configuration: job 3705072 OOM'd at 2048.
# HOST RAM peak is the two running ~28 GB fp32 flat-grad accumulators plus
# one ~28 GB incoming chunk (~84 GB), plus from_pretrained's transient
# staging during load — 128G OOM-killed job 3686474 at ~6 min. 188000M is
# the per-GPU ceiling Capella allows for non-exclusive (single-GPU) jobs.
#
# n_hutchinson / boost / snapshots come from diagnostics.py defaults
# (8 base + 64 on the first 20 pairs; snapshots at 10,40,160,640,B);
# override per-submit via "$@".
"$REPO"/.venv/bin/python -u examples/dpo_spectral_filter/diagnostics.py "$@"

# Plots + tables (CPU-only; safe to run here or later on a login node).
# analyze_results.py needs matplotlib (the `examples` extra), which a bare
# `uv sync` venv lacks — job 3686513's compute succeeded but this step crashed
# with ModuleNotFoundError and marked the whole job FAILED. Ensure it's present
# (offline-safe: a no-op once the extra is already installed).
"$REPO"/.venv/bin/python -c "import matplotlib" 2>/dev/null \
    || uv pip install --python "$REPO"/.venv/bin/python matplotlib

# Mirror diagnostics.py's --tag (if any) so the analysis reads the right
# runs/{tag}/ directory.
TAG=""
prev=""
for a in "$@"; do
    if [ "$prev" = "--tag" ]; then TAG="$a"; fi
    prev="$a"
done
"$REPO"/.venv/bin/python examples/dpo_spectral_filter/analyze_results.py ${TAG:+--tag "$TAG"}
# Error-scaling figures from the per-probe/per-pair sweep records.
"$REPO"/.venv/bin/python examples/dpo_spectral_filter/analyze_sweep.py ${TAG:+--tag "$TAG"}
