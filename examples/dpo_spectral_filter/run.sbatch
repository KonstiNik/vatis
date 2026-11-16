#!/usr/bin/env bash
#SBATCH --job-name=dpo-spectral-filter
# Partition / account come from .env via examples/benchmark/submit.sh
# (SBATCH_PARTITION / SBATCH_ACCOUNT); override on the CLI to change per-submit.
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=188000M
#SBATCH --time=01:30:00
#SBATCH --output=examples/dpo_spectral_filter/logs/dpo_spectral_filter_%j.log

set -euo pipefail

# IMPORTANT: run prefetch.py on a LOGIN node first (it needs internet) so both
# checkpoints' weights are in the HF cache and samples.json exists. This job
# runs OFFLINE — see this example's README.
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

# Memory profile differs from olmo_sft_analysis: instead of one cached
# parameter-space gradient for the cross pair, diagnostics.py builds TWO
# full-size (7B, fp32, ~28 GB) flat-grad buffers per checkpoint — one for the
# CE loss, one for the DPO loss — to compute R_dir's cosine. Each is offloaded
# to host RAM immediately after its own pass (see diagnostics.py's `ce_pass`/
# `dpo_pass`), so they never both sit on the GPU; the chi_net Hutchinson pass
# (per olmo_sft_analysis: ~75-84 GB peak at 7B) then runs with that headroom
# back. --mem is 188000M: the two ~29 GB fp32 host-RAM grad buffers are held
# simultaneously for the final dot product (~58 GB), and from_pretrained's
# transient staging during load spikes well above that — 128G OOM-killed
# job 3686474 at ~6 min. 188000M is the per-GPU ceiling Capella allows for
# non-exclusive (single-GPU) jobs; that is the headroom we take.
"$REPO"/.venv/bin/python -u examples/dpo_spectral_filter/diagnostics.py \
    --n-hutchinson 32 "$@"

# Plots + tables (CPU-only; safe to run here or later on a login node).
# analyze_results.py needs matplotlib (the `examples` extra), which a bare
# `uv sync` venv lacks — job 3686513's compute succeeded but this step crashed
# with ModuleNotFoundError and marked the whole job FAILED. Ensure it's present
# (offline-safe: a no-op once the extra is already installed).
"$REPO"/.venv/bin/python -c "import matplotlib" 2>/dev/null \
    || uv pip install --python "$REPO"/.venv/bin/python matplotlib
"$REPO"/.venv/bin/python examples/dpo_spectral_filter/analyze_results.py
