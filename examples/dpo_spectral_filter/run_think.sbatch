#!/usr/bin/env bash
#SBATCH --job-name=dpo-think
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=188000M
#SBATCH --time=08:00:00
#SBATCH --output=examples/dpo_spectral_filter/logs/dpo_think_%j.log

set -euo pipefail

# One Think-sweep checkpoint per job: R_dir / R_spec on the fixed pair set
# from select_think_pairs.py, with the checkpoint as its own DPO reference.
# Runs OFFLINE: the checkpoint's weights must already be in the HF cache.
# Same memory profile as run.sbatch (seq_len 1024, ~72 GB GPU, ~84 GB host).
#
#   examples/benchmark/submit.sh examples/dpo_spectral_filter/run_think.sbatch \
#       --tag think_sweep/step1000 \
#       --samples examples/dpo_spectral_filter/runs/think_sweep/samples.json \
#       --checkpoint allenai/Olmo-3-7B-Think-SFT@step1000
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "node: $(hostname)"
nvidia-smi -L

"$REPO"/.venv/bin/python -u examples/dpo_spectral_filter/diagnostics.py "$@"
