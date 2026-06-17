#!/usr/bin/env bash
#SBATCH --job-name=olmo-mem-profile
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=olmo_mem_profile_%j.log

set -euo pipefail

# Submit from the repo root via the wrapper that pulls SBATCH_* from .env:
#   examples/benchmark/submit.sh examples/olmo_sft_analysis/mem_profile.sbatch
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Measure the allocator's true peak, not fragmentation artefacts.
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "node: $(hostname)"
nvidia-smi -L

"$REPO"/.venv/bin/python examples/olmo_sft_analysis/mem_profile.py "$@"
