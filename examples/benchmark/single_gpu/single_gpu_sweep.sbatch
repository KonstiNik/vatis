#!/usr/bin/env bash
#SBATCH --job-name=vatis-bench-160m
# Partition / account come from .env via examples/benchmark/submit.sh
# (SBATCH_PARTITION / SBATCH_ACCOUNT); override on the CLI to change per-submit.
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=single_gpu_sweep_%j.log

set -euo pipefail

# Submit from the repo root: sbatch examples/benchmark/single_gpu/single_gpu_sweep.sbatch
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

# HF cache: set HF_HOME in your shell or the repo-root .env (see run_config.py).
# Offline mode assumes the cache is already populated on the compute node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "node: $(hostname)"
nvidia-smi -L

"$REPO"/.venv/bin/python examples/benchmark/single_gpu/single_gpu_sweep.py \
    --model EleutherAI/pythia-160m --revision step3000
