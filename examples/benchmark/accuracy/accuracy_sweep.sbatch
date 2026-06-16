#!/usr/bin/env bash
#SBATCH --job-name=vatis-accuracy
# Partition / account come from .env via examples/benchmark/submit.sh
# (SBATCH_PARTITION / SBATCH_ACCOUNT); override on the CLI to change per-submit.
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=accuracy_sweep_%j.log

set -euo pipefail
# Submit from the repo root: sbatch examples/benchmark/accuracy/accuracy_sweep.sbatch
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
# HF cache: set HF_HOME in your shell or the repo-root .env (see run_config.py).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "node: $(hostname)"; nvidia-smi -L
"$REPO"/.venv/bin/python examples/benchmark/accuracy/accuracy_sweep.py \
    --model EleutherAI/pythia-14m --revision step3000 --batch-size 8 --seq-len 64 --seeds 100 \
    --n-h-grid 1 2 4 8 16 32 64 --ref-n-h 2048 \
    --batch-file "$REPO/examples/benchmark/eval_batch_b64_s512.pt" \
    --out "$REPO/examples/benchmark/accuracy/accuracy_pythia-14m.json"
