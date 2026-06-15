#!/usr/bin/env bash
#SBATCH --job-name=vatis-bench-160m
#SBATCH --partition=alpha
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/horse/ws/koni010i-dpo_sft_transition/vatis/examples/benchmark/single_gpu/single_gpu_sweep_160m.log

set -euo pipefail

REPO=/data/horse/ws/koni010i-dpo_sft_transition/vatis
cd "$REPO"

# Use the workspace HF cache (pre-populated on the login node) and stay offline
# in case the compute node has no internet.
export HF_HOME=/data/horse/ws/koni010i-dpo_sft_transition/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "node: $(hostname)"
nvidia-smi -L

"$REPO"/.venv/bin/python examples/benchmark/single_gpu/single_gpu_sweep.py \
    --model EleutherAI/pythia-160m --revision step3000
