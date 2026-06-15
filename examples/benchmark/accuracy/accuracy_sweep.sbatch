#!/usr/bin/env bash
#SBATCH --job-name=vatis-accuracy
#SBATCH --partition=alpha
#SBATCH --account=p_neurasearch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/data/horse/ws/koni010i-dpo_sft_transition/vatis/examples/benchmark/accuracy/accuracy_sweep.log

set -euo pipefail
REPO=/data/horse/ws/koni010i-dpo_sft_transition/vatis
cd "$REPO"
export HF_HOME=/data/horse/ws/koni010i-dpo_sft_transition/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "node: $(hostname)"; nvidia-smi -L
"$REPO"/.venv/bin/python examples/benchmark/accuracy/accuracy_sweep.py \
    --model EleutherAI/pythia-14m --revision step3000 --batch-size 8 --seq-len 64 --seeds 100 \
    --n-h-grid 1 2 4 8 16 32 64 --ref-n-h 2048 \
    --batch-file "$REPO/examples/benchmark/eval_batch_b64_s512.pt" \
    --out "$REPO/examples/benchmark/accuracy/accuracy_pythia-14m.json"
