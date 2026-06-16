#!/usr/bin/env bash
#SBATCH --job-name=vatis-ddp-scaling
# Partition / account come from .env via examples/benchmark/submit.sh
# (SBATCH_PARTITION / SBATCH_ACCOUNT); override on the CLI to change per-submit.
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=ddp_scaling_%j.log

set -euo pipefail

# Submit from the repo root: sbatch examples/benchmark/ddp_scaling/ddp_scaling.sbatch
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

# HF cache: set HF_HOME in your shell or the repo-root .env (see run_config.py);
# the worker reads it. Offline mode assumes the cache is already populated — run
# examples/benchmark/_build_eval_batch.py on a login node first.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=6
# CRITICAL: drop any inherited CUDA_VISIBLE_DEVICES (e.g. =0 from the launching
# shell) so torchrun gives each rank its own GPU instead of stacking all ranks
# onto GPU 0 (which OOM'd the first run).
unset CUDA_VISIBLE_DEVICES

PY="$REPO/.venv/bin/python"
WORKER="$REPO/examples/benchmark/ddp_scaling/ddp_scaling_worker.py"
BATCH="$REPO/examples/benchmark/eval_batch_b64_s512.pt"
OUT="$REPO/examples/benchmark/ddp_scaling/out"
MODEL=EleutherAI/pythia-1.4b
REV=step143000
NH=32
SL=128   # 1.4b @ S=512 OOMs a 40GB card even at W=1; S=128 fits comfortably

echo "node: $(hostname)"; nvidia-smi -L
rm -rf "$OUT"; mkdir -p "$OUT"

run() {  # run W=$1 with the remaining args
  local W=$1; shift
  echo "=== torchrun nproc=$W $* ==="
  "$PY" -m torch.distributed.run --nproc_per_node="$W" --nnodes=1 --standalone "$WORKER" \
    --model "$MODEL" --revision "$REV" --batch-file "$BATCH" --seq-len "$SL" --out-dir "$OUT" "$@"
}

# Strong scaling: fixed total batch B=32 (hutchinson).
for W in 1 2 4; do
  run "$W" --batch-size 32 --n-hutchinson "$NH" --method hutchinson --tag strong
done

# Weak scaling: fixed 8 sequences per GPU (hutchinson).
for W in 1 2 4; do
  run "$W" --batch-size $((8 * W)) --n-hutchinson "$NH" --method hutchinson --tag weak
done

# (The per_sequence_cv "memory rescue" demo was dropped: the O(P)-memory
# reformulation removed the per-sample-grad cache it was illustrating.)

echo "=== analysis ==="
"$PY" "$REPO/examples/benchmark/ddp_scaling/ddp_scaling_analyze.py" \
  --in-dir "$OUT" --out "$REPO/examples/benchmark/ddp_scaling/DDP_SCALING_RESULTS.md"
