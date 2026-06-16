#!/usr/bin/env bash
#SBATCH --job-name=olmo-sft-lnp
# Partition / account come from .env via examples/benchmark/submit.sh
# (SBATCH_PARTITION / SBATCH_ACCOUNT); override on the CLI to change per-submit.
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=olmo_sft_lnp_%j.log

set -euo pipefail

# IMPORTANT: run prefetch.py on a LOGIN node first (it needs internet) so the
# model weights are in the HF cache and samples.json exists. This job runs
# OFFLINE — see this example's README.
#
# Submit from the repo root via the wrapper that pulls SBATCH_* from .env:
#   examples/benchmark/submit.sh examples/olmo_sft_analysis/run.sbatch
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

# Cache-only on the compute node (no internet). run_config also reads HF_HOME
# from .env; these exports make the offline intent explicit and independent of it.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Reclaim reserved-but-unallocated fragmentation. A 7B chi_net run sits near the
# 7B-on-94GB ceiling (the analyzer holds one batch's full gradient for the
# pending cross-pair while computing the next), so this measurably lowers peak.
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "node: $(hostname)"
nvidia-smi -L
dcgmi discovery -l 2>/dev/null | head -5 || echo "(dcgmi discovery unavailable; tracker will fall back to nvidia-smi)"

# Memory (MEASURED via mem_profile.py on a 93 GiB H100, this 7B model):
#   per-batch peak ~75-84 GB, essentially FLAT in seq_len and batch size — it is
#   param-bound (bf16 weights 14.6 + fp32 flat-grad 29.2 + bf16 grad tuple 14.6
#   + workspace). Activations are negligible, so seq_len is ~free.
# The cross pair needs BOTH batches' gradients for the dot; holding the second
# ~29 GB gradient on-GPU during the next backward would OOM (~104 GB). So we let
# cross_grad_storage=auto offload the cached gradients to host RAM (the dot then
# runs on CPU) -> GPU peak stays at one batch's ~84 GB, the cross story is back.
# hutchinson keeps the chi_net step (~73 GB with the live flat-grad) under the
# delta_loss peak; per_sequence_cv would push it to ~94 GB (too tight).
# seq_len=1024 captures more of the long <think> traces at no memory cost; raise
# it (and --time) for more.
"$REPO"/.venv/bin/python examples/olmo_sft_analysis/olmo_sft_analyze.py \
    --chi-net-method hutchinson --seq-len 1024 --cross-grad-storage auto "$@"

# Plots + tables (CPU-only; safe to run here or later on a login node):
"$REPO"/.venv/bin/python examples/olmo_sft_analysis/analyze_results.py
