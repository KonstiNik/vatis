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
#   delta_loss peak ~75 GB, essentially FLAT in seq_len and batch size — it is
#   param-bound (bf16 weights 14.6 + fp32 flat-grad 29.2 + bf16 grad tuple 14.6
#   + ~16 backward workspace). Activations are negligible, so seq_len is ~free.
# The earlier OOMs were a vatis bug (now fixed in analyzer.py): the analyzer
# cached EVERY batch's full gradient unconditionally, so batch 2 ran with
# batch 1's ~29 GB gradient still resident (75 + 29 -> OOM). With cross pairs
# off nothing is cached, so each batch peaks at ~75 GB — ample headroom on 93 GiB.
# hutchinson keeps the chi_net step (~73 GB with the live flat-grad) under the
# delta_loss peak; per_sequence_cv would push it to ~94 GB (too tight).
# Cross pairs still need two gradients resident (~104 GB) -> left off for now
# (future work: CPU-offload the cached gradient). seq_len=1024 captures more of
# the long <think> traces at no memory cost; raise it (and --time) for more.
"$REPO"/.venv/bin/python examples/olmo_sft_analysis/olmo_sft_analyze.py \
    --chi-net-method hutchinson --seq-len 1024 --no-cross-pairs "$@"

# Plots + tables (CPU-only; safe to run here or later on a login node):
"$REPO"/.venv/bin/python examples/olmo_sft_analysis/analyze_results.py
