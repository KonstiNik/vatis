#!/usr/bin/env bash
# Submit a benchmark sbatch script with cluster settings pulled from the
# repo-root .env — the single source of truth (see run_config.py).
#
# Put your cluster-specific values in .env:
#     SBATCH_PARTITION=...
#     SBATCH_ACCOUNT=...
# SLURM reads these SBATCH_* variables automatically at submit time, so the
# .sbatch scripts themselves carry no site-specific partition/account.
#
# Usage (from anywhere):
#     examples/benchmark/submit.sh examples/benchmark/single_gpu/single_gpu_sweep.sbatch
#
# Override any setting per-submit on the command line, which wins over .env:
#     examples/benchmark/submit.sh --partition=foo <script>
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# Export .env into the environment so SLURM picks up the SBATCH_* vars.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec sbatch "$@"
