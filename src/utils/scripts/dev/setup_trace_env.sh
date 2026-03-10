#!/usr/bin/env bash

# Source this file:
#   source decluttering/scripts/dev/setup_trace_env.sh

REPO_ROOT="/home/justinyu/multicable-decluttering"
CONDA_ENV_NAME="handloom"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTHONPATH="${REPO_ROOT}/decluttering/src:${REPO_ROOT}/decluttering/data/detectron2_repo:${REPO_ROOT}"
export LD_LIBRARY_PATH="${HOME}/anaconda3/envs/${CONDA_ENV_NAME}/lib/python3.10/site-packages/torch/lib:${HOME}/anaconda3/envs/${CONDA_ENV_NAME}/lib:${LD_LIBRARY_PATH:-}"

# Prevent occasional OpenMP shared-memory issues in constrained environments.
export OMP_NUM_THREADS=1
export KMP_AFFINITY=disabled

echo "Trace env ready:"
echo "  CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "  PYTHONPATH=${PYTHONPATH}"
echo "  python=$(command -v python)"
