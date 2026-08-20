set -e
cd "$(dirname "$0")/../.." || exit 1

GPU="${GPUS:-${FLU_GPU:-0}}"
GPU="${GPU%%,*}"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/dlm/run_fluency.py "$@"
