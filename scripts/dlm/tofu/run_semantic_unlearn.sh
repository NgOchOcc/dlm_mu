#!/usr/bin/env bash
# Train forget-only Semantic_Unlearn on TOFU. The frozen teacher distribution is
# tilted away from each masked gold token's input-embedding neighbourhood, and
# the student minimizes forward KL to that target without a retain corpus.
#
# Usage:
#   bash scripts/dlm/tofu/run_semantic_unlearn.sh
#   MODEL=dream GPUS=0,1 SPLIT=forget05 bash scripts/dlm/tofu/run_semantic_unlearn.sh
#   TARGET_MODE=cosine ETA=2.0 bash scripts/dlm/tofu/run_semantic_unlearn.sh
#
# Env vars (all optional):
#   MODEL              llada | dream                  (default: llada)
#   GPUS               comma-separated device ids     (default: 0,1,2,3)
#   SPLIT              forget10 | forget05 | forget01 (default: forget10)
#   TASK_SUFFIX        suffix for the save directory  (mode-aware; see below)
#   ETA                semantic suppression strength  (default: 10.0)
#   TARGET_MODE        rbf | cosine                    (default: rbf)
#   SIGMA_FALLBACK     fallback RBF scale              (default: 0.05)
#   CENTER_CHUNK_SIZE  distinct centers per chunk      (default: 8)
#   VOCAB_CHUNK_SIZE   vocabulary rows per chunk       (default: 4096)
# Any command-line arguments are passed through to unlearn.sh / Hydra.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODEL="${MODEL:-llada}"
GPUS="${GPUS:-0,1,2,3}"
SPLIT="${SPLIT:-forget10}"
ETA="${ETA:-10.0}"
TARGET_MODE="${TARGET_MODE:-rbf}"
SIGMA_FALLBACK="${SIGMA_FALLBACK:-0.05}"
CENTER_CHUNK_SIZE="${CENTER_CHUNK_SIZE:-8}"
VOCAB_CHUNK_SIZE="${VOCAB_CHUNK_SIZE:-4096}"
DEFAULT_SUFFIX="_semantic_eta${ETA}"
if [[ "${TARGET_MODE}" != "rbf" ]]; then
    DEFAULT_SUFFIX="_semantic_${TARGET_MODE}_eta${ETA}"
fi
SUFFIX="${TASK_SUFFIX:-${DEFAULT_SUFFIX}}"

mkdir -p logs
LOG="logs/semantic_unlearn${SUFFIX}.log"

echo "[run_semantic_unlearn] model=${MODEL} gpus=${GPUS} split=${SPLIT} suffix=${SUFFIX}" >&2
echo "[run_semantic_unlearn] target_mode=${TARGET_MODE} eta=${ETA} sigma_fallback=${SIGMA_FALLBACK} center_chunk=${CENTER_CHUNK_SIZE} vocab_chunk=${VOCAB_CHUNK_SIZE}" >&2
echo "[run_semantic_unlearn] log -> ${LOG}" >&2

TASK_SUFFIX="${SUFFIX}" setsid bash scripts/dlm/tofu/unlearn.sh \
    "${MODEL}" "${GPUS}" "${SPLIT}" SemanticUnlearn \
    trainer.method_args.eta="${ETA}" \
    trainer.method_args.target_mode="${TARGET_MODE}" \
    trainer.method_args.sigma_fallback="${SIGMA_FALLBACK}" \
    trainer.method_args.center_chunk_size="${CENTER_CHUNK_SIZE}" \
    trainer.method_args.vocab_chunk_size="${VOCAB_CHUNK_SIZE}" \
    "$@" > "${LOG}" 2>&1 < /dev/null &

echo "[run_semantic_unlearn] launched pid=$! (detached via setsid); tail -f ${LOG}" >&2
