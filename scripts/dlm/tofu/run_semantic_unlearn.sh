#!/usr/bin/env bash
# Train forget-only Semantic_Unlearn on TOFU. The frozen teacher distribution is
# tilted away from each masked gold token's input-embedding neighbourhood, and
# the student minimizes forward KL to that target without a retain corpus.
#
# Usage:
#   bash scripts/dlm/tofu/run_semantic_unlearn.sh
#   MODEL=dream GPUS=0,1 SPLIT=forget05 bash scripts/dlm/tofu/run_semantic_unlearn.sh
#   ETA=2.0 CENTER_CHUNK_SIZE=4 bash scripts/dlm/tofu/run_semantic_unlearn.sh
#
# Env vars (all optional):
#   MODEL              llada | dream                  (default: llada)
#   GPUS               comma-separated device ids     (default: 0,1,2,3)
#   SPLIT              forget10 | forget05 | forget01 (default: forget10)
#   TASK_SUFFIX        suffix for the save directory  (default: _semantic_eta${ETA})
#   ETA                semantic suppression strength  (default: 1.0)
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
SIGMA_FALLBACK="${SIGMA_FALLBACK:-0.05}"
CENTER_CHUNK_SIZE="${CENTER_CHUNK_SIZE:-8}"
VOCAB_CHUNK_SIZE="${VOCAB_CHUNK_SIZE:-4096}"
SUFFIX="${TASK_SUFFIX:-_semantic_eta${ETA}}"

mkdir -p logs
LOG="logs/semantic_unlearn${SUFFIX}.log"

echo "[run_semantic_unlearn] model=${MODEL} gpus=${GPUS} split=${SPLIT} suffix=${SUFFIX}" >&2
echo "[run_semantic_unlearn] eta=${ETA} sigma_fallback=${SIGMA_FALLBACK} center_chunk=${CENTER_CHUNK_SIZE} vocab_chunk=${VOCAB_CHUNK_SIZE}" >&2
echo "[run_semantic_unlearn] log -> ${LOG}" >&2

TASK_SUFFIX="${SUFFIX}" setsid bash scripts/dlm/tofu/unlearn.sh \
    "${MODEL}" "${GPUS}" "${SPLIT}" SemanticUnlearn \
    trainer.method_args.eta="${ETA}" \
    trainer.method_args.sigma_fallback="${SIGMA_FALLBACK}" \
    trainer.method_args.center_chunk_size="${CENTER_CHUNK_SIZE}" \
    trainer.method_args.vocab_chunk_size="${VOCAB_CHUNK_SIZE}" \
    "$@" > "${LOG}" 2>&1 < /dev/null &

echo "[run_semantic_unlearn] launched pid=$! (detached via setsid); tail -f ${LOG}" >&2
