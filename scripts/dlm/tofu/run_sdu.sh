#!/usr/bin/env bash
# Train SDU (offline Monte Carlo mask selection) on the TOFU forget split.
# Identical to Selective Unlearning except the query/answer entity masks come
# from a precomputed cache (scripts/dlm/sdu_precompute.py) instead of the
# hand-written ADU entity heuristic.
#
# Usage:
#   MASK_CACHE=saves/sdu_cache/forget10_retain100.json \
#     bash scripts/dlm/tofu/run_sdu.sh
#   MODEL=dream GPUS=0,1 SPLIT=forget05 BETA=1.0 \
#     MASK_CACHE=saves/sdu_cache/forget05_retain100.json \
#     bash scripts/dlm/tofu/run_sdu.sh
#
# Env vars:
#   MASK_CACHE   path to the sdu_precompute.py cache JSON  (required)
#   MODEL        llada | dream                  (default: llada)
#   GPUS         comma-separated device ids     (default: 0,1,2,3)
#   SPLIT        forget10 | forget05 | forget01 (default: forget10)
#   BETA         suppression strength           (default: 1.0; value >= 0)
#   PRESERVE_WEIGHT preservation-loss weight     (default: 1.0; value >= 0)
#   TASK_SUFFIX  suffix for the save directory  (default includes both weights)
# Any command-line arguments are passed through to unlearn.sh / Hydra.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

: "${MASK_CACHE:?MASK_CACHE must point to a scripts/dlm/sdu_precompute.py cache JSON}"

MODEL="${MODEL:-llada}"
GPUS="${GPUS:-0,1,2,3}"
SPLIT="${SPLIT:-forget10}"
BETA="${BETA:-1.0}"
PRESERVE_WEIGHT="${PRESERVE_WEIGHT:-1.0}"
SUFFIX="${TASK_SUFFIX:-_sdu_beta${BETA}_preserve${PRESERVE_WEIGHT}}"

mkdir -p logs
LOG="logs/sdu${SUFFIX}.log"

echo "[run_sdu] model=${MODEL} gpus=${GPUS} split=${SPLIT} suffix=${SUFFIX}" >&2
echo "[run_sdu] beta=${BETA}" >&2
echo "[run_sdu] preserve_weight=${PRESERVE_WEIGHT}" >&2
echo "[run_sdu] mask_cache=${MASK_CACHE}" >&2
echo "[run_sdu] log -> ${LOG}" >&2

TASK_SUFFIX="${SUFFIX}" setsid bash scripts/dlm/tofu/unlearn.sh \
    "${MODEL}" "${GPUS}" "${SPLIT}" SDU \
    trainer.method_args.beta="${BETA}" \
    trainer.method_args.preserve_weight="${PRESERVE_WEIGHT}" \
    trainer.method_args.mask_cache_path="${MASK_CACHE}" \
    "$@" > "${LOG}" 2>&1 < /dev/null &

echo "[run_sdu] launched pid=$! (detached via setsid); tail -f ${LOG}" >&2
