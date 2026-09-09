#!/usr/bin/env bash
# Train selective Auto Unlearning on TOFU. The frozen teacher supplies a
# conditional distribution and a prompt-masked distribution on the same noised
# answer; the conditional distribution remains the base while probability added
# by the prompt is suppressed in the target.
#
# Usage:
#   bash scripts/dlm/tofu/run_auto_unlearning.sh
#   BETA=0.5 bash scripts/dlm/tofu/run_auto_unlearning.sh
#   MODEL=dream GPUS=0,1 SPLIT=forget05 BETA=1.0 \
#     bash scripts/dlm/tofu/run_auto_unlearning.sh
#
# Env vars (all optional):
#   MODEL        llada | dream                  (default: llada)
#   GPUS         comma-separated device ids     (default: 0,1,2,3)
#   SPLIT        forget10 | forget05 | forget01 (default: forget10)
#   BETA         suppression strength             (default: 1.0; value >= 0)
#   TASK_SUFFIX  suffix for the save directory   (default: _auto_beta${BETA})
# Any command-line arguments are passed through to unlearn.sh / Hydra.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODEL="${MODEL:-llada}"
GPUS="${GPUS:-0,1,2,3}"
SPLIT="${SPLIT:-forget10}"
BETA="${BETA:-1.0}"
SUFFIX="${TASK_SUFFIX:-_auto_beta${BETA}}"

mkdir -p logs
LOG="logs/auto_unlearning${SUFFIX}.log"

echo "[run_auto_unlearning] model=${MODEL} gpus=${GPUS} split=${SPLIT} suffix=${SUFFIX}" >&2
echo "[run_auto_unlearning] beta=${BETA}" >&2
echo "[run_auto_unlearning] log -> ${LOG}" >&2

TASK_SUFFIX="${SUFFIX}" setsid bash scripts/dlm/tofu/unlearn.sh \
    "${MODEL}" "${GPUS}" "${SPLIT}" AutoUnlearning \
    trainer.method_args.beta="${BETA}" \
    "$@" > "${LOG}" 2>&1 < /dev/null &

echo "[run_auto_unlearning] launched pid=$! (detached via setsid); tail -f ${LOG}" >&2
