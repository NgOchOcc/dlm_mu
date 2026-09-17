#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODEL="${MODEL:-llada}"
GPUS="${GPUS:-0,1,2,3}"
SPLIT="${SPLIT:-forget10}"
BETA="${BETA:-1.0}"
PRESERVE_WEIGHT="${PRESERVE_WEIGHT:-1.0}"
SUFFIX="${TASK_SUFFIX:-_selective_beta${BETA}_preserve${PRESERVE_WEIGHT}}"

TASK_SUFFIX="${SUFFIX}" bash scripts/dlm/tofu/unlearn.sh \
    "${MODEL}" "${GPUS}" "${SPLIT}" SelectiveUnlearning \
    trainer.method_args.beta="${BETA}" \
    trainer.method_args.preserve_weight="${PRESERVE_WEIGHT}" \
    "$@"
