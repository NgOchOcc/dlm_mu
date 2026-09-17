#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

: "${MASK_CACHE:?MASK_CACHE must point to a scripts/dlm/sdu_precompute.py cache JSON}"

MODEL="${MODEL:-llada}"
GPUS="${GPUS:-0,1,2,3}"
SPLIT="${SPLIT:-forget10}"
BETA="${BETA:-1.0}"
PRESERVE_WEIGHT="${PRESERVE_WEIGHT:-1.0}"
SUFFIX="${TASK_SUFFIX:-_sdu_beta${BETA}_preserve${PRESERVE_WEIGHT}}"

TASK_SUFFIX="${SUFFIX}" bash scripts/dlm/tofu/unlearn.sh \
    "${MODEL}" "${GPUS}" "${SPLIT}" SDU \
    trainer.method_args.beta="${BETA}" \
    trainer.method_args.preserve_weight="${PRESERVE_WEIGHT}" \
    trainer.method_args.mask_cache_path="${MASK_CACHE}" \
    "$@"
