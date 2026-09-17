#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

GPUS="${GPUS:-0,1,2,3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-saves}"
export OUTPUT_ROOT

dlm_config dream "$GPUS"

mkdir -p "${OUTPUT_ROOT}/sdu_cache"
CACHE="${OUTPUT_ROOT}/sdu_cache/dream_forget01.json"

echo "=== recomputing MC mask cache -> ${CACHE} ==="
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python scripts/dlm/sdu_precompute.py \
    --model-dir "$SFT_CKPT" \
    --split forget01 \
    --num-mc-trials 32 \
    --k-percent-query 25 \
    --max-length 256 \
    --out "$CACHE"

echo "=== beta sweep (preserve_weight fixed at 0.1) ==="
MASK_CACHE="$CACHE" GPUS="$GPUS" BETAS="0 1 5 10 15" \
    bash scripts/dlm/tofu/tune_sdu_beta.sh --setting forget01 --model dream

echo "=== preserve_weight sweep (beta fixed at 10.0) ==="
MASK_CACHE="$CACHE" GPUS="$GPUS" PRESERVES="0.01 0.05 0.1 0.5 1.0" \
    bash scripts/dlm/tofu/tune_sdu_preserve.sh --setting forget01 --model dream
