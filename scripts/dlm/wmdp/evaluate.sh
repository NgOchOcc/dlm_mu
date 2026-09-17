#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

CKPT_ARG="${1:?usage: bash scripts/dlm/wmdp/evaluate.sh <ckpt_path> [dream|llada|llada1.5] [gpus] [task_name]}"
MODEL_KIND="${2:-llada}"
dlm_config "${MODEL_KIND}" "${3:-}"

CKPT="$(resolve_ckpt "$CKPT_ARG")"
TASK_NAME="${4:-wmdp_$(basename "${CKPT}")}"
OUTPUT_DIR="${OUTPUT_DIR:-saves/eval/${TASK_NAME}}"

nproc=$(ngpus "$GPUS")
port=$(free_port "${EVAL_PORT:-29600}")

CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --config_file "${EVAL_ACCEL}" --num_processes "${nproc}" --main_process_port "${port}" \
    src/eval.py \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path="${CKPT}" \
    model.tokenizer_args.pretrained_model_name_or_path="${CKPT}" \
    eval=wmdp_dlm \
    eval.wmdp_dlm.overwrite="${OVERWRITE:-true}" \
    eval.wmdp_dlm.right_shift_logits="${RIGHT_SHIFT}" \
    eval.wmdp_dlm.batch_size="${WMDP_EVAL_BS:-8}" \
    task_name="${TASK_NAME}" \
    "paths.output_dir=${OUTPUT_DIR}"

echo "[wmdp-eval] summary: ${OUTPUT_DIR}/WMDP_SUMMARY.json" >&2
