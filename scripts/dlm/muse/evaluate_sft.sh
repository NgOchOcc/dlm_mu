#!/usr/bin/env bash
# Evaluate saved MUSE SFT checkpoints (pre-unlearn "target" models) for
# Dream, LLaDA (and LLaDA-1.5), on News and Books.
# Reads saves/finetune/muse_<MODEL>_<split>_full (or the latest checkpoint-N/
# inside it), writes to saves/eval/muse_<MODEL>_<split>_full/MUSE_SUMMARY.json.
# Missing checkpoints are skipped, not fatal, so partial coverage still runs.
#
# Usage: evaluate_sft.sh [gpus] [models] [splits]
#   evaluate_sft.sh 0,1,2,3
#   evaluate_sft.sh 0,1,2,3 "dream llada" News
#   MODELS="llada llada1.5" evaluate_sft.sh 0,1,2,3
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

GPUS_ARG="${1:-0,1,2,3}"
MODELS="${MODELS:-${2:-dream llada}}"
DATA_SPLITS="${3:-News Books}"

muse_eval() {
    local task=$1 ckpt=$2 data_split=$3
    local nproc; nproc=$(ngpus "$GPUS")
    local port; port=$(free_port "${EVAL_PORT:-29600}")
    echo "[eval] task=${task} split=${data_split} gpus=${GPUS} ckpt=${ckpt}" >&2
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${EVAL_ACCEL}" --num_processes "${nproc}" --main_process_port "${port}" \
        src/eval.py \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path="${ckpt}" \
        model.tokenizer_args.pretrained_model_name_or_path="${ckpt}" \
        eval=muse_dlm \
        eval.muse_dlm.overwrite=true \
        eval.muse_dlm.data_split="${data_split}" \
        eval.muse_dlm.right_shift_logits="${RIGHT_SHIFT}" \
        task_name="${task}"
}

fail=0
for model in $MODELS; do
    dlm_config "$model" "$GPUS_ARG"

    for data_split in $DATA_SPLITS; do
        sft_dir="saves/finetune/muse_${MODEL}_${data_split}_full"
        if [[ ! -e "$sft_dir" ]]; then
            echo "[skip] ${sft_dir} does not exist" >&2
            continue
        fi
        ckpt="$(resolve_ckpt "$sft_dir")"
        task="muse_${MODEL}_${data_split}_full"

        if [[ -f "saves/eval/${task}/MUSE_SUMMARY.json" ]] && ! flag_on "${OVERWRITE:-}"; then
            echo "skip  ${task} (already evaluated; OVERWRITE=1 to redo)" >&2
            continue
        fi

        echo "==================== EVAL ${task} ====================" >&2
        muse_eval "${task}" "${ckpt}" "${data_split}" \
            || { echo "[warn] eval FAILED for ${task}; continuing" >&2; fail=1; }
    done
done

exit "$fail"
