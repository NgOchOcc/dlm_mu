#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
MODEL_KIND="${1:-}"
dlm_config "${1:-}" "${2:-}"
if (( $# > 2 )); then shift 2; else shift $#; fi

: "${START_MODEL:?WMDP has no TOFU-style SFT checkpoint; set START_MODEL to a base DLM instruct checkpoint}"

DATA_SPLIT="${DATA_SPLIT:-cyber}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
NPROC=$(ngpus "$GPUS")
TRAINERS=(DiffusionGA DiffusionGD DiffusionNPO DiffusionSimNPO DiffusionWGA)

wmdp_unlearn() {
    local trainer=$1 task=$2
    local port; port=$(free_port 29500)
    local extra_args=()
    flag_on "${FORGET_ONLY:-0}" && extra_args+=("~data.retain")
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${UNLEARN_ACCEL}" --num_processes "${NPROC}" --main_process_port "${port}" \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/wmdp/dlm \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${START_MODEL} \
        trainer=${trainer} \
        task_name=${task} \
        data_split=${DATA_SPLIT} \
        trainer.args.num_train_epochs=${UNLEARN_EPOCHS} \
        trainer.args.per_device_train_batch_size="${TRAIN_BS}" \
        trainer.args.gradient_accumulation_steps="${GRAD_ACCUM}" \
        trainer.args.save_strategy="${UNLEARN_SAVE_STRATEGY:-epoch}" \
        +trainer.args.save_total_limit="${UNLEARN_SAVE_TOTAL_LIMIT:-1}" \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing="${GRAD_CKPT}" \
        trainer.method_args.right_shift_logits="${RIGHT_SHIFT}" \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        "paths.output_dir=saves/unlearn/${task}" \
        "${extra_args[@]}" "$@"

    flag_on "${NO_EVAL:-}" || bash scripts/dlm/wmdp/evaluate.sh \
        "saves/unlearn/${task}" "${MODEL_KIND}" "${GPUS}" "${task}"
}

for trainer in "${TRAINERS[@]}"; do
    task="wmdp_${MODEL}_${DATA_SPLIT}_${trainer}"
    wmdp_unlearn "${trainer}" "${task}"
done
