#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

SFT_EPOCHS="${SFT_EPOCHS:-10}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
NPROC=$(ngpus "$GPUS")
DATA_SPLITS=(News Books)
TRAINERS=(DiffusionGradAscent DiffusionGradDiff DiffusionNPO DiffusionSimNPO DiffusionWGA)

muse_eval() {
    local task=$1 ckpt=$2 data_split=$3
    local gpus="${EVAL_GPUS:-$GPUS}"
    local port; port=$(free_port 29600)
    CUDA_VISIBLE_DEVICES=${gpus} accelerate launch \
        --config_file ${EVAL_ACCEL} --num_processes $(ngpus "$gpus") --main_process_port ${port} \
        src/eval.py \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${ckpt} \
        model.tokenizer_args.pretrained_model_name_or_path=${ckpt} \
        eval=muse_dlm \
        eval.muse_dlm.overwrite=true \
        eval.muse_dlm.data_split=${data_split} \
        eval.muse_dlm.right_shift_logits=${RIGHT_SHIFT} \
        task_name=${task}
}

muse_unlearn() {
    local experiment=$1 trainer=$2 data_split=$3 forget=$4 start=$5 task=$6
    local port; port=$(free_port 29500)
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file ${UNLEARN_ACCEL} --num_processes ${NPROC} --main_process_port ${port} \
        src/train.py --config-name=unlearn.yaml \
        experiment=${experiment} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${start} \
        trainer=${trainer} \
        task_name=${task} \
        data_split=${data_split} \
        forget_split=${forget} \
        trainer.args.num_train_epochs=${UNLEARN_EPOCHS} \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.gradient_accumulation_steps=4 \
        trainer.args.save_strategy=no \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=${GRAD_CKPT} \
        trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        eval.muse_dlm.right_shift_logits=${RIGHT_SHIFT}
    muse_eval "${task}" "saves/unlearn/${task}" "${data_split}"
}

for data_split in "${DATA_SPLITS[@]}"; do
    task=muse_${MODEL}_${data_split}_full
    port=$(free_port 29500)
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file ${UNLEARN_ACCEL} --num_processes ${NPROC} --main_process_port ${port} \
        src/train.py \
        experiment=finetune/muse/dlm \
        model=${MODEL} \
        task_name=${task} \
        data_split=${data_split} \
        trainer.args.num_train_epochs=${SFT_EPOCHS} \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.gradient_accumulation_steps=4 \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=${GRAD_CKPT} \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        eval.muse_dlm.right_shift_logits=${RIGHT_SHIFT}
done

for data_split in "${DATA_SPLITS[@]}"; do
    start=saves/finetune/muse_${MODEL}_${data_split}_full
    for trainer in "${TRAINERS[@]}"; do
        muse_unlearn unlearn/muse/dlm "${trainer}" "${data_split}" forget "${start}" \
            muse_${MODEL}_${data_split}_${trainer}
    done
done

for data_split in "${DATA_SPLITS[@]}"; do
    start=saves/finetune/muse_${MODEL}_${data_split}_full
    for trainer in "${TRAINERS[@]}"; do
        for scal in forget_1 forget_2 forget_3 forget_4; do
            muse_unlearn unlearn/muse/dlm_scalability "${trainer}" "${data_split}" "${scal}" "${start}" \
                muse_${MODEL}_${data_split}_${trainer}_scal_${scal}
        done
    done
done

for data_split in "${DATA_SPLITS[@]}"; do
    for trainer in "${TRAINERS[@]}"; do
        model_path=saves/finetune/muse_${MODEL}_${data_split}_full
        for sust in forget_1 forget_2 forget_3 forget_4; do
            task=muse_${MODEL}_${data_split}_${trainer}_sust_${sust}
            muse_unlearn unlearn/muse/dlm_sustainability "${trainer}" "${data_split}" "${sust}" "${model_path}" "${task}"
            model_path=saves/unlearn/${task}
        done
    done
done
