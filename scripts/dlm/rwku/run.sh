#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

EPOCHS="${UNLEARN_EPOCHS:-5}"
task="rwku_${MODEL}_DiffusionGradAscent"
nproc=$(ngpus "$GPUS")
port=$(free_port 29600)

CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --config_file ${UNLEARN_ACCEL} --num_processes ${nproc} --main_process_port ${port} \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/rwku/dlm \
    model=${MODEL} \
    trainer=DiffusionGradAscent \
    task_name=${task} \
    trainer.args.num_train_epochs=${EPOCHS} \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=8 \
    trainer.args.save_strategy=no \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=${GRAD_CKPT} \
    trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
    trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
    eval.rwku_dlm.right_shift_logits=${RIGHT_SHIFT}

eval_port=$(free_port 29600)
eval_gpus="${EVAL_GPUS:-$GPUS}"
CUDA_VISIBLE_DEVICES=${eval_gpus} accelerate launch \
    --config_file ${EVAL_ACCEL} --num_processes $(ngpus "$eval_gpus") --main_process_port ${eval_port} \
    src/eval.py \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${task} \
    model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/${task} \
    eval=rwku_dlm \
    eval.rwku_dlm.overwrite=true \
    eval.rwku_dlm.right_shift_logits=${RIGHT_SHIFT} \
    task_name=${task}
