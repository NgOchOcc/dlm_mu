set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

EPOCHS="${UNLEARN_EPOCHS:-1}"
NPROC=$(ngpus "$GPUS")
assert_gpus
read -r -a METHODS <<< "${3:-DiffusionGA DiffusionNPO DiffusionSimNPO DiffusionWGA}"

rwku_eval() {
    local task=$1
    local gpus="${EVAL_GPUS:-$GPUS}"
    local port; port=$(free_port 29600)
    CUDA_VISIBLE_DEVICES=${gpus} accelerate launch \
        --config_file ${EVAL_ACCEL} --num_processes $(ngpus "$gpus") --main_process_port ${port} \
        src/eval.py \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task} \
        model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/${task} \
        eval=rwku_dlm \
        eval.rwku_dlm.overwrite=true \
        eval.rwku_dlm.right_shift_logits=${RIGHT_SHIFT} \
        task_name=${task}
}

for method in "${METHODS[@]}"; do
    [[ -f "configs/trainer/${method}.yaml" ]] || die "no such trainer: ${method}"
    task=rwku_${MODEL}_${method}
    out=saves/unlearn/${task}
    if [[ -f "${out}/config.json" ]] && ! flag_on "${OVERWRITE:-}"; then
        echo "skip  ${task} (already unlearned at ${out}; OVERWRITE=1 to redo)" >&2
        rwku_eval "${task}"
        continue
    fi
    port=$(free_port 29500)
    echo "==================== RWKU ${task} (${EPOCHS} epochs, gpus=${GPUS}) ====================" >&2
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file ${UNLEARN_ACCEL} --num_processes ${NPROC} --main_process_port ${port} \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/rwku/dlm \
        model=${MODEL} \
        trainer=${method} \
        task_name=${task} \
        trainer.args.num_train_epochs=${EPOCHS} \
        trainer.args.per_device_train_batch_size=${TRAIN_BS} \
        trainer.args.gradient_accumulation_steps=${GRAD_ACCUM:-8} \
        trainer.args.save_strategy=no \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=${GRAD_CKPT} \
        trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        eval.rwku_dlm.right_shift_logits=${RIGHT_SHIFT}
    rwku_eval "${task}"
done
