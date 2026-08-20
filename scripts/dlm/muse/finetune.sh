set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
# Capture explicit batch overrides before lib.sh applies its shared defaults, so
# the MUSE-specific memory guard below only kicks in when the user hasn't chosen.
_USER_TRAIN_BS="${TRAIN_BS:-}"
_USER_GRAD_ACCUM="${GRAD_ACCUM:-}"
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

# MUSE finetunes on 2048-token pretraining sequences — far longer than TOFU's QA
# pairs — so the shared TRAIN_BS=4 default OOMs an 80GB card. Dream copes via
# gradient checkpointing; LLaDA has no HF checkpointing hook (GRAD_CKPT=false),
# so shrink its micro-batch and compensate with accumulation to keep the same
# effective batch (1x16 == 4x4). Explicit TRAIN_BS/GRAD_ACCUM env still win.
if ! flag_on "${GRAD_CKPT}"; then
    TRAIN_BS="${_USER_TRAIN_BS:-1}"
    GRAD_ACCUM="${_USER_GRAD_ACCUM:-16}"
fi

SFT_EPOCHS="${SFT_EPOCHS:-100}"
DATA_SPLITS="${3:-News Books}"
NPROC=$(ngpus "$GPUS")
assert_gpus

for data_split in $DATA_SPLITS; do
    task=muse_${MODEL}_${data_split}_full
    out=saves/finetune/${task}
    if [[ -f "${out}/config.json" ]] && ! flag_on "${OVERWRITE:-}"; then
        echo "skip  ${task} (already trained at ${out}; OVERWRITE=1 to redo)" >&2
        continue
    fi
    port=$(free_port "${MAIN_PORT:-29800}")
    echo "==================== SFT ${task} (${SFT_EPOCHS} epochs, gpus=${GPUS}) ====================" >&2
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${TRAIN_ACCEL}" --num_processes "${NPROC}" --main_process_port "${port}" \
        src/train.py \
        experiment=finetune/muse/dlm \
        model=${MODEL} \
        task_name=${task} \
        data_split=${data_split} \
        trainer.args.num_train_epochs=${SFT_EPOCHS} \
        trainer.args.per_device_train_batch_size=${TRAIN_BS} \
        trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
        trainer.args.save_strategy=${SAVE_STRATEGY:-epoch} \
        +trainer.args.save_total_limit=${SAVE_TOTAL_LIMIT:-2} \
        trainer.args.save_only_model=${SAVE_ONLY_MODEL:-false} \
        resume_from_checkpoint=${RESUME:-auto} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=${GRAD_CKPT} \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        eval.muse_dlm.right_shift_logits=${RIGHT_SHIFT}
done
