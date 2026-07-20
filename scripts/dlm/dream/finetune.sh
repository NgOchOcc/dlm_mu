set -e
cd "$(dirname "$0")/../../.." || exit 1  # -> repo root (relative paths below)

# H100/NVSwitch: NCCL tries NVLS (NVLink multicast) and fails at init with
# "Cuda failure 401" (transport/nvls.cc) — fabric-manager/multicast not usable
# in this environment. Disable NVLS; P2P over NVLink still works and is fast.
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1

find_free_port() {
    local port=$1
    while python -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1', $port)); s.close(); sys.exit(0 if r==0 else 1)" 2>/dev/null; do
        echo "Port $port is busy, trying $((port + 10))" >&2
        port=$((port + 10))
    done
    echo $port
}

MODEL="Dream-v0-Instruct-7B"
EXPERIMENT="finetune/tofu/dlm"
ACCELERATE_CONFIG="configs/accelerate/dlm_config_4gpu.yaml" 
TRAIN_GPUS="0,1,2,3"
EVAL_GPU="0"

RIGHT_SHIFT=True
LOSS_WEIGHT='cart[geo_p:0.3]'

forget_split=forget10
holdout_split=holdout10

per_device_train_batch_size=4
gradient_accumulation_steps=4
num_train_epochs=100


# Continue training from an existing SFT checkpoint (loads weights only; fresh
# optimizer, since save_strategy=no saved no optimizer state). Empty = finetune
# from the raw instruct model. When set, task_name gets a _cont suffix so the
# original checkpoint is NOT overwritten. Set num_train_epochs above to the
# number of ADDITIONAL epochs you want.
CONTINUE_FROM="saves/finetune/tofu_Dream-v0-Instruct-7B_full_cont/checkpoint-4725"
task_name=tofu_${MODEL}_full
MODEL_OVERRIDE=""
if [ -n "${CONTINUE_FROM}" ]; then
    MODEL_OVERRIDE="model.model_args.pretrained_model_name_or_path=${CONTINUE_FROM} model.tokenizer_args.pretrained_model_name_or_path=${CONTINUE_FROM}"
    task_name=${task_name}_cont
fi

# Save a rolling checkpoint every 5 epochs, keeping only the most recent one
# (save_total_limit=1). HF Trainer has no "every N epochs" option, so convert to
# steps: save_steps = 5 * ceil(N_examples / (per_device * n_gpus * grad_accum)).
DATA_N=4000  # TOFU_QA_full train size
n_gpus=$(awk -F',' '{print NF}' <<< "${TRAIN_GPUS}")
eff=$(( per_device_train_batch_size * n_gpus * gradient_accumulation_steps ))
steps_per_epoch=$(( (DATA_N + eff - 1) / eff ))
save_steps=$(( 5 * steps_per_epoch ))

save_dir=saves/finetune/${task_name}
MASTER_PORT=$(find_free_port 29500)
echo "==================== ${task_name} (port ${MASTER_PORT}) ===================="

CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file ${ACCELERATE_CONFIG} --num_processes ${n_gpus} --main_process_port ${MASTER_PORT} \
    src/train.py \
    experiment=${EXPERIMENT} \
    model=${MODEL} \
    ${MODEL_OVERRIDE} \
    task_name=${task_name} \
    forget_split=${forget_split} \
    holdout_split=${holdout_split} \
    trainer.args.num_train_epochs=${num_train_epochs} \
    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
    trainer.args.save_strategy=steps \
    +trainer.args.save_steps=${save_steps} \
    +trainer.args.save_total_limit=1 \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true \
    trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
    trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
    eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT}

# ---- Evaluate the finetuned (Base) checkpoint, 1 GPU ----
CUDA_VISIBLE_DEVICES=${EVAL_GPU} python src/eval.py \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${save_dir} \
    model.tokenizer_args.pretrained_model_name_or_path=${save_dir} \
    eval=tofu_dlm \
    eval.tofu_dlm.overwrite=true \
    eval.tofu_dlm.forget_split=${forget_split} \
    eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT} \
    task_name=${task_name}
