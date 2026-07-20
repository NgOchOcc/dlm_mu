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

MODEL="LLaDA-8B-Instruct"
EXPERIMENT="unlearn/tofu/dlm_fcp"
ACCELERATE_CONFIG="configs/accelerate/dlm_config.yaml"

TRAIN_GPUS="0,1"
EVAL_GPU="0"

RIGHT_SHIFT=False
# theta_0: TOFU-finetuned checkpoint (has seen the forget facts).
START_MODEL="saves/finetune/tofu_LLaDA-8B-Instruct_full_cont/checkpoint-4725"
# theta_b: pre-exposure model (never saw the fictitious authors) = raw base
# instruct. Used ONLY during precompute; swap for a retain-only checkpoint later.
THETA_B="GSAI-ML/LLaDA-8B-Instruct"

forget_split=forget10
retain_split=retain90
holdout_split=holdout10
KAPPA=0.9

per_device_train_batch_size=4
gradient_accumulation_steps=4
num_train_epochs=5

task_name=tofu_${MODEL}_${forget_split}_DiffusionFCP
save_dir=saves/unlearn/${task_name}
CACHE_PATH=saves/unlearn/fcp_cache/${task_name}.pt

# ---- Stage 1: offline cap construction (Algorithm 1), single GPU ----
echo "==================== FCP precompute -> ${CACHE_PATH} ===================="
CUDA_VISIBLE_DEVICES=${EVAL_GPU} python src/precompute_fcp.py \
    --config-name=unlearn.yaml \
    experiment=${EXPERIMENT} \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${START_MODEL} \
    model.tokenizer_args.pretrained_model_name_or_path=${START_MODEL} \
    forget_split=${forget_split} \
    task_name=${task_name} \
    +fcp.theta_b_path=${THETA_B} \
    +fcp.output_path=${CACHE_PATH} \
    +fcp.kappa=${KAPPA} \
    +fcp.right_shift_logits=${RIGHT_SHIFT}

# ---- Stage 2: primal-dual FCP training ----
MASTER_PORT=$(find_free_port 29500)
echo "==================== ${task_name} (port ${MASTER_PORT}) ===================="
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file ${ACCELERATE_CONFIG} --main_process_port ${MASTER_PORT} \
    src/train.py --config-name=unlearn.yaml \
    experiment=${EXPERIMENT} \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${START_MODEL} \
    task_name=${task_name} \
    forget_split=${forget_split} \
    retain_split=${retain_split} \
    holdout_split=${holdout_split} \
    trainer.args.num_train_epochs=${num_train_epochs} \
    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
    trainer.args.save_strategy=no \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=false \
    trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
    trainer.method_args.cache_path=${CACHE_PATH} \
    eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT}

# ---- Stage 3: evaluate the unlearned checkpoint, 1 GPU ----
CUDA_VISIBLE_DEVICES=${EVAL_GPU} python src/eval.py \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${save_dir} \
    eval=tofu_dlm \
    eval.tofu_dlm.overwrite=true \
    eval.tofu_dlm.forget_split=${forget_split} \
    eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT} \
    task_name=${task_name}
