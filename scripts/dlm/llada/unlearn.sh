set -e
cd "$(dirname "$0")/../../.." || exit 1

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
EXPERIMENT="unlearn/tofu/dlm"
ACCELERATE_CONFIG="configs/accelerate/dlm_config.yaml" 

TRAIN_GPUS="0,1"
EVAL_GPU="0"

RIGHT_SHIFT=False
START_MODEL="saves/finetune/tofu_LLaDA-8B-Instruct_full_cont/checkpoint-4725"
MODEL_OVERRIDE=""
[ -n "${START_MODEL}" ] && MODEL_OVERRIDE="model.model_args.pretrained_model_name_or_path=${START_MODEL}"

trainers=(
    "DiffusionGradAscent"   # GA
    "DiffusionGradDiff"     # GradDiff
    "DiffusionNPO"          # NPO
    "DiffusionSimNPO"       # SimNPO
    "DiffusionWGA"          # WGA
    "DiffusionAMTU"         # AMTU (adversarial masked-transition unlearning)
)

forget_split=forget10
retain_split=retain90
holdout_split=holdout10

per_device_train_batch_size=4
gradient_accumulation_steps=4
num_train_epochs=5

for trainer in "${trainers[@]}"; do
    task_name=tofu_${MODEL}_${forget_split}_${trainer}
    save_dir=saves/unlearn/${task_name}
    MASTER_PORT=$(find_free_port 29500)
    echo "==================== ${task_name} (port ${MASTER_PORT}) ===================="

    CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
        --config_file ${ACCELERATE_CONFIG} --main_process_port ${MASTER_PORT} \
        src/train.py --config-name=unlearn.yaml \
        experiment=${EXPERIMENT} \
        model=${MODEL} \
        ${MODEL_OVERRIDE} \
        trainer=${trainer} \
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
        eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT}

    CUDA_VISIBLE_DEVICES=${EVAL_GPU} python src/eval.py \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${save_dir} \
        eval=tofu_dlm \
        eval.tofu_dlm.overwrite=true \
        eval.tofu_dlm.forget_split=${forget_split} \
        eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT} \
        task_name=${task_name}
done
