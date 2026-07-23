set -e
cd "$(dirname "$0")/../.." || exit 1 

export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1

find_free_port() {
    local port=$1
    while python -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1', $port)); s.close(); sys.exit(0 if r==0 else 1)" 2>/dev/null; do
        port=$((port + 10))
    done
    echo $port
}

RETAIN_SPLIT=retain99
FORGET_SPLIT=forget01
HOLDOUT_SPLIT=holdout01

EXPERIMENT="finetune/tofu/dlm"
ACCELERATE_CONFIG="configs/accelerate/dlm_config.yaml"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"

per_device_train_batch_size=4
gradient_accumulation_steps=4
DATA_N=4000
n_gpus=$(awk -F',' '{print NF}' <<< "${TRAIN_GPUS}")
eff=$(( per_device_train_batch_size * n_gpus * gradient_accumulation_steps ))
steps_per_epoch=$(( (DATA_N + eff - 1) / eff ))

run_retrain() {
    local MODEL=$1 RIGHT_SHIFT=$2 LOSS_WEIGHT=$3 GRAD_CKPT=$4 EPOCHS=$5
    local task_name=tofu_${MODEL}_${RETAIN_SPLIT}
    local save_dir=saves/finetune/${task_name}
    local save_steps=$(( 25 * steps_per_epoch ))
    local MASTER_PORT=$(find_free_port 29500)
    echo "==================== ${task_name} :: retrain (${EPOCHS} ep, port ${MASTER_PORT}) ===================="

    CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
        --config_file ${ACCELERATE_CONFIG} --num_processes ${n_gpus} --main_process_port ${MASTER_PORT} \
        src/train.py \
        experiment=${EXPERIMENT} \
        model=${MODEL} \
        task_name=${task_name} \
        data/datasets@data.train=TOFU_QA_retain \
        data.train.TOFU_QA_retain.args.hf_args.name=${RETAIN_SPLIT} \
        forget_split=${FORGET_SPLIT} \
        holdout_split=${HOLDOUT_SPLIT} \
        trainer.args.num_train_epochs=${EPOCHS} \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.save_strategy=steps \
        +trainer.args.save_steps=${save_steps} \
        +trainer.args.save_total_limit=1 \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=${GRAD_CKPT} \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT}
}

run_retrain "LLaDA-8B-Instruct" False scheduler false 300
run_retrain "Dream-v0-Instruct-7B" True 'cart[geo_p:0.3]' true 150
