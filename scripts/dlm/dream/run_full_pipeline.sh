#!/bin/bash
# End-to-end Dream x TOFU pipeline (run from anywhere; cd's to repo root).
#
#   Step 1  Finetune Dream-v0-Instruct-7B on TOFU_QA_full, 300 epochs -> "Base" model
#   Step 2  Evaluate the Base checkpoint (eval=tofu_dlm)
#   Step 3  Unlearn 5 methods (GA, GD, NPO, WGA, SimNPO) from the Base checkpoint
#   Step 4  Evaluate each unlearned checkpoint
#   Step 5  Aggregate every eval into saves/eval/tofu_results_summary.csv
#
# Metrics per checkpoint: model_utility, forget_truth_ratio, and per-split
# probability/rougeL. forget_quality is excluded (no retain-only reference).
#
# Assumes the conda env is already active with a ROCm/CUDA torch + dllm installed.
# Dream needs right_shift_logits=True and CART loss weighting.
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1  # -> repo root

# --- RCCL/NCCL: NVLS multicast unusable in this env; P2P still works. -----------
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1

# --- Knobs ---------------------------------------------------------------------
MODEL="Dream-v0-Instruct-7B"
RIGHT_SHIFT=True                  # Dream: right-shifted logits
LOSS_WEIGHT='cart[geo_p:0.3]'     # Dream: CART loss weighting (finetune + unlearn)
forget_split=forget10
retain_split=retain90
holdout_split=holdout10

FT_GPUS="0,1,2,3,4,5,6,7"         # GPUs for finetuning (ZeRO-3, 8x MI300)
UNLEARN_GPUS="0,1,2,3,4,5,6,7"    # GPUs for each unlearning run (8x MI300)
EVAL_GPU="0"                      # eval is single-process -> one GPU

FT_EPOCHS=300                     # finetune epochs
FT_ACCEL="configs/accelerate/dlm_config_4gpu.yaml"
UNLEARN_ACCEL="configs/accelerate/dlm_config.yaml"
per_device_train_batch_size=4
gradient_accumulation_steps=4
unlearn_epochs=5

# GA, GD (GradDiff), NPO, WGA, SimNPO -- AMTU intentionally omitted.
trainers=(
    "DiffusionGradAscent"
    "DiffusionGradDiff"
    "DiffusionNPO"
    "DiffusionWGA"
    "DiffusionSimNPO"
)

ft_task=tofu_${MODEL}_full
ft_dir=saves/finetune/${ft_task}

find_free_port() {
    local port=$1
    while python -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1', $port)); s.close(); sys.exit(0 if r==0 else 1)" 2>/dev/null; do
        port=$((port + 10))
    done
    echo $port
}

banner() { echo; echo "==================== $* ===================="; }

# ================================================================= Step 1: FT ==
banner "STEP 1/5  finetune ${ft_task}  (${FT_EPOCHS} epochs)"
# Rolling checkpoint every 5 epochs (HF has no per-epoch save -> convert to steps).
DATA_N=4000
n_ft_gpus=$(awk -F',' '{print NF}' <<< "${FT_GPUS}")
eff=$(( per_device_train_batch_size * n_ft_gpus * gradient_accumulation_steps ))
steps_per_epoch=$(( (DATA_N + eff - 1) / eff ))
save_steps=$(( 5 * steps_per_epoch ))
FT_PORT=$(find_free_port 29500)

CUDA_VISIBLE_DEVICES=${FT_GPUS} accelerate launch \
    --config_file ${FT_ACCEL} --num_processes ${n_ft_gpus} --main_process_port ${FT_PORT} \
    src/train.py \
    experiment=finetune/tofu/dlm \
    model=${MODEL} \
    task_name=${ft_task} \
    forget_split=${forget_split} \
    holdout_split=${holdout_split} \
    trainer.args.num_train_epochs=${FT_EPOCHS} \
    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
    trainer.args.save_strategy=steps \
    +trainer.args.save_steps=${save_steps} \
    +trainer.args.save_total_limit=1 \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=false \
    trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
    trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
    eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT}

# ================================================================ Step 2: eval =
banner "STEP 2/5  eval Base ${ft_task}"
CUDA_VISIBLE_DEVICES=${EVAL_GPU} python src/eval.py \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${ft_dir} \
    model.tokenizer_args.pretrained_model_name_or_path=${ft_dir} \
    eval=tofu_dlm \
    eval.tofu_dlm.overwrite=true \
    eval.tofu_dlm.forget_split=${forget_split} \
    eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT} \
    task_name=${ft_task}

# ========================================================= Step 3+4: unlearn ===
for trainer in "${trainers[@]}"; do
    task_name=tofu_${MODEL}_${forget_split}_${trainer}
    save_dir=saves/unlearn/${task_name}
    n_ul_gpus=$(awk -F',' '{print NF}' <<< "${UNLEARN_GPUS}")
    UL_PORT=$(find_free_port 29500)

    banner "STEP 3/5  unlearn ${task_name}"
    CUDA_VISIBLE_DEVICES=${UNLEARN_GPUS} accelerate launch \
        --config_file ${UNLEARN_ACCEL} --num_processes ${n_ul_gpus} --main_process_port ${UL_PORT} \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/dlm \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${ft_dir} \
        model.tokenizer_args.pretrained_model_name_or_path=${ft_dir} \
        trainer=${trainer} \
        task_name=${task_name} \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        holdout_split=${holdout_split} \
        trainer.args.num_train_epochs=${unlearn_epochs} \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.save_strategy=no \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=false \
        trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT}

    banner "STEP 4/5  eval ${task_name}"
    CUDA_VISIBLE_DEVICES=${EVAL_GPU} python src/eval.py \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${save_dir} \
        model.tokenizer_args.pretrained_model_name_or_path=${save_dir} \
        eval=tofu_dlm \
        eval.tofu_dlm.overwrite=true \
        eval.tofu_dlm.forget_split=${forget_split} \
        eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT} \
        task_name=${task_name}
done

# ============================================================ Step 5: CSV ======
banner "STEP 5/5  aggregate results CSV"
python scripts/dlm/aggregate_results.py
echo "Done. Results -> saves/eval/tofu_results_summary.csv"
