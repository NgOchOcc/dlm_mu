#!/usr/bin/env bash
# Forget-only MUSE unlearning + eval for one data split, over the core methods.
# Assumes the SFT target already exists at saves/finetune/muse_<MODEL>_<split>_full.
# Runs each method from that same SFT start (independent arms), then evaluates.
#
# Usage: run_forget_only.sh <dream|llada> <gpus> <News|Books> [methods]
#   run_forget_only.sh llada 0,1,2,3 News
#   METHODS="DiffusionNPO,DiffusionWGA" run_forget_only.sh llada 4,5,6,7 Books
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
_USER_TRAIN_BS="${TRAIN_BS:-}"
_USER_GRAD_ACCUM="${GRAD_ACCUM:-}"
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

DATA_SPLIT="${3:?usage: run_forget_only.sh <dream|llada> <gpus> <News|Books> [methods]}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
EXP=unlearn/muse/dlm_forget_only

# MUSE forget corpus is 2048-token pretraining text. LLaDA has no HF gradient
# checkpointing hook (GRAD_CKPT=false), so shrink the micro-batch and compensate
# with accumulation to keep the effective batch. Explicit env still wins.
if ! flag_on "${GRAD_CKPT}"; then
    TRAIN_BS="${_USER_TRAIN_BS:-1}"
    GRAD_ACCUM="${_USER_GRAD_ACCUM:-16}"
fi

parse_methods "${METHODS:-}"        # defaults to the 5 core trainers
NPROC=$(ngpus "$GPUS")
assert_gpus

SFT_START="saves/finetune/muse_${MODEL}_${DATA_SPLIT}_full"
[[ -f "${SFT_START}/config.json" ]] || die "no SFT target at ${SFT_START} (train it first)"

muse_eval() {
    local task=$1 ckpt=$2
    local port; port=$(free_port "${EVAL_PORT:-29600}")
    echo "[eval] task=${task} split=${DATA_SPLIT} gpus=${GPUS} ckpt=${ckpt}" >&2
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${EVAL_ACCEL}" --num_processes "${NPROC}" --main_process_port "${port}" \
        src/eval.py \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${ckpt} \
        model.tokenizer_args.pretrained_model_name_or_path=${ckpt} \
        eval=muse_dlm \
        eval.muse_dlm.overwrite=true \
        eval.muse_dlm.data_split=${DATA_SPLIT} \
        eval.muse_dlm.right_shift_logits=${RIGHT_SHIFT} \
        task_name=${task}
}

for trainer in "${METHODS_LIST[@]}"; do
    task=muse_${MODEL}_${DATA_SPLIT}_${trainer}_forget_only
    out=saves/unlearn/${task}
    # Eval writes to saves/eval/<task>/MUSE_SUMMARY.json (mode=eval), not the
    # unlearn dir; the presence of that summary means this arm is fully done.
    if [[ -f "saves/eval/${task}/MUSE_SUMMARY.json" ]] && ! flag_on "${OVERWRITE:-}"; then
        echo "skip  ${task} (already done; OVERWRITE=1 to redo)" >&2
        continue
    fi
    port=$(free_port "${MAIN_PORT:-29500}")
    echo "==================== UNLEARN ${task} (${UNLEARN_EPOCHS} ep, gpus=${GPUS}, bs=${TRAIN_BS}x${GRAD_ACCUM}) ====================" >&2
    # Don't let one method's failure (OOM, eval bug, ...) abort the whole split.
    if ! CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${UNLEARN_ACCEL}" --num_processes "${NPROC}" --main_process_port "${port}" \
        src/train.py --config-name=unlearn.yaml \
        experiment=${EXP} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${SFT_START} \
        trainer=${trainer} \
        task_name=${task} \
        data_split=${DATA_SPLIT} \
        forget_split=forget \
        trainer.args.num_train_epochs=${UNLEARN_EPOCHS} \
        trainer.args.per_device_train_batch_size=${TRAIN_BS} \
        trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
        trainer.args.save_strategy=${UNLEARN_SAVE_STRATEGY:-no} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=${GRAD_CKPT} \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        eval.muse_dlm.right_shift_logits=${RIGHT_SHIFT}; then
        echo "[warn] unlearn FAILED for ${task}; skipping to next method" >&2
        continue
    fi
    muse_eval "${task}" "${out}" || echo "[warn] eval FAILED for ${task}; continuing" >&2
done
