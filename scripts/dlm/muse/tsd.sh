#!/usr/bin/env bash
# TSD -- trajectory-selective distillation (retain-free) unlearning on MUSE.
#
# Reuses the MUSE forget-only experiment (retain dropped) with the DiffusionTSD
# trainer. TSD edits the frozen teacher at answer-bearing masked positions
# (demote the gold token + renormalise) and self-distils elsewhere; alpha=0 keeps
# it retain-free, so no retain stream is needed. Starts from the MUSE SFT target
# at saves/finetune/muse_<MODEL>_<split>_full, then evaluates.
#
#   usage: bash scripts/dlm/muse/tsd.sh <dream|llada> <gpus> <News|Books>
#   env:
#     DEMOTIONS="minimal belief"  demotion ablation sweep (default: minimal).
#     MASK_STATES=pathwise        "pathwise" (partial masks) | "single".
#     BELIEF_TOP_K=5              teacher top-K demotion set (demotion=belief only).
#     UNLEARN_EPOCHS=10           training epochs.
#     TRAIN_BS / GRAD_ACCUM       override the micro-batch / accumulation.
#     OVERWRITE=1                 re-run arms that already have MUSE_SUMMARY.json.
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
_USER_TRAIN_BS="${TRAIN_BS:-}"
_USER_GRAD_ACCUM="${GRAD_ACCUM:-}"
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

DATA_SPLIT="${3:?usage: tsd.sh <dream|llada> <gpus> <News|Books>}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
MASK_STATES="${MASK_STATES:-pathwise}"
BELIEF_TOP_K="${BELIEF_TOP_K:-5}"
EXP=unlearn/muse/dlm_forget_only

# MUSE forget corpus is 2048-token pretraining text. LLaDA has no HF gradient
# checkpointing hook (GRAD_CKPT=false), so shrink the micro-batch and compensate
# with accumulation to keep the effective batch (1x16 == 4x4). Explicit env wins.
if ! flag_on "${GRAD_CKPT}"; then
    TRAIN_BS="${_USER_TRAIN_BS:-1}"
    GRAD_ACCUM="${_USER_GRAD_ACCUM:-16}"
fi

NPROC=$(ngpus "$GPUS")
assert_gpus

SFT_START="saves/finetune/muse_${MODEL}_${DATA_SPLIT}_full"
[[ -f "${SFT_START}/config.json" ]] || die "no SFT target at ${SFT_START} (train it first)"

# Only tag the task dir when sweeping more than one demotion, so a plain single
# run stays at muse_<MODEL>_<split>_DiffusionTSD_forget_only.
read -r -a DEMOTION_ARR <<< "${DEMOTIONS:-minimal}"

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

for demotion in "${DEMOTION_ARR[@]}"; do
    dsuffix=""
    (( ${#DEMOTION_ARR[@]} > 1 )) && dsuffix="_${demotion}"
    task=muse_${MODEL}_${DATA_SPLIT}_DiffusionTSD${dsuffix}_forget_only
    out=saves/unlearn/${task}
    if [[ -f "saves/eval/${task}/MUSE_SUMMARY.json" ]] && ! flag_on "${OVERWRITE:-}"; then
        echo "skip  ${task} (already done; OVERWRITE=1 to redo)" >&2
        continue
    fi
    port=$(free_port "${MAIN_PORT:-29500}")
    echo "==================== UNLEARN ${task} (${UNLEARN_EPOCHS} ep, gpus=${GPUS}, bs=${TRAIN_BS}x${GRAD_ACCUM}, demotion=${demotion}) ====================" >&2
    # Don't let one arm's failure (OOM, eval bug, ...) abort the whole sweep.
    if ! CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${UNLEARN_ACCEL}" --num_processes "${NPROC}" --main_process_port "${port}" \
        src/train.py --config-name=unlearn.yaml \
        experiment=${EXP} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${SFT_START} \
        trainer=DiffusionTSD \
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
        trainer.method_args.mask_states=${MASK_STATES} \
        trainer.method_args.demotion=${demotion} \
        trainer.method_args.belief_top_k=${BELIEF_TOP_K} \
        trainer.method_args.self_distill=true \
        trainer.method_args.balanced_weight=true \
        trainer.method_args.alpha=0.0 \
        trainer.method_args.gamma=1.0 \
        trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        eval.muse_dlm.right_shift_logits=${RIGHT_SHIFT}; then
        echo "[warn] unlearn FAILED for ${task}; skipping to next" >&2
        continue
    fi
    muse_eval "${task}" "${out}" || echo "[warn] eval FAILED for ${task}; continuing" >&2
done
