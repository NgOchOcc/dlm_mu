#!/usr/bin/env bash
# SFT-checkpoint eval + forget-only DiffusionSDU unlearning + post-unlearn eval
# on MUSE (News and Books). Per split: (1) eval the pre-unlearn SFT checkpoint,
# (2) precompute the SDU query-mask cache against that same checkpoint,
# (3) unlearn with DiffusionSDU, (4) eval the resulting unlearn checkpoint.
# Unlike GA/NPO/SimNPO/WGA, SDU needs query/answer boundaries and a stable
# per-example index, which the MUSE_forget raw pretraining corpus has neither
# of -- so it trains on MUSE_forget_knowmem (the knowmem forget_qa QA split)
# instead.
# Assumes the SFT target already exists at saves/finetune/muse_<MODEL>_<split>_full
# (train with finetune.sh first).
#
# Usage: run_sdu.sh <dream|llada|llada1.5> <gpus> [splits]
#   run_sdu.sh llada 0,1,2,3
#   run_sdu.sh llada 0,1,2,3 News
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
_USER_TRAIN_BS="${TRAIN_BS:-}"
_USER_GRAD_ACCUM="${GRAD_ACCUM:-}"
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

DATA_SPLITS="${3:-News Books}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
EXP=unlearn/muse/dlm_sdu

# MUSE knowmem QA is shorter than the raw pretraining corpus, but keep the
# same LLaDA memory guard as the other MUSE unlearn scripts for consistency.
if ! flag_on "${GRAD_CKPT}"; then
    TRAIN_BS="${_USER_TRAIN_BS:-1}"
    GRAD_ACCUM="${_USER_GRAD_ACCUM:-16}"
fi

NPROC=$(ngpus "$GPUS")
assert_gpus
mkdir -p saves/sdu_cache

precompute_cache() {
    local sft_start=$1 data_split=$2 cache=$3
    CUDA_VISIBLE_DEVICES="${GPUS%%,*}" \
        python scripts/dlm/sdu_precompute.py \
            --model-dir "$sft_start" \
            --dataset-path "muse-bench/MUSE-${data_split}" \
            --dataset-name knowmem \
            --split forget_qa \
            --num-mc-trials 32 \
            --k-percent-query 25 \
            --max-length 512 \
            --out "$cache"
}

muse_eval() {
    local task=$1 ckpt=$2 data_split=$3
    local port; port=$(free_port "${EVAL_PORT:-29600}")
    echo "[eval] task=${task} split=${data_split} gpus=${GPUS} ckpt=${ckpt}" >&2
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${EVAL_ACCEL}" --num_processes "${NPROC}" --main_process_port "${port}" \
        src/eval.py \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${ckpt} \
        model.tokenizer_args.pretrained_model_name_or_path=${ckpt} \
        eval=muse_dlm \
        eval.muse_dlm.overwrite=true \
        eval.muse_dlm.data_split=${data_split} \
        eval.muse_dlm.right_shift_logits=${RIGHT_SHIFT} \
        task_name=${task}
}

for data_split in $DATA_SPLITS; do
    SFT_START="saves/finetune/muse_${MODEL}_${data_split}_full"
    [[ -f "${SFT_START}/config.json" ]] || die "no SFT target at ${SFT_START} (train it first)"

    sft_task=muse_${MODEL}_${data_split}_full
    task=muse_${MODEL}_${data_split}_DiffusionSDU
    out=saves/unlearn/${task}
    cache="saves/sdu_cache/muse_${MODEL}_${data_split}.json"

    if [[ -f "saves/eval/${sft_task}/MUSE_SUMMARY.json" ]] && ! flag_on "${OVERWRITE:-}"; then
        echo "skip  ${sft_task} (already evaluated; OVERWRITE=1 to redo)" >&2
    else
        echo "==================== EVAL SFT ${sft_task} ====================" >&2
        muse_eval "${sft_task}" "${SFT_START}" "${data_split}" \
            || echo "[warn] SFT eval FAILED for ${sft_task}; continuing" >&2
    fi

    if [[ -f "saves/eval/${task}/MUSE_SUMMARY.json" ]] && ! flag_on "${OVERWRITE:-}"; then
        echo "skip  ${task} (already done; OVERWRITE=1 to redo)" >&2
        continue
    fi

    echo "==================== SDU-PRECOMPUTE ${data_split} ====================" >&2
    precompute_cache "${SFT_START}" "${data_split}" "${cache}"

    port=$(free_port "${MAIN_PORT:-29500}")
    echo "==================== UNLEARN ${task} (${UNLEARN_EPOCHS} ep, gpus=${GPUS}, bs=${TRAIN_BS}x${GRAD_ACCUM}) ====================" >&2
    if ! CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${UNLEARN_ACCEL}" --num_processes "${NPROC}" --main_process_port "${port}" \
        src/train.py --config-name=unlearn.yaml \
        experiment=${EXP} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${SFT_START} \
        task_name=${task} \
        data_split=${data_split} \
        "trainer.method_args.mask_cache_path=${cache}" \
        trainer.method_args.beta="${SDU_BETA:-10.0}" \
        trainer.method_args.preserve_weight="${SDU_PRESERVE_WEIGHT:-0.1}" \
        trainer.args.num_train_epochs=${UNLEARN_EPOCHS} \
        trainer.args.per_device_train_batch_size=${TRAIN_BS} \
        trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
        trainer.args.save_strategy=${UNLEARN_SAVE_STRATEGY:-no} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=${GRAD_CKPT} \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
        eval.muse_dlm.right_shift_logits=${RIGHT_SHIFT}; then
        echo "[warn] unlearn FAILED for ${task}; skipping to next split" >&2
        continue
    fi
    muse_eval "${task}" "${out}" "${data_split}" || echo "[warn] eval FAILED for ${task}; continuing" >&2
done
