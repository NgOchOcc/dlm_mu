#!/usr/bin/env bash
# MUSE pipeline for masked-diffusion models. Per model, per data split:
#   1. SFT the base model on the MUSE corpus    [DISABLED -- see below]
#   1b. evaluate that SFT checkpoint             [DISABLED -- see below]
#   2. forget-only unlearning, once per method
#   3. evaluate each unlearned checkpoint
#
# SFT is commented out below: this run assumes the SFT checkpoint for each
# model already exists on disk (saves/finetune/muse_<MODEL>_<split>_full) and
# only runs unlearning + eval, for every method, sequentially over models
# llada1.5 then dream.
#
# All methods train on the SAME forget set -- MUSE_forget_knowmem (the knowmem
# forget_qa QA split) -- so their scores are directly comparable. DiffusionSDU
# requires that QA structure (it needs a query/answer boundary to compute its
# entity-attribution score); the other methods simply follow suit.
#
# Usage:
#   run.sh [models] [gpus] [splits]
#
#   run.sh                                     # llada1.5 then dream, all splits, all 6 methods
#   run.sh llada1.5 0,1,2,3                    # one model only
#   METHODS=SDU run.sh llada1.5 0,1,2,3 News   # one method, one model, one split
#   METHODS="GA,NPO" run.sh "llada1.5 dream" 0,1,2,3
#
# Flags (environment variables):
#   METHODS=GA,GD,NPO,SimNPO,WGA,SDU   which methods to unlearn with
#                              (default: all six). "Diffusion" prefix optional.
#   NO_EVAL=1                  skip every evaluation
#   OVERWRITE=1                redo work that already has results on disk
#   UNLEARN_EPOCHS             epochs for every method except SDU (default: 2)
#   SDU_EPOCHS                 epochs for SDU/SelectiveUnlearning (default: 10)
#   RUN_TAG                    suffix for checkpoint/eval names (sweeps)
#   TRAIN_BS / GRAD_ACCUM / SDU_BETA / SDU_PRESERVE_WEIGHT
#                              other hyperparameter overrides
set -euo pipefail
# Run from the repository root. The job runner starts the script there and asks
# it not to change directory, so this only verifies the assumption instead of
# cd-ing: $0 is not a usable path when the script is sourced, piped in, or
# copied into a batch spool, and guessing from it silently landed elsewhere.
if [[ ! -f scripts/dlm/lib.sh ]]; then
    echo "error: run this from the repository root (no scripts/dlm/lib.sh in $PWD)." >&2
    exit 1
fi

source scripts/dlm/lib.sh

MODEL_KINDS="${1:-llada1.5 dream}"
GPUS_ARG="${2:-}"
DATA_SPLITS="${3:-News Books}"

# ---- SFT is disabled for this run; kept commented out for reference ----
# SFT="${SFT:-auto}"
# case "$SFT" in auto|skip|only|force) ;; *) die "SFT must be auto|skip|only|force, got '$SFT'" ;; esac
# SFT_EPOCHS="${SFT_EPOCHS:-50}"

UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-2}"
SDU_EPOCHS="${SDU_EPOCHS:-10}"
RUN_TAG="${RUN_TAG:-}"
TAG_SUFFIX="${RUN_TAG:+_${RUN_TAG}}"

parse_methods "${METHODS:-DiffusionGA,DiffusionGD,DiffusionNPO,DiffusionSimNPO,DiffusionWGA,DiffusionSDU}"

echo "[muse] models=${MODEL_KINDS} splits=${DATA_SPLITS}" >&2
echo "[muse] methods: ${METHODS_LIST[*]}" >&2

# ---------------------------------------------------------------- SFT --------
# Disabled -- kept for reference. To re-enable, uncomment this function and
# the "step 1"/"step 1b" block inside run_for_model() below.
# run_sft() {
#     local data_split=$1 out=$2
#     local port; port=$(free_port "${MAIN_PORT:-29800}")
#     echo "==================== SFT ${data_split} (${SFT_EPOCHS} epochs) ====================" >&2
#     CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
#         --config_file "${TRAIN_ACCEL}" --num_processes "${NPROC}" --main_process_port "${port}" \
#         src/train.py \
#         experiment=finetune/muse/dlm \
#         model=${MODEL} \
#         task_name="$(basename "$out")" \
#         data_split="${data_split}" \
#         trainer.args.num_train_epochs="${SFT_EPOCHS}" \
#         trainer.args.per_device_train_batch_size="${SFT_TRAIN_BS}" \
#         trainer.args.gradient_accumulation_steps="${SFT_GRAD_ACCUM}" \
#         trainer.args.save_strategy="${SAVE_STRATEGY:-epoch}" \
#         +trainer.args.save_total_limit="${SAVE_TOTAL_LIMIT:-2}" \
#         trainer.args.save_only_model="${SAVE_ONLY_MODEL:-false}" \
#         resume_from_checkpoint="${RESUME:-auto}" \
#         trainer.args.ddp_find_unused_parameters=true \
#         trainer.args.gradient_checkpointing="${GRAD_CKPT}" \
#         trainer.args.do_eval=false \
#         trainer.args.eval_on_start=false \
#         trainer.method_args.right_shift_logits="${RIGHT_SHIFT}" \
#         trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
#         eval.muse_dlm.right_shift_logits="${RIGHT_SHIFT}"
# }

# ------------------------------------------------------ SDU mask cache -------
precompute_sdu_cache() {
    local sft_start=$1 data_split=$2 cache=$3
    [[ -f "$cache" ]] && ! flag_on "${OVERWRITE:-}" && {
        echo "skip  sdu cache ${cache} (exists; OVERWRITE=1 to redo)" >&2; return 0; }
    mkdir -p "$(dirname "$cache")"
    echo "==================== SDU-PRECOMPUTE ${data_split} ====================" >&2
    CUDA_VISIBLE_DEVICES="${GPUS%%,*}" \
        python scripts/dlm/sdu_precompute.py \
            --model-dir "$sft_start" \
            --dataset-path "muse-bench/MUSE-${data_split}" \
            --dataset-name knowmem \
            --split forget_qa \
            --num-mc-trials "${SDU_MC_TRIALS:-32}" \
            --k-percent-query "${SDU_K_PERCENT:-25}" \
            --max-length 512 \
            --out "$cache"
}

# ------------------------------------------------------------ unlearn --------
run_unlearn() {
    local trainer=$1 data_split=$2 start=$3 task=$4 cache=$5
    local port; port=$(free_port "${MAIN_PORT:-29500}")
    # Every Diffusion* trainer config inherits DiffusionFinetune, which declares
    # loss_weight_type, so the override is valid for all of them (SDU accepts it
    # via DiffusionFinetuneTrainer even though its own compute_loss ignores it).
    local exp="unlearn/muse/dlm_forget_only_knowmem"
    local epochs="${UNLEARN_EPOCHS}"
    local extra=("trainer.method_args.loss_weight_type=\"${LOSS_WEIGHT}\"")
    # Both selective methods take the same beta / preserve_weight loss balance
    # and the same longer schedule; only SDU additionally reads its query mask
    # from the offline cache.
    case "$trainer" in
        DiffusionSDU)
            exp="unlearn/muse/dlm_sdu"
            extra+=("trainer.method_args.mask_cache_path=${cache}")
            ;;
        DiffusionSelectiveUnlearning)
            exp="unlearn/muse/dlm_selective_unlearning"
            ;;
    esac
    case "$trainer" in
        DiffusionSDU|DiffusionSelectiveUnlearning)
            epochs="${SDU_EPOCHS}"
            extra+=("trainer.method_args.beta=${SDU_BETA:-10.0}"
                    "trainer.method_args.preserve_weight=${SDU_PRESERVE_WEIGHT:-0.1}")
            ;;
    esac

    echo "==================== UNLEARN ${task} (${epochs} ep) ====================" >&2
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${UNLEARN_ACCEL}" --num_processes "${NPROC}" --main_process_port "${port}" \
        src/train.py --config-name=unlearn.yaml \
        experiment="${exp}" \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path="${start}" \
        trainer="${trainer}" \
        task_name="${task}" \
        data_split="${data_split}" \
        trainer.args.num_train_epochs="${epochs}" \
        trainer.args.per_device_train_batch_size="${TRAIN_BS}" \
        trainer.args.gradient_accumulation_steps="${GRAD_ACCUM}" \
        trainer.args.save_strategy="${UNLEARN_SAVE_STRATEGY:-no}" \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing="${GRAD_CKPT}" \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.method_args.right_shift_logits="${RIGHT_SHIFT}" \
        eval.muse_dlm.right_shift_logits="${RIGHT_SHIFT}" \
        "${extra[@]}"
}

# --------------------------------------------------------------- eval --------
muse_eval() {
    local task=$1 ckpt=$2 data_split=$3
    local port; port=$(free_port "${EVAL_PORT:-29600}")
    echo "[eval] task=${task} split=${data_split} ckpt=${ckpt}" >&2
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${EVAL_ACCEL}" --num_processes "${NPROC}" --main_process_port "${port}" \
        src/eval.py \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path="${ckpt}" \
        model.tokenizer_args.pretrained_model_name_or_path="${ckpt}" \
        eval=muse_dlm \
        eval.muse_dlm.overwrite=true \
        eval.muse_dlm.data_split="${data_split}" \
        eval.muse_dlm.right_shift_logits="${RIGHT_SHIFT}" \
        task_name="${task}"
}

# ---------------------------------------------------------------- main -------
run_for_model() {
    local model_kind=$1
    dlm_config "$model_kind" "$GPUS_ARG"
    NPROC=$(ngpus "$GPUS")
    assert_gpus

    for data_split in $DATA_SPLITS; do
        sft_task="muse_${MODEL}_${data_split}_full"
        sft_dir="saves/finetune/${sft_task}"

        # ---- step 1: SFT [DISABLED] ----
        # case "$SFT" in
        #     force) run_sft "$data_split" "$sft_dir" ;;
        #     skip)  ;;
        #     auto|only)
        #         if [[ -f "${sft_dir}/config.json" ]]; then
        #             echo "skip  ${sft_task} (already trained; SFT=force to redo)" >&2
        #         else
        #             run_sft "$data_split" "$sft_dir"
        #         fi
        #         ;;
        # esac

        [[ -e "$sft_dir" ]] || { echo "[warn] no SFT checkpoint at ${sft_dir}; skipping ${data_split}" >&2; fail=1; continue; }
        start="$(resolve_ckpt "$sft_dir")"

        # ---- step 1b: evaluate the SFT checkpoint [DISABLED] ----
        # if flag_on "${NO_EVAL_SFT:-}" || flag_on "${NO_EVAL:-}"; then
        #     :
        # elif [[ -f "saves/eval/${sft_task}/MUSE_SUMMARY.json" ]] && ! flag_on "${OVERWRITE:-}"; then
        #     echo "skip  eval ${sft_task} (already evaluated; OVERWRITE=1 to redo)" >&2
        # else
        #     echo "==================== EVAL SFT ${sft_task} ====================" >&2
        #     muse_eval "$sft_task" "$start" "$data_split" \
        #         || { echo "[warn] SFT eval FAILED for ${sft_task}; continuing" >&2; fail=1; }
        # fi

        # ---- step 2: unlearn each method ----
        for trainer in "${METHODS_LIST[@]}"; do
            # RUN_TAG separates hyperparameter-sweep runs of the same method
            # so their checkpoints and eval results never overwrite each other.
            task="muse_${MODEL}_${data_split}_${trainer}${TAG_SUFFIX}"
            out="saves/unlearn/${task}"
            cache="saves/sdu_cache/muse_${MODEL}_${data_split}.json"

            if [[ -f "saves/eval/${task}/MUSE_SUMMARY.json" ]] && ! flag_on "${OVERWRITE:-}"; then
                echo "skip  ${task} (already done; OVERWRITE=1 to redo)" >&2
                continue
            fi

            if [[ "$trainer" == "DiffusionSDU" ]]; then
                precompute_sdu_cache "$start" "$data_split" "$cache" \
                    || { echo "[warn] SDU precompute FAILED for ${data_split}; skipping SDU" >&2; fail=1; continue; }
            fi

            # One method failing (OOM, bad config, ...) must not abort the sweep.
            if ! run_unlearn "$trainer" "$data_split" "$start" "$task" "$cache"; then
                echo "[warn] unlearn FAILED for ${task}; continuing" >&2
                fail=1
                continue
            fi

            flag_on "${NO_EVAL:-}" || muse_eval "$task" "$out" "$data_split" \
                || { echo "[warn] eval FAILED for ${task}; continuing" >&2; fail=1; }
        done
    done
}

fail=0
for model_kind in $MODEL_KINDS; do
    echo "==================== MODEL ${model_kind} ====================" >&2
    run_for_model "$model_kind"
done

echo "[muse] done (fail=${fail})" >&2
exit "$fail"
