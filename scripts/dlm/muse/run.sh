#!/usr/bin/env bash
# MUSE pipeline for masked-diffusion models. Per data split:
#   1. SFT the base model on the MUSE corpus   (optional, see SFT=)
#   1b. evaluate that SFT checkpoint            (the pre-unlearn baseline)
#   2. forget-only unlearning, once per method
#   3. evaluate each unlearned checkpoint
#
# Step 1b runs by default because the SFT numbers are the reference the
# unlearned ones are read against: forget scores should be HIGH there and drop
# after unlearning. Without them an unlearned score has nothing to be compared
# to.
#
# All methods train on the SAME forget set -- MUSE_forget_knowmem (the knowmem
# forget_qa QA split) -- so their scores are directly comparable. DiffusionSDU
# requires that QA structure (it needs a query/answer boundary to compute its
# entity-attribution score); the other methods simply follow suit.
#
# Usage:
#   run.sh <dream|llada|llada1.5> [gpus] [splits]
#
#   run.sh llada 0,1,2,3                       # SFT (if missing) + eval + all 6 methods
#   SFT=skip run.sh llada 0,1,2,3              # assume SFT exists, go straight to unlearning
#   SFT=only run.sh llada 0,1,2,3              # SFT + its eval, no unlearning
#   METHODS=SDU run.sh llada 0,1,2,3 News      # one method, one split
#   METHODS="GA,NPO" run.sh llada 0,1,2,3
#
# Flags (environment variables):
#   SFT=auto|skip|only|force   auto (default): SFT only when the checkpoint is
#                              missing. skip: never SFT. only: stop after SFT
#                              (its eval still runs). force: always re-run SFT.
#   METHODS=GA,GD,NPO,SimNPO,WGA,SDU   which methods to unlearn with
#                              (default: all six). "Diffusion" prefix optional.
#   NO_EVAL=1                  skip every evaluation (SFT and unlearned alike)
#   NO_EVAL_SFT=1              skip only the SFT-checkpoint evaluation
#   OVERWRITE=1                redo work that already has results on disk
#   SFT_EPOCHS / UNLEARN_EPOCHS / TRAIN_BS / GRAD_ACCUM / SDU_BETA /
#   SDU_PRESERVE_WEIGHT        hyperparameter overrides
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

_USER_TRAIN_BS="${TRAIN_BS:-}"
_USER_GRAD_ACCUM="${GRAD_ACCUM:-}"
source scripts/dlm/lib.sh

MODEL_KIND="${1:?usage: run.sh <dream|llada|llada1.5> [gpus] [splits]}"
dlm_config "$MODEL_KIND" "${2:-}"
DATA_SPLITS="${3:-News Books}"

SFT="${SFT:-auto}"
case "$SFT" in auto|skip|only|force) ;; *) die "SFT must be auto|skip|only|force, got '$SFT'" ;; esac

SFT_EPOCHS="${SFT_EPOCHS:-50}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
NPROC=$(ngpus "$GPUS")
assert_gpus

# MUSE SFT runs on 2048-token pretraining text -- far longer than the knowmem QA
# used for unlearning -- so LLaDA (no HF gradient-checkpointing hook) needs a
# smaller micro-batch there. Explicit TRAIN_BS/GRAD_ACCUM still win.
if ! flag_on "${GRAD_CKPT}"; then
    SFT_TRAIN_BS="${_USER_TRAIN_BS:-1}"
    SFT_GRAD_ACCUM="${_USER_GRAD_ACCUM:-16}"
else
    SFT_TRAIN_BS="${_USER_TRAIN_BS:-$TRAIN_BS}"
    SFT_GRAD_ACCUM="${_USER_GRAD_ACCUM:-$GRAD_ACCUM}"
fi

parse_methods "${METHODS:-DiffusionGA,DiffusionGD,DiffusionNPO,DiffusionSimNPO,DiffusionWGA,DiffusionSDU}"

echo "[muse] model=${MODEL} gpus=${GPUS} splits=${DATA_SPLITS} sft=${SFT}" >&2
echo "[muse] methods: ${METHODS_LIST[*]}" >&2

# ---------------------------------------------------------------- SFT --------
run_sft() {
    local data_split=$1 out=$2
    local port; port=$(free_port "${MAIN_PORT:-29800}")
    echo "==================== SFT ${data_split} (${SFT_EPOCHS} epochs) ====================" >&2
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${TRAIN_ACCEL}" --num_processes "${NPROC}" --main_process_port "${port}" \
        src/train.py \
        experiment=finetune/muse/dlm \
        model=${MODEL} \
        task_name="$(basename "$out")" \
        data_split="${data_split}" \
        trainer.args.num_train_epochs="${SFT_EPOCHS}" \
        trainer.args.per_device_train_batch_size="${SFT_TRAIN_BS}" \
        trainer.args.gradient_accumulation_steps="${SFT_GRAD_ACCUM}" \
        trainer.args.save_strategy="${SAVE_STRATEGY:-epoch}" \
        +trainer.args.save_total_limit="${SAVE_TOTAL_LIMIT:-2}" \
        trainer.args.save_only_model="${SAVE_ONLY_MODEL:-false}" \
        resume_from_checkpoint="${RESUME:-auto}" \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing="${GRAD_CKPT}" \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.method_args.right_shift_logits="${RIGHT_SHIFT}" \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        eval.muse_dlm.right_shift_logits="${RIGHT_SHIFT}"
}

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
    local extra=("trainer.method_args.loss_weight_type=\"${LOSS_WEIGHT}\"")
    if [[ "$trainer" == "DiffusionSDU" ]]; then
        exp="unlearn/muse/dlm_sdu"
        extra+=("trainer.method_args.mask_cache_path=${cache}"
                "trainer.method_args.beta=${SDU_BETA:-10.0}"
                "trainer.method_args.preserve_weight=${SDU_PRESERVE_WEIGHT:-0.1}")
    fi

    echo "==================== UNLEARN ${task} (${UNLEARN_EPOCHS} ep) ====================" >&2
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${UNLEARN_ACCEL}" --num_processes "${NPROC}" --main_process_port "${port}" \
        src/train.py --config-name=unlearn.yaml \
        experiment="${exp}" \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path="${start}" \
        trainer="${trainer}" \
        task_name="${task}" \
        data_split="${data_split}" \
        trainer.args.num_train_epochs="${UNLEARN_EPOCHS}" \
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
fail=0
for data_split in $DATA_SPLITS; do
    sft_task="muse_${MODEL}_${data_split}_full"
    sft_dir="saves/finetune/${sft_task}"

    # ---- step 1: SFT ----
    case "$SFT" in
        force) run_sft "$data_split" "$sft_dir" ;;
        skip)  ;;
        auto|only)
            if [[ -f "${sft_dir}/config.json" ]]; then
                echo "skip  ${sft_task} (already trained; SFT=force to redo)" >&2
            else
                run_sft "$data_split" "$sft_dir"
            fi
            ;;
    esac

    [[ -e "$sft_dir" ]] || { echo "[warn] no SFT checkpoint at ${sft_dir}; skipping ${data_split}" >&2; fail=1; continue; }
    start="$(resolve_ckpt "$sft_dir")"

    # ---- step 1b: evaluate the SFT checkpoint (the pre-unlearn baseline) ----
    # Runs by default: these numbers are the reference every unlearned
    # checkpoint is read against (forget scores should be HIGH here and drop
    # after unlearning). NO_EVAL_SFT=1 turns it off.
    if flag_on "${NO_EVAL_SFT:-}" || flag_on "${NO_EVAL:-}"; then
        :
    elif [[ -f "saves/eval/${sft_task}/MUSE_SUMMARY.json" ]] && ! flag_on "${OVERWRITE:-}"; then
        echo "skip  eval ${sft_task} (already evaluated; OVERWRITE=1 to redo)" >&2
    else
        echo "==================== EVAL SFT ${sft_task} ====================" >&2
        muse_eval "$sft_task" "$start" "$data_split" \
            || { echo "[warn] SFT eval FAILED for ${sft_task}; continuing" >&2; fail=1; }
    fi

    [[ "$SFT" == "only" ]] && continue

    # ---- step 2: unlearn each method ----
    for trainer in "${METHODS_LIST[@]}"; do
        task="muse_${MODEL}_${data_split}_${trainer}"
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

echo "[muse] done (fail=${fail})" >&2
exit "$fail"
