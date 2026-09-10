#!/usr/bin/env bash
# Per-epoch unlearn + eval for LLaDA (forget10 by default).
#
# For every method we train N epochs while saving a checkpoint at the end of
# each epoch (save_only_model, ~16GB each), then evaluate *every* epoch
# checkpoint so you get a full metric curve over epochs instead of a single
# final number.
#
# Two methods run at a time, each pinned to 4 GPUs (8 GPUs total). When both
# trainings in a wave finish, all their epoch checkpoints are evaluated in
# parallel across the 8 GPUs (one checkpoint per GPU).
#
#   usage: bash scripts/dlm/tofu/per_epoch.sh [gpus] [methods] [epochs] [split]
#
#     gpus     default 0,1,2,3,4,5,6,7 (split into two 4-GPU groups)
#     methods  default GA,WGA,NPO,SimNPO
#     epochs   default 10
#     split    default forget10 (unlearn90)
#
#   per-epoch eval task names: <split>/tofu_<MODEL>_<method>_epNN
#   env: OVERWRITE=1 to redo evals that already have a TOFU_SUMMARY.json
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export LOSS_WEIGHT_OVERRIDE="${LOSS_WEIGHT_OVERRIDE:-scheduler}"
source scripts/dlm/lib.sh

gpus="${1:-0,1,2,3,4,5,6,7}"
epochs="${3:-10}"
dlm_config llada "$gpus"
tofu_split "${4:-forget10}"
parse_methods "${2:-GA,WGA,NPO,SimNPO}"

# Split the visible GPUs into two equal groups (one per concurrent method).
IFS=',' read -r -a gpu_list <<< "$gpus"
half=$(( ${#gpu_list[@]} / 2 ))
(( half > 0 )) || die "need at least 2 GPUs, got '$gpus'"
gpu_a=$(IFS=,; echo "${gpu_list[*]:0:half}")
gpu_b=$(IFS=,; echo "${gpu_list[*]:half}")

# Keep one checkpoint per epoch, model weights only.
export UNLEARN_SAVE_STRATEGY=epoch
export UNLEARN_SAVE_TOTAL_LIMIT=$(( epochs + 1 ))

mkdir -p logs/per_epoch

# train_one <gpus> <method> <main_port>
train_one() {
    local gpus=$1 method=$2 port=$3
    local task="${FORGET}/tofu_${MODEL}_${method}"
    local log="logs/per_epoch/train_${method}.log"
    echo "[train] ${method} epochs=${epochs} gpus=${gpus} port=${port} -> ${log}" >&2
    method_setup "$method"
    GPUS="$gpus" MAIN_PORT="$port" \
        tofu_unlearn "$method" "$task" "$FORGET" "$RETAIN" "$HOLDOUT" "$epochs" \
        trainer.args.save_only_model=true \
        "${METHOD_ARGS[@]}" > "$log" 2>&1
}

# eval_wave <method...> : evaluate every epoch checkpoint of the given methods,
# in parallel across all GPUs.
eval_wave() {
    eval_plan_reset
    local method task dir ep ckpt etask
    for method in "$@"; do
        task="${FORGET}/tofu_${MODEL}_${method}"
        dir="saves/unlearn/${task}"
        ep=0
        while read -r ckpt; do
            [[ -n "$ckpt" ]] || continue
            ep=$(( ep + 1 ))
            etask=$(printf "%s/tofu_%s_%s_ep%02d" "$FORGET" "$MODEL" "$method" "$ep")
            eval_plan_add "$etask" "$ckpt" "$FORGET"
        done < <(find "$dir" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | awk -F'checkpoint-' '{print $2"\t"$0}' | sort -n | cut -f2-)
    done
    PARALLEL=1 EVAL_GPUS="$gpus" eval_plan_run
}

echo "[per_epoch] model=${MODEL} split=${FORGET} epochs=${epochs}" \
     "methods=${METHODS_LIST[*]} gpuA=${gpu_a} gpuB=${gpu_b}" >&2

fail=0
for (( i=0; i<${#METHODS_LIST[@]}; i+=2 )); do
    first="${METHODS_LIST[$i]}"
    second="${METHODS_LIST[$((i+1))]:-}"

    train_one "$gpu_a" "$first" 29600 & first_pid=$!
    second_pid=""
    if [[ -n "$second" ]]; then
        train_one "$gpu_b" "$second" 29700 & second_pid=$!
    fi
    wait "$first_pid" || { echo "[per_epoch] TRAIN FAILED: $first (see logs/per_epoch/train_${first}.log)" >&2; fail=1; }
    if [[ -n "$second_pid" ]]; then
        wait "$second_pid" || { echo "[per_epoch] TRAIN FAILED: $second (see logs/per_epoch/train_${second}.log)" >&2; fail=1; }
    fi

    echo "[per_epoch] wave done, evaluating epoch checkpoints: $first ${second}" >&2
    eval_wave "$first" ${second:+$second} || fail=1
done

echo "[per_epoch] all done (fail=${fail}). Table:" >&2
python scripts/dlm/show_results.py || true
exit "$fail"
