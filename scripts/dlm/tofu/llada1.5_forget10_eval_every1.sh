#!/usr/bin/env bash
# Train LLaDA-1.5 forget-only baselines on TOFU forget01 for 25 epochs and evaluate
# checkpoints at epochs 5, 10, 15, 20, and 25 after each method finishes training.
# Evaluations run sequentially, with all requested GPUs used for one checkpoint
# at a time.
#
# Methods: GA, WGA, NPO, SimNPO, BST
#
# Usage:
#   bash scripts/dlm/tofu/llada1.5_forget10_eval_every1.sh [gpus]
#
# Environment:
#   OVERWRITE=1   re-evaluate checkpoint results that already exist
#   SKIP_TRAIN=1  skip training and evaluate existing checkpoints only
#   BST_LAMBDA=0.2
#   BST_TOP_K=10
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export LOSS_WEIGHT_OVERRIDE="${LOSS_WEIGHT_OVERRIDE:-scheduler}"
source scripts/dlm/lib.sh

gpus="${1:-0,1,2,3}"
epochs=25
save_every=5
methods=(DiffusionGA DiffusionWGA DiffusionNPO DiffusionSimNPO DiffusionBST)

dlm_config llada1.5 "$gpus"
tofu_split forget01
mkdir -p logs/per_epoch logs/eval

train_method() {
    local method=$1
    local task="${FORGET}/tofu_${MODEL}_${method}_forgetonly_every5"
    local log="logs/per_epoch/train_${MODEL}_${FORGET}_${method}_every5.log"

    if flag_on "${SKIP_TRAIN:-}"; then
        echo "[train] skip ${method}; using checkpoints in saves/unlearn/${task}" >&2
        return
    fi

    method_setup "$method"
    local method_args=("${METHOD_ARGS[@]}")
    if [[ "$method" == DiffusionBST ]]; then
        method_args+=(
            trainer.method_args.lambda_bst="${BST_LAMBDA:-0.2}"
            trainer.method_args.top_k="${BST_TOP_K:-10}"
        )
    fi

    echo "[train] ${method} model=${MODEL} split=${FORGET} epochs=${epochs}" \
         "save_every=${save_every} gpus=${gpus} -> ${log}" >&2

    UNLEARN_SAVE_STRATEGY=steps \
    UNLEARN_SAVE_TOTAL_LIMIT=$((epochs / save_every)) \
    tofu_unlearn \
        "$method" \
        "$task" \
        "$FORGET" \
        "$RETAIN" \
        "$HOLDOUT" \
        "$epochs" \
        trainer.args.save_only_model=true \
        +trainer.args.save_steps="$save_every" \
        "${method_args[@]}" \
        '~data.retain' \
        2>&1 | tee "$log"
}

evaluate_method() {
    local method=$1
    local task="${FORGET}/tofu_${MODEL}_${method}_forgetonly_every5"
    local checkpoint_root="saves/unlearn/${task}"
    local checkpoint step epoch eval_task

    [[ -d "$checkpoint_root" ]] || die "missing checkpoint directory: ${checkpoint_root}"

    eval_plan_reset
    while IFS= read -r checkpoint; do
        [[ -n "$checkpoint" ]] || continue
        step=${checkpoint##*checkpoint-}
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        (( step % save_every == 0 && step <= epochs )) || continue
        epoch=$step
        eval_task=$(printf \
            "%s/tofu_%s_%s_forgetonly_ep%02d" \
            "$FORGET" "$MODEL" "$method" "$epoch"
        )
        eval_plan_add "$eval_task" "$checkpoint" "$FORGET"
    done < <(
        find "$checkpoint_root" -maxdepth 1 -type d -name 'checkpoint-*' \
            | awk -F'checkpoint-' '{print $2 "\t" $0}' \
            | sort -n \
            | cut -f2-
    )

    echo "[eval] ${method}: ${#PLAN_TASKS[@]} pending checkpoint(s)," \
         "sequential with all GPUs=${gpus}" >&2
    PARALLEL=0 EVAL_GPUS="$gpus" eval_plan_run
}

fail=0
for method in "${methods[@]}"; do
    echo "==================== ${method} ====================" >&2
    if ! train_method "$method"; then
        echo "[failed] training ${method}" >&2
        fail=1
        continue
    fi
    if ! evaluate_method "$method"; then
        echo "[failed] evaluation ${method}" >&2
        fail=1
    fi
done

echo "[done] fail=${fail}; results: python scripts/dlm/show_results.py --mode both" >&2
python scripts/dlm/show_results.py --mode both || true
exit "$fail"
