#!/usr/bin/env bash
# Train LLaDA-1.5 forget-only baselines from the retain100 SFT checkpoint.
#
# Schedules:
#   forget05: train 15 epochs; evaluate epochs 3, 6, 9, 12, 15
#   forget10: train 10 epochs; evaluate every epoch, 1 through 10
#
# Methods: GA, WGA, SimNPO, NPO, BST
#
# Usage:
#   bash scripts/dlm/tofu/llada15_forget05_forget10_eval_epochs.sh \
#       [gpus] [hydra overrides...]
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
if (( $# > 0 )); then
    shift
fi

methods=(DiffusionGA DiffusionWGA)
splits=(forget10)

dlm_config llada1.5 "$gpus"
mkdir -p logs/per_epoch logs/eval

schedule_for_split() {
    case "$1" in
        # forget05) epochs=15; eval_every=3 ;;
        forget10) epochs=10; eval_every=1 ;;
        *) die "unsupported split: $1" ;;
    esac
}

train_method() {
    local method=$1 task=$2 log=$3
    shift 3

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
         "save=every_epoch gpus=${gpus} base=${SFT_CKPT} -> ${log}" >&2

    UNLEARN_SAVE_STRATEGY=epoch \
    UNLEARN_SAVE_TOTAL_LIMIT="$epochs" \
    tofu_unlearn \
        "$method" \
        "$task" \
        "$FORGET" \
        "$RETAIN" \
        "$HOLDOUT" \
        "$epochs" \
        trainer.args.save_only_model=true \
        "${method_args[@]}" \
        '~data.retain' \
        "$@" \
        2>&1 | tee "$log"
}

evaluate_method() {
    local method=$1 task=$2
    local checkpoint_root="saves/unlearn/${task}"
    local epoch checkpoint eval_task
    local checkpoints=()

    [[ -d "$checkpoint_root" ]] || die "missing checkpoint directory: ${checkpoint_root}"
    mapfile -t checkpoints < <(
        find "$checkpoint_root" -maxdepth 1 -type d -name 'checkpoint-*' \
            | awk -F'checkpoint-' '{print $2 "\t" $0}' \
            | sort -n \
            | cut -f2-
    )

    eval_plan_reset
    for (( epoch=eval_every; epoch<=epochs; epoch+=eval_every )); do
        checkpoint="${checkpoints[$((epoch - 1))]:-}"
        [[ -n "$checkpoint" ]] || die \
            "missing epoch ${epoch} checkpoint for ${task}; found ${#checkpoints[@]}"
        eval_task=$(printf \
            "%s/tofu_%s_%s_forgetonly_ep%02d" \
            "$FORGET" "$MODEL" "$method" "$epoch"
        )
        eval_plan_add "$eval_task" "$checkpoint" "$FORGET"
    done

    echo "[eval] ${method}: ${#PLAN_TASKS[@]} pending checkpoint(s)," \
         "epochs every ${eval_every}, all GPUs=${gpus}" >&2
    PARALLEL=0 EVAL_GPUS="$gpus" eval_plan_run
}

fail=0
for split in "${splits[@]}"; do
    tofu_split "$split"
    schedule_for_split "$split"
    echo "==================== ${MODEL} / ${FORGET} ====================" >&2

    for method in "${methods[@]}"; do
        task="${FORGET}/tofu_${MODEL}_${method}_forgetonly_epochcheckpoints"
        log="logs/per_epoch/train_${MODEL}_${FORGET}_${method}_epochcheckpoints.log"
        echo "-------------------- ${method} --------------------" >&2

        if ! train_method "$method" "$task" "$log" "$@"; then
            echo "[failed] training ${method} on ${FORGET}" >&2
            fail=1
            continue
        fi
        if ! evaluate_method "$method" "$task"; then
            echo "[failed] evaluation ${method} on ${FORGET}" >&2
            fail=1
        fi
    done
done

echo "[done] fail=${fail}; results: python scripts/dlm/show_results.py --mode both" >&2
python scripts/dlm/show_results.py --mode both || true
exit "$fail"
