#!/usr/bin/env bash
# Train Dream forget-only baselines on TOFU forget10 for 5 epochs and evaluate
# every epoch checkpoint after each method finishes training.
# Evaluations run sequentially, with all requested GPUs used for one checkpoint
# at a time.
#
# Methods: GA, WGA, NPO, SimNPO, BST
#
# Usage:
#   bash scripts/dlm/tofu/dream_forget10_eval_every1.sh [gpus]
#
# Environment:
#   OVERWRITE=1   retrain and re-evaluate completed results
#   SKIP_TRAIN=1  skip training and evaluate existing checkpoints only
#   BST_LAMBDA=0.2
#   BST_TOP_K=10
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export LOSS_WEIGHT_OVERRIDE="${LOSS_WEIGHT_OVERRIDE:-scheduler}"
source scripts/dlm/lib.sh

gpus="${1:-0,1,2,3}"
epochs=5
methods=(DiffusionGA DiffusionWGA DiffusionNPO DiffusionSimNPO DiffusionBST)

dlm_config dream "$gpus"
tofu_split forget10
mkdir -p logs/per_epoch logs/eval

checkpoint_complete() {
    local checkpoint=$1
    [[ -f "${checkpoint}/config.json" ]] || return 1
    [[ -f "${checkpoint}/trainer_state.json" ]] || return 1
    [[ -f "${checkpoint}/model.safetensors" ]] || \
        [[ -f "${checkpoint}/model.safetensors.index.json" ]]
}

checkpoint_resumable() {
    local checkpoint=$1
    checkpoint_complete "$checkpoint" || return 1
    [[ -f "${checkpoint}/optimizer.pt" ]] || return 1
    [[ -f "${checkpoint}/scheduler.pt" ]] || return 1
    compgen -G "${checkpoint}/rng_state*.pth" >/dev/null
}

latest_resumable_checkpoint() {
    local output_dir=$1 checkpoint
    while IFS= read -r checkpoint; do
        checkpoint_resumable "$checkpoint" && echo "$checkpoint"
    done < <(
        find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
            | sort -t- -k2 -n
    ) | tail -1
}

training_complete() {
    local output_dir=$1 state
    state="${output_dir}/trainer_state.json"
    [[ -f "$state" ]] || return 1
    python - "$state" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    state = json.load(handle)

step = state.get("global_step")
max_steps = state.get("max_steps")
raise SystemExit(0 if step is not None and max_steps is not None and step >= max_steps else 1)
PY
}

evaluation_complete() {
    local eval_dir=$1
    python - "${eval_dir}/TOFU_SUMMARY.json" "${eval_dir}/TOFU_EVAL.json" <<'PY'
import json
import sys
from pathlib import Path

required = {
    "forget_Q_A_PARA_Prob",
    "forget_Q_A_PARA_ROUGE",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "forget_truth_ratio",
    "ra_Q_A_ROUGE",
    "retain_Q_A_Prob",
    "retain_Q_A_ROUGE",
    "retain_Truth_Ratio",
    "wf_Q_A_ROUGE",
}

try:
    summary = json.loads(Path(sys.argv[1]).read_text())
    evaluation = json.loads(Path(sys.argv[2]).read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)

valid = (
    required.issubset(summary)
    and all(summary[key] is not None for key in required)
    and required.issubset(evaluation)
    and all(
        isinstance(evaluation[key], dict)
        and evaluation[key].get("agg_value") is not None
        and evaluation[key].get("value_by_index")
        for key in required
    )
)
raise SystemExit(0 if valid else 1)
PY
}

checkpoint_epoch() {
    python - "$1/trainer_state.json" <<'PY'
import json
import math
import sys

with open(sys.argv[1]) as handle:
    state = json.load(handle)

epoch = state.get("epoch")
if epoch is None or not math.isclose(epoch, round(epoch), abs_tol=1e-6):
    raise SystemExit(1)
print(round(epoch))
PY
}

train_method() {
    local method=$1
    local task="${FORGET}/tofu_${MODEL}_${method}_forgetonly_every5"
    local output_dir="saves/unlearn/${task}"
    local log="logs/per_epoch/train_${MODEL}_${FORGET}_${method}_every5.log"

    if flag_on "${SKIP_TRAIN:-}"; then
        echo "[train] skip ${method}; using checkpoints in ${output_dir}" >&2
        return
    fi

    if ! flag_on "${OVERWRITE:-}" \
        && training_complete "$output_dir" \
        && checkpoint_complete "$output_dir"; then
        echo "[train] skip ${method}; training already complete (${output_dir})" >&2
        return
    fi

    local resume_checkpoint=""
    if ! flag_on "${OVERWRITE:-}" && [[ -d "$output_dir" ]]; then
        resume_checkpoint=$(latest_resumable_checkpoint "$output_dir")
    fi
    if [[ -n "$resume_checkpoint" ]]; then
        echo "[train] resume ${method} from ${resume_checkpoint}" >&2
    elif [[ -d "$output_dir" ]]; then
        local backup="${output_dir}.previous.$(date +%Y%m%d-%H%M%S).$$"
        echo "[train] archive existing output before restart: ${backup}" >&2
        mv -- "$output_dir" "$backup"
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
         "save_strategy=epoch gpus=${gpus} -> ${log}" >&2

    UNLEARN_SAVE_STRATEGY=epoch \
    UNLEARN_SAVE_TOTAL_LIMIT=$epochs \
    tofu_unlearn \
        "$method" \
        "$task" \
        "$FORGET" \
        "$RETAIN" \
        "$HOLDOUT" \
        "$epochs" \
        trainer.args.save_only_model=false \
        resume_from_checkpoint="${resume_checkpoint:-null}" \
        "${method_args[@]}" \
        '~data.retain' \
        2>&1 | tee "$log"
}

evaluate_method() {
    local method=$1
    local task="${FORGET}/tofu_${MODEL}_${method}_forgetonly_every5"
    local checkpoint_root="saves/unlearn/${task}"
    local checkpoint epoch eval_task eval_dir

    [[ -d "$checkpoint_root" ]] || die "missing checkpoint directory: ${checkpoint_root}"

    eval_plan_reset
    while IFS= read -r checkpoint; do
        [[ -n "$checkpoint" ]] || continue
        if ! checkpoint_complete "$checkpoint"; then
            echo "[eval] skip incomplete checkpoint: ${checkpoint}" >&2
            continue
        fi
        if ! epoch=$(checkpoint_epoch "$checkpoint"); then
            echo "[eval] skip non-epoch checkpoint: ${checkpoint}" >&2
            continue
        fi
        (( epoch >= 1 && epoch <= epochs )) || continue
        eval_task=$(printf \
            "%s/tofu_%s_%s_forgetonly_ep%02d" \
            "$FORGET" "$MODEL" "$method" "$epoch"
        )
        eval_dir="saves/eval/${eval_task}"
        if ! flag_on "${OVERWRITE:-}" && evaluation_complete "$eval_dir"; then
            echo "[eval] skip ${eval_task}; all metrics already complete" >&2
            continue
        fi
        PLAN_TASKS+=("$eval_task")
        PLAN_CKPTS+=("$checkpoint")
        PLAN_FORGETS+=("$FORGET")
    done < <(
        find "$checkpoint_root" -maxdepth 1 -type d -name 'checkpoint-*' \
            | awk -F'checkpoint-' '{print $2 "\t" $0}' \
            | sort -n \
            | cut -f2-
    )

    echo "[eval] ${method}: ${#PLAN_TASKS[@]} pending epoch checkpoint(s)," \
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
