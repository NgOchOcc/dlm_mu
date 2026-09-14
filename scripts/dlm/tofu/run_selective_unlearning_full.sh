#!/usr/bin/env bash
# Train and evaluate SelectiveUnlearning for Dream and LLaDA on all TOFU
# forget splits with beta=10.0 and preserve_weight=0.1.
#
# Usage:
#   bash scripts/dlm/tofu/run_selective_unlearning_full.sh [gpus]
#
# Environment:
#   DRY_RUN=1     print the six-run plan without launching jobs
#   FORCE=1       rerun completed training and evaluation
#   SKIP_TRAIN=1  evaluate existing trained models only
#   NO_EVAL=1     train without checkpoint evaluation
#   BETA=10.0
#   PRESERVE_WEIGHT=0.1
#   EPOCHS=20
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export LOSS_WEIGHT_OVERRIDE="${LOSS_WEIGHT_OVERRIDE:-scheduler}"
export EXP="${EXP:-unlearn/tofu/dlm_selective_unlearning}"
source scripts/dlm/lib.sh

gpus="${1:-0,1,2,3}"
beta="${BETA:-10.0}"
preserve_weight="${PRESERVE_WEIGHT:-0.1}"
epochs="${EPOCHS:-20}"
suffix="_selective_beta${beta}_preserve${preserve_weight}"
models=(dream llada)
splits=(forget10 forget05 forget01)

mkdir -p logs/selective_unlearning

training_complete() {
    local output_dir=$1
    [[ -f "${output_dir}/config.json" ]] || return 1
    [[ -f "${output_dir}/trainer_state.json" ]] || return 1
    [[ -f "${output_dir}/model.safetensors" ]] || \
        [[ -f "${output_dir}/model.safetensors.index.json" ]]
}

completed_eval_dir() {
    local split=$1 run_name=$2
    python - "$split" "$run_name" <<'PY'
import json
import re
import sys
from pathlib import Path

split, run_name = sys.argv[1:]
required = {
    "forget_Q_A_PARA_Prob",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "forget_truth_ratio",
    "ra_Q_A_ROUGE",
    "retain_Q_A_Prob",
    "retain_Q_A_ROUGE",
    "retain_Truth_Ratio",
    "wf_Q_A_ROUGE",
}

for summary_path in Path("saves/eval").rglob(f"{run_name}/TOFU_SUMMARY.json"):
    eval_dir = summary_path.parent
    config_path = eval_dir / ".hydra/config.yaml"
    try:
        summary = json.loads(summary_path.read_text())
        evaluation = json.loads((eval_dir / "TOFU_EVAL.json").read_text())
        config = config_path.read_text()
    except (OSError, json.JSONDecodeError):
        continue

    match = re.search(
        r"^\s*forget_split:\s*(forget(?:10|05|01))\s*$",
        config,
        re.MULTILINE,
    )
    if match is None or match.group(1) != split:
        continue
    if not required.issubset(summary) or not all(
        isinstance(summary[key], (int, float)) for key in required
    ):
        continue
    if not required.issubset(evaluation) or not all(
        isinstance(evaluation[key], dict)
        and evaluation[key].get("agg_value") is not None
        and evaluation[key].get("value_by_index")
        for key in required
    ):
        continue
    print(eval_dir)
    break
PY
}

run_one() {
    local model_key=$1 split=$2

    dlm_config "$model_key" "$gpus"
    tofu_split "$split"
    method_setup DiffusionSelectiveUnlearning

    local run_name="tofu_${MODEL}_DiffusionSelectiveUnlearning${suffix}"
    local task="${FORGET}/${run_name}"
    local output_dir="saves/unlearn/${task}"
    local eval_dir="saves/eval/${task}"
    local log="logs/selective_unlearning/${FORGET}_${MODEL}${suffix}.log"
    local completed_eval=""

    if ! flag_on "${FORCE:-}"; then
        completed_eval=$(completed_eval_dir "$FORGET" "$run_name")
        if [[ -n "$completed_eval" ]]; then
            echo "[skip] ${MODEL} ${FORGET}; evaluation complete at ${completed_eval}" >&2
            return
        fi
    fi

    echo "==================== ${MODEL} ${FORGET} ====================" >&2
    echo "[plan] epochs=${epochs} beta=${beta} preserve=${preserve_weight}" \
         "gpus=${gpus} task=${task}" >&2

    if flag_on "${DRY_RUN:-}"; then
        if training_complete "$output_dir"; then
            echo "[dry-run] training complete; would evaluate ${output_dir}" >&2
        else
            echo "[dry-run] would train then evaluate ${output_dir}" >&2
        fi
        return
    fi

    if flag_on "${SKIP_TRAIN:-}"; then
        training_complete "$output_dir" \
            || die "SKIP_TRAIN=1 but trained model is incomplete: ${output_dir}"
        echo "[train] skip; using ${output_dir}" >&2
    elif ! flag_on "${FORCE:-}" && training_complete "$output_dir"; then
        echo "[train] skip; already complete: ${output_dir}" >&2
    else
        if [[ -d "$output_dir" ]]; then
            local backup="${output_dir}.previous.$(date +%Y%m%d-%H%M%S).$$"
            echo "[train] archive previous output: ${backup}" >&2
            mv -- "$output_dir" "$backup"
        fi
        echo "[train] ${MODEL} ${FORGET} -> ${log}" >&2
        UNLEARN_SAVE_STRATEGY=no \
        UNLEARN_SAVE_TOTAL_LIMIT=1 \
        tofu_unlearn \
            DiffusionSelectiveUnlearning \
            "$task" \
            "$FORGET" \
            "$RETAIN" \
            "$HOLDOUT" \
            "$epochs" \
            trainer.args.save_only_model=true \
            trainer.method_args.beta="$beta" \
            trainer.method_args.preserve_weight="$preserve_weight" \
            2>&1 | tee "$log"
    fi

    if flag_on "${NO_EVAL:-}"; then
        echo "[eval] skip because NO_EVAL=1" >&2
        return
    fi

    if [[ -d "$eval_dir" ]]; then
        local backup="${eval_dir}.previous.$(date +%Y%m%d-%H%M%S).$$"
        echo "[eval] archive incomplete output: ${backup}" >&2
        mv -- "$eval_dir" "$backup"
    fi
    echo "[eval] ${MODEL} ${FORGET} using all GPUs=${gpus}" >&2
    EVAL_GPUS="$gpus" tofu_eval "$output_dir" "$task" "$FORGET" \
        2>&1 | tee -a "$log"
}

fail=0
for model in "${models[@]}"; do
    for split in "${splits[@]}"; do
        if ! run_one "$model" "$split"; then
            echo "[failed] model=${model} split=${split}" >&2
            fail=1
        fi
    done
done

echo "[done] fail=${fail}" >&2
echo "[results] python scripts/dlm/show_results.py --detail" >&2
exit "$fail"
