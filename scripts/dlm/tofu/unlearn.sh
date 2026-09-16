#!/usr/bin/env bash
# Train and optionally evaluate one or more DLM unlearning methods.
# Usage: bash scripts/dlm/tofu/unlearn.sh <dream|llada> [gpus] [splits] [methods] [overrides...]
set -euo pipefail
umask 0000
cd "$(dirname "$0")/../../.." || exit 1

export LOSS_WEIGHT_OVERRIDE="${LOSS_WEIGHT_OVERRIDE:-scheduler}"
source scripts/dlm/lib.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-saves}"
export OUTPUT_ROOT
mkdir -p "${OUTPUT_ROOT}/unlearn" "${OUTPUT_ROOT}/eval"
echo "[output] ${OUTPUT_ROOT}" >&2

dlm_config "${1:-}" "${2:-}"
IFS=', ' read -r -a splits <<< "${3:-unlearn90}"
parse_methods "${4:-}"
if (( $# > 4 )); then shift 4; else shift $#; fi

requested_epochs="${UNLEARN_EPOCHS:-}"
IFS=', ' read -r -a epoch_overrides <<< "${EPOCHS:-}"

echo "[suite] model=${MODEL} splits=${splits[*]} epochs=${EPOCHS:-${requested_epochs:-default}}" \
     "weight=${LOSS_WEIGHT} gpus=${GPUS} base=${START_MODEL:-$SFT_CKPT}" >&2
echo "[suite] methods: ${METHODS_LIST[*]}" >&2

for index in "${!splits[@]}"; do
    tofu_split "${splits[$index]}"
    epochs="${epoch_overrides[$index]:-${requested_epochs:-$SPLIT_EPOCHS}}"
    for method in "${METHODS_LIST[@]}"; do
        suffix="${TASK_SUFFIX:-}"
        flag_on "${EPOCH_IN_NAME:-}" && suffix="${suffix}_${epochs}epochs"
        task="${FORGET}/tofu_${MODEL}_${method}${suffix}"
        if flag_on "${SKIP_DONE:-}" && [[ -f "${OUTPUT_ROOT}/eval/${task}/TOFU_SUMMARY.json" ]]; then
            echo "[skip] ${task} already evaluated" >&2
            continue
        fi

        echo "==================== ${task} (${epochs} epochs) ====================" >&2
        method_setup "$method"
        tofu_unlearn "$method" "$task" "$FORGET" "$RETAIN" "$HOLDOUT" "$epochs" \
            "${METHOD_ARGS[@]}" "paths.output_dir=${OUTPUT_ROOT}/unlearn/${task}" "$@"
        flag_on "${NO_EVAL:-}" || tofu_eval \
            "${OUTPUT_ROOT}/unlearn/${task}" "$task" "$FORGET" \
            "paths.output_dir=${OUTPUT_ROOT}/eval/${task}"
    done
done

echo "[suite] done. Table: python scripts/dlm/show_results.py" >&2
