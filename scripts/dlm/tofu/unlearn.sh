#!/usr/bin/env bash
set -euo pipefail
umask 0000
cd "$(dirname "$0")/../../.." || exit 1

export LOSS_WEIGHT_OVERRIDE="${LOSS_WEIGHT_OVERRIDE:-scheduler}"
source scripts/dlm/lib.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-saves}"
export OUTPUT_ROOT
mkdir -p "${OUTPUT_ROOT}/unlearn" "${OUTPUT_ROOT}/eval"

dlm_config "${1:-}" "${2:-}"
IFS=', ' read -r -a splits <<< "${3:-unlearn90}"
parse_methods "${4:-}"
if (( $# > 4 )); then shift 4; else shift $#; fi

requested_epochs="${UNLEARN_EPOCHS:-}"
IFS=', ' read -r -a epoch_overrides <<< "${EPOCHS:-}"

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

        method_setup "$method"
        extra_args=("$@")
        if flag_on "${FORGET_ONLY:-1}" && ! flag_on "${RETAIN_ALREADY_NULL:-0}"; then
            extra_args+=("~data.retain")
        fi
        tofu_unlearn "$method" "$task" "$FORGET" "$RETAIN" "$HOLDOUT" "$epochs" \
            "${METHOD_ARGS[@]}" "paths.output_dir=${OUTPUT_ROOT}/unlearn/${task}" "${extra_args[@]}"
        flag_on "${NO_EVAL:-}" || tofu_eval \
            "${OUTPUT_ROOT}/unlearn/${task}" "$task" "$FORGET" \
            "paths.output_dir=${OUTPUT_ROOT}/eval/${task}"
    done
done

python scripts/dlm/show_results.py || true
