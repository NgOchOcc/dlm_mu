set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export LOSS_WEIGHT_OVERRIDE="${LOSS_WEIGHT_OVERRIDE:-scheduler}"
source scripts/dlm/lib.sh

dlm_config "${1:-}" "${2:-}"
SPLITS="${3:-unlearn90}"
parse_methods "${4:-}"
if (( $# > 4 )); then shift 4; else shift $#; fi

# $3 is a comma/space-separated list of splits, run sequentially. Per-split epochs
# come from the EPOCHS env (a comma list aligned to the splits) if set, else the
# single UNLEARN_EPOCHS env, else tofu_split's default. Captured before the loop
# because tofu_split writes UNLEARN_EPOCHS itself.
#   EPOCHS=100,200,400 scripts/dlm/tofu/unlearn.sh llada 0,1,2,3 forget10,forget05,forget01 TSD
ENV_EPOCHS="${UNLEARN_EPOCHS:-}"
IFS=', ' read -r -a SPLIT_ARR <<< "$SPLITS"
IFS=', ' read -r -a EPOCH_ARR <<< "${EPOCHS:-}"

SUFFIX="${TASK_SUFFIX:-}"
echo "[suite] model=${MODEL} splits=${SPLIT_ARR[*]} epochs=${EPOCHS:-${ENV_EPOCHS:-default}}" \
     "weight=${LOSS_WEIGHT} gpus=${GPUS} base=${START_MODEL:-$SFT_CKPT}" >&2
echo "[suite] methods: ${METHODS_LIST[*]}" >&2

for idx in "${!SPLIT_ARR[@]}"; do
    tofu_split "${SPLIT_ARR[$idx]}"
    epochs="${EPOCH_ARR[$idx]:-${ENV_EPOCHS:-$SPLIT_EPOCHS}}"
    for method in "${METHODS_LIST[@]}"; do
        # EPOCH_IN_NAME=1 appends "_<epochs>epochs" to the task so epoch sweeps of
        # the same method land in distinct save/eval dirs (e.g. a TSD sweep:
        # tofu_<model>_DiffusionTSD_400epochs). show_results.py strips this suffix
        # before categorising the method but keeps it in the row label.
        suffix="${SUFFIX}"
        flag_on "${EPOCH_IN_NAME:-}" && suffix="${suffix}_${epochs}epochs"
        task="${FORGET}/tofu_${MODEL}_${method}${suffix}"
        if flag_on "${SKIP_DONE:-}" && [[ -f "saves/eval/${task}/TOFU_SUMMARY.json" ]]; then
            echo "[skip] ${task} already evaluated" >&2
            continue
        fi

        echo "==================== ${task} (${epochs} epochs) ====================" >&2
        method_setup "$method"
        tofu_unlearn "$method" "$task" "$FORGET" "$RETAIN" "$HOLDOUT" "$epochs" \
            "${METHOD_ARGS[@]}" "$@"
        flag_on "${NO_EVAL:-}" || tofu_eval "saves/unlearn/${task}" "$task" "$FORGET"
    done
done

echo "[suite] done. Table: python scripts/dlm/show_results.py" >&2
