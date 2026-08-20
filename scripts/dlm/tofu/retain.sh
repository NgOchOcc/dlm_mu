set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

dlm_config "${1:-}" "${2:-}"
SPLIT="${3:-retain100}"
if (( $# > 3 )); then shift 3; else shift $#; fi

case "$SPLIT" in
    retain100)
        tofu_split unlearn90
        DATA_ARGS=()
        ;;
    retain90|retain95|retain99)
        tofu_split "unlearn${SPLIT#retain}"
        DATA_ARGS=(
            data/datasets@data.train=TOFU_QA_retain
            data.train.TOFU_QA_retain.args.hf_args.name="${SPLIT}"
        )
        ;;
    *)
        die "unknown split '${SPLIT}' (want retain100|retain90|retain95|retain99)"
        ;;
esac
task="$(basename "$(rundir_for "$SPLIT")")"
echo "[retain] model=${MODEL} split=${SPLIT} task=${task}" \
     "epochs=${EPOCHS:-$FINETUNE_EPOCHS} gpus=${GPUS} resume=${RESUME:-null}" >&2

tofu_train "$task" "${EPOCHS:-$FINETUNE_EPOCHS}" \
    forget_split="${FORGET}" \
    holdout_split="${HOLDOUT}" \
    "${DATA_ARGS[@]}" "$@"