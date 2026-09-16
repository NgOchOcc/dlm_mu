#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

dlm_config "${1:-}" "${2:-}"
split="${3:-retain100}"
if (( $# > 3 )); then shift 3; else shift $#; fi

data_args=()
case "$split" in
    retain100)
        tofu_split unlearn90
        ;;
    retain90|retain95|retain99)
        tofu_split "unlearn${split#retain}"
        data_args=(
            data/datasets@data.train=TOFU_QA_retain
            data.train.TOFU_QA_retain.args.hf_args.name="$split"
        )
        ;;
    *)
        die "unknown split '$split' (want retain100|retain90|retain95|retain99)"
        ;;
esac
rundir="$(rundir_for "$split")"
task="$(basename "$rundir")"

ckpt="$(resolve_ckpt "$rundir")"
if [[ "$ckpt" == "$rundir"/checkpoint-* ]]; then
    RESUME="${RESUME:-$ckpt}"
elif [[ -f "$rundir/config.json" ]] && ! flag_on "${OVERWRITE:-}"; then
    die "training already complete in ${rundir} (set OVERWRITE=1 to retrain from scratch)"
fi

tofu_train "$task" "${EPOCHS:-$FINETUNE_EPOCHS}" \
    forget_split="${FORGET}" \
    holdout_split="${HOLDOUT}" \
    "${data_args[@]}" "$@"
