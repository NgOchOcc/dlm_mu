#!/usr/bin/env bash
# Train a TOFU reference model and evaluate it.
# Usage: bash scripts/dlm/tofu/retain.sh <dream|llada> [gpus] [retain split]
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
task="$(basename "$(rundir_for "$split")")"
echo "[retain] model=${MODEL} split=${split} task=${task}" \
     "epochs=${EPOCHS:-$FINETUNE_EPOCHS} gpus=${GPUS} resume=${RESUME:-null}" >&2

tofu_train "$task" "${EPOCHS:-$FINETUNE_EPOCHS}" \
    forget_split="${FORGET}" \
    holdout_split="${HOLDOUT}" \
    "${data_args[@]}" "$@"
