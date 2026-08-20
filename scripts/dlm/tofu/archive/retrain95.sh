#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

task="$(basename "$RETAIN95_CKPT")"
tofu_train "$task" "$EPOCHS" \
    data/datasets@data.train=TOFU_QA_retain \
    data.train.TOFU_QA_retain.args.hf_args.name=retain95 \
    forget_split=forget05 \
    holdout_split=holdout05
