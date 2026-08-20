#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

FORGET=forget05
RETAIN=retain95
HOLDOUT=holdout05
EPOCHS=10

for trainer in "${TOFU_TRAINERS[@]}"; do
    task="${FORGET}/tofu_${MODEL}_${trainer}"
    tofu_unlearn "$trainer" "$task" "$FORGET" "$RETAIN" "$HOLDOUT" "$EPOCHS"
    tofu_eval "saves/unlearn/${task}" "$task" "$FORGET"
done
