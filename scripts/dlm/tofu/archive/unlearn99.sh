#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

FORGET=forget01
RETAIN=retain99
HOLDOUT=holdout01
EPOCHS=25

for trainer in "${TOFU_TRAINERS[@]}"; do
    task="${FORGET}/tofu_${MODEL}_${trainer}"
    tofu_unlearn "$trainer" "$task" "$FORGET" "$RETAIN" "$HOLDOUT" "$EPOCHS"
    tofu_eval "saves/unlearn/${task}" "$task" "$FORGET"
done
