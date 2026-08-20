#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"
# Drop the args dlm_config consumed; anything left is a hydra override to pass
# through, e.g. trainer.args.learning_rate=1.7e-6
(( $# )) && shift $(( $# < 2 ? $# : 2 ))

FORGET=forget10
RETAIN=retain90
HOLDOUT=holdout10
EPOCHS=10
# Appended to every task name so a variant sweep (e.g. LOSS_WEIGHT_OVERRIDE)
# lands beside the existing results instead of overwriting them.
SUFFIX="${TASK_SUFFIX:-}"
# Subset of TOFU_TRAINERS to run, e.g. TRAINERS="DiffusionSimNPO DiffusionWGA"
# to fill in the arms a previous sweep did not cover.
read -r -a ARMS <<< "${TRAINERS:-${TOFU_TRAINERS[*]}}"

for trainer in "${ARMS[@]}"; do
    task="${FORGET}/tofu_${MODEL}_${trainer}${SUFFIX}"
    tofu_unlearn "$trainer" "$task" "$FORGET" "$RETAIN" "$HOLDOUT" "$EPOCHS" "$@"
    tofu_eval "saves/unlearn/${task}" "$task" "$FORGET"
done
