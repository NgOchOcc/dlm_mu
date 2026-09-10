#!/usr/bin/env bash
# Train forget-only WGA on Dream, save one checkpoint per epoch, then evaluate
# every saved checkpoint.
#
# Usage:
#   bash scripts/dlm/tofu/wga_dream_per_epoch.sh [gpus] [split] [epochs]
#
# Defaults:
#   gpus=0,1,2,3
#   split=forget05
#   epochs=10
#
# Set OVERWRITE=1 to re-evaluate epoch results that already exist.
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export LOSS_WEIGHT_OVERRIDE="${LOSS_WEIGHT_OVERRIDE:-scheduler}"
source scripts/dlm/lib.sh

gpus="${1:-0,1,2,3}"
split="${2:-forget05}"
epochs="${3:-10}"

[[ "$epochs" =~ ^[1-9][0-9]*$ ]] || die "epochs must be a positive integer"

dlm_config dream "$gpus"
tofu_split "$split"
method_setup DiffusionWGA

task="${FORGET}/tofu_${MODEL}_DiffusionWGA_forgetonly_per_epoch"
checkpoint_root="saves/unlearn/${task}"
train_log="logs/per_epoch/train_Dream_${FORGET}_DiffusionWGA.log"

mkdir -p logs/per_epoch

echo "[train] model=${MODEL} method=DiffusionWGA split=${FORGET}" \
     "epochs=${epochs} gpus=${gpus} -> ${train_log}" >&2

UNLEARN_SAVE_STRATEGY=epoch \
UNLEARN_SAVE_TOTAL_LIMIT=$((epochs + 1)) \
tofu_unlearn \
    DiffusionWGA \
    "$task" \
    "$FORGET" \
    "$RETAIN" \
    "$HOLDOUT" \
    "$epochs" \
    trainer.args.save_only_model=true \
    "${METHOD_ARGS[@]}" \
    '~data.retain' \
    2>&1 | tee "$train_log"

mapfile -t checkpoints < <(
    find "$checkpoint_root" -maxdepth 1 -type d -name 'checkpoint-*' \
        | awk -F'checkpoint-' '{print $2 "\t" $0}' \
        | sort -n \
        | cut -f2-
)

if (( ${#checkpoints[@]} == 0 )); then
    die "no epoch checkpoints found under ${checkpoint_root}"
fi

echo "[eval] found ${#checkpoints[@]} checkpoint(s); evaluating across ${gpus}" >&2
eval_plan_reset
for index in "${!checkpoints[@]}"; do
    epoch=$((index + 1))
    eval_task=$(printf \
        "%s/tofu_%s_DiffusionWGA_forgetonly_ep%02d" \
        "$FORGET" "$MODEL" "$epoch"
    )
    eval_plan_add "$eval_task" "${checkpoints[$index]}" "$FORGET"
done

PARALLEL=1 EVAL_GPUS="$gpus" eval_plan_run

echo "[done] Dream WGA ${FORGET}: trained ${epochs} epochs and evaluated all checkpoints"
python scripts/dlm/show_results.py --mode both || true
