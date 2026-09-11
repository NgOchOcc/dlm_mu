#!/usr/bin/env bash
# Run retain-free baseline methods with model-specific epoch schedules.
# Usage: bash scripts/dlm/tofu/unlearn_forget_only.sh <dream|llada> [gpus] [splits]
#
# Default epochs by split (forget10, forget05, forget01):
#   Dream: 2, 5, 7
#   LLaDA: 5, 7, 25
#
# Each value can be overridden independently, for example:
#   DREAM_FORGET05_EPOCHS=10 bash scripts/dlm/tofu/unlearn_forget_only.sh dream
#   LLADA_FORGET10_EPOCHS=10 bash scripts/dlm/tofu/unlearn_forget_only.sh llada
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export LOSS_WEIGHT_OVERRIDE=scheduler
source scripts/dlm/lib.sh

model_kind="${1:-}"
dlm_config "$model_kind" "${2:-}"

epochs_for_split() {
    local split=$1
    case "${model_kind}:${split}" in
        dream:forget10|dream:unlearn90) echo "${DREAM_FORGET10_EPOCHS:-2}" ;;
        dream:forget05|dream:unlearn95) echo "${DREAM_FORGET05_EPOCHS:-5}" ;;
        dream:forget01|dream:unlearn99) echo "${DREAM_FORGET01_EPOCHS:-7}" ;;
        llada:forget10|llada:unlearn90) echo "${LLADA_FORGET10_EPOCHS:-5}" ;;
        llada:forget05|llada:unlearn95) echo "${LLADA_FORGET05_EPOCHS:-7}" ;;
        llada:forget01|llada:unlearn99) echo "${LLADA_FORGET01_EPOCHS:-25}" ;;
        *) die "unknown model/split combination: ${model_kind}/${split}" ;;
    esac
}

for split in ${3:-forget10 forget05 forget01}; do
    epochs=$(epochs_for_split "$split")
    tofu_split "$split"

    for method in DiffusionGA DiffusionNPO DiffusionSimNPO DiffusionWGA; do
        task="${FORGET}/tofu_${MODEL}_${method}_forgetonly"
        echo "==================== ${task} (${epochs} epochs) ====================" >&2
        tofu_unlearn "$method" "$task" "$FORGET" "$RETAIN" "$HOLDOUT" "$epochs" \
            "~data.retain"

        tofu_eval "saves/unlearn/${task}" "$task" "$FORGET"
    done
done
