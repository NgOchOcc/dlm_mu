#!/usr/bin/env bash
# Run retain-free baseline methods over one or more TOFU forget splits.
# Usage: bash scripts/dlm/tofu/unlearn_forget_only.sh <dream|llada> [gpus] [splits]
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export LOSS_WEIGHT_OVERRIDE=scheduler
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"
for split in ${3:-forget10 forget05 forget01}; do
    tofu_split "$split"

    case "$split" in
        forget10) epochs=10 ;;
        forget05) epochs=5  ;;
        forget01) epochs=25 ;;
    esac

    for method in DiffusionGradAscent DiffusionNPO DiffusionSimNPO DiffusionWGA; do
        task="${FORGET}/tofu_${MODEL}_${method}_forgetonly"
        tofu_unlearn "$method" "$task" "$FORGET" "$RETAIN" "$HOLDOUT" "$epochs" \
            "~data.retain"

        tofu_eval "saves/unlearn/${task}" "$task" "$FORGET"
    done
done
