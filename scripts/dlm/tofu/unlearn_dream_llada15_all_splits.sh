#!/usr/bin/env bash
# Run SelectiveUnlearning for Dream and LLaDA-1.5 on all three TOFU forget splits.
#
# Usage:
#   bash scripts/dlm/tofu/unlearn_dream_llada15_all_splits.sh \
#       [gpus] [hydra overrides...]
#
# Examples:
#   bash scripts/dlm/tofu/unlearn_dream_llada15_all_splits.sh 0,1,2,3
#   NO_EVAL=1 bash scripts/dlm/tofu/unlearn_dream_llada15_all_splits.sh \
#       0,1,2,3 trainer.method_args.beta=10.0 \
#       trainer.method_args.preserve_weight=0.1
set -euo pipefail

cd "$(dirname "$0")/../../.." || exit 1

gpus="${1:-}"
if (( $# > 0 )); then
    shift
fi

models=(dream llada1.5)
splits=(forget10 forget05 forget01)

for model in "${models[@]}"; do
    for split in "${splits[@]}"; do
        echo "==================== ${model} / ${split} ====================" >&2
        bash scripts/dlm/tofu/unlearn.sh \
            "$model" "$gpus" "$split" SelectiveUnlearning "$@"
    done
done

echo "[done] completed all 6 model/split settings" >&2
