#!/usr/bin/env bash
# Run SelectiveUnlearning for LLaDA-1.5 on all three TOFU forget splits.
#
# Usage:
#   bash scripts/dlm/tofu/unlearn_llada15_all_splits.sh \
#       [gpus] [hydra overrides...]
#
# Example:
#   bash scripts/dlm/tofu/unlearn_llada15_all_splits.sh \
#       0,1,2,3 trainer.method_args.beta=10.0 \
#       trainer.method_args.preserve_weight=0.1
set -euo pipefail

cd "$(dirname "$0")/../../.." || exit 1

gpus="${1:-}"
if (( $# > 0 )); then
    shift
fi

for split in forget10 forget05 forget01; do
    echo "==================== llada1.5 / ${split} ====================" >&2
    bash scripts/dlm/tofu/unlearn.sh \
        llada1.5 "$gpus" "$split" SelectiveUnlearning "$@"
done

echo "[done] completed all 3 LLaDA-1.5 forget splits" >&2
