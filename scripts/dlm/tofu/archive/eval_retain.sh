#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"
OVERWRITE="${OVERWRITE:-0}"

eval_ref() {
    local ckpt=$1 forget=$2 task=$3
    if [[ "$OVERWRITE" != 1 && -f "saves/eval/${task}/TOFU_SUMMARY.json" ]]; then
        echo "skip ${task} (already evaluated)"
        return 0
    fi
    if [[ ! -e "$ckpt" ]]; then
        echo "miss ${task} (checkpoint not found: ${ckpt})" >&2
        return 0
    fi
    tofu_eval "$ckpt" "$task" "$forget"
}

for forget in forget10; do
    eval_ref "$SFT_CKPT" "$forget" "tofu_${MODEL}_retain100_${forget}"
done

# eval_ref "$RETAIN99_CKPT" forget01 "tofu_${MODEL}_retain99"
# eval_ref "$RETAIN95_CKPT" forget05 "tofu_${MODEL}_retain95"
eval_ref "$RETAIN90_CKPT" forget10 "tofu_${MODEL}_retain90"
