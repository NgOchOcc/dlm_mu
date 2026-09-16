#!/usr/bin/env bash
# Evaluate classical TOFU metrics for the LLaDA-1.5 retain90 SFT checkpoint.
#
# Checkpoint:
#   saves/finetune/tofu_LLaDA-1.5_retain90_150epochs
#
# Usage:
#   bash scripts/dlm/tofu/evaluate_llada15_retain90.sh [gpus] [hydra overrides...]
#
# Examples:
#   bash scripts/dlm/tofu/evaluate_llada15_retain90.sh 0,1,2,3
#   OVERWRITE=1 bash scripts/dlm/tofu/evaluate_llada15_retain90.sh 0,1,2,3 \
#       eval.tofu_dlm.batch_size=4
set -euo pipefail

cd "$(dirname "$0")/../../.." || exit 1

gpus="${1:-}"
if (( $# > 0 )); then
    shift
fi

exec bash scripts/dlm/tofu/evaluate_retain.sh llada1.5 "$gpus" retain90 "$@"
