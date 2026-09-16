#!/usr/bin/env bash
# Run SelectiveUnlearning for LLaDA-1.5 on TOFU forget01.
#
# Usage:
#   bash scripts/dlm/tofu/unlearn_llada15_forget01_selective.sh \
#       [gpus] [hydra overrides...]
#
# Example:
#   bash scripts/dlm/tofu/unlearn_llada15_forget01_selective.sh \
#       0,1,2,3 trainer.method_args.beta=10.0 \
#       trainer.method_args.preserve_weight=0.1
set -euo pipefail
umask 0000

cd "$(dirname "$0")/../../.." || exit 1

gpus="${1:-}"
if (( $# > 0 )); then
    shift
fi

exec bash scripts/dlm/tofu/unlearn.sh \
    llada1.5 "$gpus" forget01 SelectiveUnlearning "$@"
