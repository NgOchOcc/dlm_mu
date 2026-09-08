#!/usr/bin/env bash
# MDU tau sweep (masked diffusion unlearning, arXiv:2605.18253).
#
# tau multiplies the target logits: 1.0 targets the frozen model's prompt-masked
# prediction as-is, 0.0 targets uniform. The paper sweeps {0 0.25 0.5 0.75 1.0}.
#
#   usage: bash scripts/dlm/tofu/mdu.sh <dream|llada> [gpus] [split]
#   env:   TAUS="0 0.5 1.0"; anything unlearn.sh reads also applies
#
# Epochs are not set here: an unset UNLEARN_EPOCHS falls through to the per-split
# SPLIT_EPOCHS in lib.sh (tofu_split), so MDU trains for exactly as long as every
# other method on the same split. Export UNLEARN_EPOCHS to override.
set -euo pipefail
for tau in ${TAUS:-0.5}; do
    TASK_SUFFIX="_tau${tau}" \
        bash "$(dirname "$0")/unlearn.sh" \
        "${1:-llada}" "${2:-}" "${3:-unlearn90}" MDU \
        trainer.method_args.tau="$tau" \
        trainer.method_args.eta=0.0 \
        trainer.method_args.kl_direction=forward \
        trainer.method_args.alpha=1.0 \
        trainer.method_args.gamma=1.0
done
