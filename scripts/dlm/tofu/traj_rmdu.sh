#!/usr/bin/env bash
# Trajectory-aligned residual MDU beta sweep (docs/traj_rmdu_training_algorithms.tex).
#
# beta is the residual strength; beta=0 reduces to MDU at tau=1, so the sweep
# carries its own ablation baseline.
#
#   usage: bash scripts/dlm/tofu/traj_rmdu.sh <dream|llada> [gpus] [split]
#   env:   BETAS="0 0.5 2"; anything unlearn.sh reads also applies
#
# Epochs are not set here: an unset UNLEARN_EPOCHS falls through to the per-split
# SPLIT_EPOCHS in lib.sh (tofu_split), so this trains for exactly as long as every
# other method on the same split. Export UNLEARN_EPOCHS to override.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SPLIT="${3:-unlearn90}"

for beta in ${BETAS:-0 0.25 0.5 1 2 4}; do
    TASK_SUFFIX="_beta${beta}" \
        bash "$HERE/unlearn.sh" "${1:-}" "${2:-}" "$SPLIT" TrajRMDU \
        trainer.method_args.beta="$beta" \
        trainer.method_args.mask_source=trajectory \
        trainer.method_args.target_mode=residual \
        trainer.method_args.trajectory_steps=0 \
        trainer.method_args.alpha=1.0 \
        trainer.method_args.gamma=1.0
done
