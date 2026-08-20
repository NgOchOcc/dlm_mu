#!/usr/bin/env bash
# TSD -- trajectory-selective distillation (retain-free) on TOFU.
#
# TSD edits the frozen teacher at answer masks (demote the gold token and
# renormalise) and self-distills elsewhere, over answer-stratified partial masks.
# On TOFU the answer is the whole response, so it reduces to pathwise selective
# demotion; alpha=0 keeps it retain-free (no retain dataset needed).
#
#   usage: bash scripts/dlm/tofu/tsd.sh <dream|llada> [gpus] [split]
#   env:
#     DEMOTIONS="minimal belief"  demotion ablation sweep (default: minimal).
#                                 "minimal" demotes only the gold token; "belief"
#                                 also demotes the teacher top-K (paraphrase leak).
#     MASK_STATES=pathwise        "pathwise" (partial masks) | "single" (fully
#                                 masked answer only -- single-state ablation).
#     BELIEF_TOP_K=5              teacher top-K demotion set (demotion=belief only).
#     anything unlearn.sh reads also applies (EPOCHS, SKIP_DONE, NO_EVAL, ...).
#
# Epochs are not set here: an unset UNLEARN_EPOCHS falls through to the per-split
# SPLIT_EPOCHS in lib.sh (tofu_split), so TSD trains for exactly as long as every
# other method on the same split. Export UNLEARN_EPOCHS to override.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SPLIT="${3:-unlearn90}"
MASK_STATES="pathwise"
BELIEF_TOP_K=5
read -r -a DEMOTION_ARR <<< "${DEMOTIONS:-minimal}"

for demotion in "${DEMOTION_ARR[@]}"; do
    suffix=""
    (( ${#DEMOTION_ARR[@]} > 1 )) && suffix="_${demotion}"
    TASK_SUFFIX="${suffix}" \
        bash "$HERE/unlearn.sh" "${1:-}" "${2:-}" "$SPLIT" TSD \
        trainer.method_args.mask_states="$MASK_STATES" \
        trainer.method_args.demotion="$demotion" \
        trainer.method_args.belief_top_k="$BELIEF_TOP_K" \
        trainer.method_args.self_distill=true \
        trainer.method_args.balanced_weight=true \
        trainer.method_args.alpha=0.0 \
        trainer.method_args.gamma=1.0
done
