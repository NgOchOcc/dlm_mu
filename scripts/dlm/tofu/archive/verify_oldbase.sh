#!/usr/bin/env bash
# Verify whether saves/eval_old's high Dream retain numbers were real model health
# or an artifact of the pre-truncation eval.
#
# The original eval_old unlearn weights are gone (the 2026-08-08 reruns wrote to the
# same saves/unlearn paths), but the pre-eos_pad_to SFT base survives in
# saves/finetune_old. So the arm is reproduced from that base under the old-era
# objective and scored with the CURRENT eval code:
#
#   B (no training) eval the old SFT base itself. Controls for the eval change
#     alone on a model known to score 0.9778 / gen_len 656 under the old code.
#   A (one unlearn)  NPO from the old base with cart and eos_pad_to disabled, i.e.
#     the eval_old recipe, then eval. If retain lands near the old 0.7119 the old
#     model really was healthier and eos_pad_to caused a regression; if it lands
#     near the current 0.118 the old number was metric inflation.
#
# Task names carry an _oldbase suffix so nothing existing is overwritten.
#
# Usage: PHASE=B scripts/dlm/tofu/verify_oldbase.sh [dream] [gpu-list]
#        PHASE defaults to all; B is eval-only and cheap, A needs a full unlearn.
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-dream}" "${2:-}"

PHASE="${PHASE:-all}"
OLD_BASE="saves/finetune_old/tofu_${MODEL}_retain100_150_epochs"
[[ -f "$OLD_BASE/config.json" ]] || { echo "error: no old base at $OLD_BASE" >&2; exit 1; }

FORGET=forget10
RETAIN=retain90
HOLDOUT=holdout10

# B: the eval-change control.
if [[ "$PHASE" == B || "$PHASE" == all ]]; then
    tofu_eval "$OLD_BASE" "tofu_${MODEL}_oldbase_retain100_${FORGET}" "$FORGET"
fi

# A: reproduce the eval_old arm, score it with the current eval.
if [[ "$PHASE" == A || "$PHASE" == all ]]; then
    task="${FORGET}/tofu_${MODEL}_DiffusionNPO_oldbase"
    tofu_unlearn DiffusionNPO "$task" "$FORGET" "$RETAIN" "$HOLDOUT" 10 \
        model.model_args.pretrained_model_name_or_path="$OLD_BASE" \
        trainer.method_args.eos_pad_to=null
    tofu_eval "saves/unlearn/${task}" "$task" "$FORGET"
fi
