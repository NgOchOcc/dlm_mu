#!/usr/bin/env bash
# Re-evaluate two specific historical TOFU forget-only checkpoints.
#
# Usage:
#   bash scripts/dlm/tofu/reevaluate_two_ckpts.sh [gpus] [hydra overrides...]
#
# Optional checkpoint overrides:
#   DREAM_CKPT=/path/to/checkpoint
#   LLADA_CKPT=/path/to/checkpoint
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

gpus="${1:-0,1,2,3}"
if (( $# > 0 )); then shift; fi

dream_task="forget01/tofu_Dream-v0-Instruct-7B_DiffusionSimNPO_forgetonly"
dream_ckpt="${DREAM_CKPT:-saves/unlearn/${dream_task}}"
llada_task="forget10/tofu_LLaDA-8B-Instruct_DiffusionGradAscent_forgetonly"
llada_ckpt="${LLADA_CKPT:-saves/unlearn/${llada_task}}"

fail=0

reevaluate() {
    local model=$1 split=$2 task=$3 checkpoint=$4
    shift 4

    if [[ ! -e "$checkpoint" ]]; then
        echo "missing checkpoint: ${checkpoint}" >&2
        return 1
    fi

    dlm_config "$model" "$gpus"
    tofu_split "$split"
    checkpoint=$(resolve_ckpt "$checkpoint")
    echo "[reevaluate] model=${MODEL} split=${FORGET} task=${task} ckpt=${checkpoint}" >&2

    if flag_on "${DRY_RUN:-}"; then
        return 0
    fi
    tofu_eval "$checkpoint" "$task" "$FORGET" "$@"
}

reevaluate dream forget01 "$dream_task" "$dream_ckpt" "$@" || fail=1
reevaluate llada forget10 "$llada_task" "$llada_ckpt" "$@" || fail=1

echo "[reevaluate] done (fail=${fail})"
if ! flag_on "${DRY_RUN:-}"; then
    python scripts/dlm/show_results.py || true
fi
exit "$fail"
