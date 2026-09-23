#!/usr/bin/env bash
# Grid-search the selective-unlearning loss balance -- SDU_BETA (entity
# suppression strength) and SDU_PRESERVE_WEIGHT (preserve-term strength) -- on
# MUSE, for each model in turn.
#
# Models run one after another (dream first, then llada1.5), and within a model
# the grid points run sequentially, each using every given GPU. Every grid point
# is its own run.sh invocation tagged via RUN_TAG, so checkpoints and eval
# results never collide:
#   saves/eval/muse_{MODEL}_{split}_{Method}_b{beta}_pw{pw}/MUSE_SUMMARY.json
#
# Defaults to SelectiveUnlearning, which picks its query-entity positions with
# the built-in heuristic. Set METHOD=SDU for the variant that reads them from
# the offline Monte Carlo cache instead; both take the same two knobs.
#
# Usage: selective_sweep.sh [models] [gpus] [splits]
#
#   scripts/dlm/muse/selective_sweep.sh
#   scripts/dlm/muse/selective_sweep.sh dream 0,1,2,3
#   scripts/dlm/muse/selective_sweep.sh "dream llada1.5" 0,1,2,3 News
#   BETAS="2 5 10 20 50" scripts/dlm/muse/selective_sweep.sh dream 0,1,2,3
#   PAIRS="1:0.1 2:0.1 10:0.2" scripts/dlm/muse/selective_sweep.sh dream 0,1,2,3
#   METHOD=SDU scripts/dlm/muse/selective_sweep.sh dream 0,1,2,3
set -euo pipefail
# Run from the repository root; see scripts/dlm/muse/run.sh for why this checks
# instead of cd-ing.
if [[ ! -f scripts/dlm/lib.sh ]]; then
    echo "error: run this from the repository root (no scripts/dlm/lib.sh in $PWD)." >&2
    exit 1
fi
source scripts/dlm/lib.sh

# Argument order matches scripts/dlm/rwku/selective_sweep.sh: models first,
# then GPUs. Pass "dream llada1.5" (quoted) to sweep both in turn.
MODELS="${1:-${MODELS:-dream llada1.5}}"
DEVICES="${2:-$GPUS}"
DATA_SPLITS="${3:-${DATA_SPLITS:-News Books}}"

METHOD="${METHOD:-SelectiveUnlearning}"
[[ "$METHOD" == Diffusion* ]] || METHOD="Diffusion${METHOD}"
[[ -f "configs/trainer/${METHOD}.yaml" ]] || die "no such trainer: configs/trainer/${METHOD}.yaml"

BETAS="${BETAS:-2 5 10 20 50}"
PRESERVE_WEIGHTS="${PRESERVE_WEIGHTS:-0.1}"

[[ -n "$DEVICES" ]] || die "no GPUs: pass them as the second argument, e.g. \
    scripts/dlm/muse/selective_sweep.sh dream 0,1,2,3"

# PAIRS names the exact (beta, preserve_weight) points to run, as a
# space-separated list of "beta:pw". Use it when the wanted points are not a
# full BETAS x PRESERVE_WEIGHTS grid -- e.g. a beta sweep at one preserve
# weight plus a preserve sweep at one beta, which as a grid would be mostly
# configurations nobody asked for.
CONFIGS=()
if [[ -n "${PAIRS:-}" ]]; then
    for pair in $PAIRS; do
        [[ "$pair" == *:* ]] || die "PAIRS entries must look like beta:pw, got '$pair'"
        CONFIGS+=("$pair")
    done
else
    for beta in $BETAS; do
        for pw in $PRESERVE_WEIGHTS; do
            CONFIGS+=("${beta}:${pw}")
        done
    done
fi

LOG_DIR="${LOG_DIR:-logs/muse_${METHOD}_sweep}"
mkdir -p "$LOG_DIR"

echo "=== MUSE selective-unlearning hyperparameter sweep ==="
echo "  method   : ${METHOD}"
echo "  models   : ${MODELS}"
echo "  splits   : ${DATA_SPLITS}"
echo "  gpus     : ${DEVICES}"
if [[ -n "${PAIRS:-}" ]]; then
    echo "  pairs    : ${PAIRS}"
else
    echo "  betas    : ${BETAS}"
    echo "  pweights : ${PRESERVE_WEIGHTS}"
fi
echo "  runs     : $(( ${#CONFIGS[@]} * $(wc -w <<< "$MODELS") ))"
echo "  logs     : ${LOG_DIR}/"
echo

fail=0
for model in $MODELS; do
    echo "######## ${model} ########"
    index=0
    for cfg in "${CONFIGS[@]}"; do
        beta="${cfg%%:*}"
        pw="${cfg##*:}"
        tag="b${beta}_pw${pw}"
        index=$((index + 1))
        log="${LOG_DIR}/${model}_${tag}.log"
        echo "--- [${index}/${#CONFIGS[@]}] beta=${beta} preserve_weight=${pw}  (${log})"
        if METHODS="$METHOD" SDU_BETA="$beta" SDU_PRESERVE_WEIGHT="$pw" RUN_TAG="$tag" \
            bash scripts/dlm/muse/run.sh "$model" "$DEVICES" "$DATA_SPLITS" > "$log" 2>&1
        then
            echo "    done"
        else
            echo "    FAILED (see ${log})"
            fail=1
        fi
    done
    echo
done

exit "$fail"
