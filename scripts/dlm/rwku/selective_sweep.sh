#!/usr/bin/env bash
# Grid-search the selective-unlearning loss balance -- SDU_BETA (entity
# suppression strength) and SDU_PRESERVE_WEIGHT (preserve-term strength) -- on a
# handful of RWKU entities.
#
# Grid points run one after another, each using every given GPU: training shards
# the model across them with ZeRO-3, and evaluation runs on the first one (the
# evaluators do not shard across ranks, so extra eval processes would only
# repeat the same work).
#
# Defaults to SelectiveUnlearning, which picks its query-entity positions with
# the built-in heuristic. Set METHOD=SDU for the variant that reads them from
# the offline Monte Carlo cache instead; both take the same two knobs.
#
# Each grid point is its own unlearn.sh run, tagged via RUN_TAG so checkpoints
# and eval results never collide:
#   eval/RWKU/{model}_{Method}_b{beta}_pw{pw}/train_v1/{idx}_{entity}/
#
# Only BETA/PRESERVE_WEIGHT vary here; SDU_K_PERCENT/SDU_ENTITY_BOOST (which
# affect the offline mask cache, not the loss) stay at their defaults, so the
# cache is built once and shared across every grid point.
#
#   scripts/dlm/rwku/selective_sweep.sh dream 0,1,2,3
#   METHOD=SDU scripts/dlm/rwku/selective_sweep.sh dream 0,1,2,3
#   BETAS="2 5 10 20 50" scripts/dlm/rwku/selective_sweep.sh dream 0,1,2,3
#   NUM_SUBJECTS=10 scripts/dlm/rwku/selective_sweep.sh dream 0,1,2,3
set -euo pipefail
# Run from the repository root. The job runner starts the script there and
# asks it not to change directory, so this only verifies the assumption instead
# of cd-ing: $0 is not a usable path when the script is sourced, piped in, or
# copied into a batch spool, and guessing from it silently landed elsewhere.
if [[ ! -f scripts/dlm/lib.sh ]]; then
    echo "error: run this from the repository root (no scripts/dlm/lib.sh in $PWD)." >&2
    exit 1
fi
source scripts/dlm/lib.sh

MODEL_KEY="${1:-dream}"
DEVICES="${2:-0,1,2,3}"
NUM_SUBJECTS="${NUM_SUBJECTS:-5}"
METHOD="${METHOD:-SelectiveUnlearning}"
[[ "$METHOD" == Diffusion* ]] || METHOD="Diffusion${METHOD}"
[[ -f "configs/trainer/${METHOD}.yaml" ]] || die "no such trainer: configs/trainer/${METHOD}.yaml"

# 3x3 around the current defaults (beta=10, preserve_weight=0.1).
BETAS="${BETAS:-5 10 20}"
PRESERVE_WEIGHTS="${PRESERVE_WEIGHTS:-0.05 0.1 0.2}"

GPUS="$DEVICES"
[[ -n "$GPUS" ]] || die "no GPUs given (got '$DEVICES')"

CONFIGS=()
for beta in $BETAS; do
    for pw in $PRESERVE_WEIGHTS; do
        CONFIGS+=("${beta}:${pw}")
    done
done

LOG_DIR="${LOG_DIR:-logs/rwku_${METHOD}_sweep}"
mkdir -p "$LOG_DIR"

echo "=== RWKU selective-unlearning hyperparameter sweep ==="
echo "  method   : ${METHOD}"
echo "  model    : ${MODEL_KEY}"
echo "  gpus     : ${GPUS}  (train shards across all; eval uses ${GPUS%%,*})"
echo "  entities : ${NUM_SUBJECTS}"
echo "  betas    : ${BETAS}"
echo "  pweights : ${PRESERVE_WEIGHTS}"
echo "  configs  : ${#CONFIGS[@]}"
echo "  logs     : ${LOG_DIR}/"
echo

fail=0
index=0
for cfg in "${CONFIGS[@]}"; do
    beta="${cfg%%:*}"
    pw="${cfg##*:}"
    tag="b${beta}_pw${pw}"
    index=$((index + 1))
    log="${LOG_DIR}/${MODEL_KEY}_${tag}.log"
    echo "--- [${index}/${#CONFIGS[@]}] beta=${beta} preserve_weight=${pw}  (${log})"
    if GPUS="$GPUS" METHODS="$METHOD" NUM_SUBJECTS="$NUM_SUBJECTS" NO_SUMMARY=1 \
        SDU_BETA="$beta" SDU_PRESERVE_WEIGHT="$pw" RUN_TAG="$tag" \
        bash scripts/dlm/rwku/unlearn.sh "$MODEL_KEY" "$GPUS" > "$log" 2>&1
    then
        echo "    done"
    else
        echo "    FAILED (see ${log})"
        fail=1
    fi
done

echo
echo "=== summary (all configs, sorted by run name) ==="
python3 scripts/dlm/rwku/aggregate.py --setting "${SETTING:-train_v1}" || true

exit "$fail"
