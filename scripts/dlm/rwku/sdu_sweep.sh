#!/usr/bin/env bash
# Grid-search SDU's two loss-balance hyperparameters -- SDU_BETA (forget vs.
# preserve weighting) and SDU_PRESERVE_WEIGHT (preserve-term strength) -- on a
# handful of RWKU entities, spreading the grid across all given GPUs in
# parallel like launch_all.sh does for methods.
#
# Each grid point is its own unlearn.sh run, tagged via RUN_TAG so checkpoints
# and eval results never collide:
#   eval/RWKU/{model}_DiffusionSDU_b{beta}_pw{pw}/train_v1/{idx}_{entity}/
#
# Only BETA/PRESERVE_WEIGHT vary here; SDU_K_PERCENT/SDU_ENTITY_BOOST (which
# affect the offline mask cache, not the loss) stay at their defaults, so the
# cache is built once and shared across every grid point.
#
#   scripts/dlm/rwku/sdu_sweep.sh dream 0,1,2,3
#   BETAS="2 5 10 20 50" scripts/dlm/rwku/sdu_sweep.sh dream 0,1,2,3
#   PRESERVE_WEIGHTS="0.02 0.05 0.1 0.2 0.5" scripts/dlm/rwku/sdu_sweep.sh dream 0,1,2,3
#   NUM_SUBJECTS=10 scripts/dlm/rwku/sdu_sweep.sh dream 0,1,2,3
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

MODEL_KEY="${1:-dream}"
DEVICES="${2:-0,1,2,3}"
NUM_SUBJECTS="${NUM_SUBJECTS:-5}"

# 3x3 around the current defaults (beta=10, preserve_weight=0.1).
BETAS="${BETAS:-5 10 20}"
PRESERVE_WEIGHTS="${PRESERVE_WEIGHTS:-0.05 0.1 0.2}"

GPU_POOL=()
for g in ${DEVICES//,/ }; do
    [[ -n "$g" ]] && GPU_POOL+=("$g")
done
(( ${#GPU_POOL[@]} > 0 )) || die "no GPUs given (got '$DEVICES')"

CONFIGS=()
for beta in $BETAS; do
    for pw in $PRESERVE_WEIGHTS; do
        CONFIGS+=("${beta}:${pw}")
    done
done

LOG_DIR="${LOG_DIR:-logs/rwku_sdu_sweep}"
mkdir -p "$LOG_DIR"

echo "=== RWKU SDU hyperparameter sweep ==="
echo "  model    : ${MODEL_KEY}"
echo "  gpus     : ${GPU_POOL[*]}"
echo "  entities : ${NUM_SUBJECTS}"
echo "  betas    : ${BETAS}"
echo "  pweights : ${PRESERVE_WEIGHTS}"
echo "  configs  : ${#CONFIGS[@]}"
echo "  logs     : ${LOG_DIR}/"
echo

# Same free-slot pattern as launch_all.sh: a GPU is reclaimed as soon as its
# current config finishes, so grid points that finish at different speeds
# never leave a device idle.
GPU_PID=()
ALL_PIDS=()
ALL_NAMES=()
claim_slot() {
    local i pid
    while :; do
        for i in "${!GPU_POOL[@]}"; do
            pid="${GPU_PID[$i]:-}"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                SLOT=$i
                return 0
            fi
        done
        wait -n 2>/dev/null || sleep 5
    done
}

for cfg in "${CONFIGS[@]}"; do
    beta="${cfg%%:*}"
    pw="${cfg##*:}"
    tag="b${beta}_pw${pw}"
    claim_slot
    gpu="${GPU_POOL[$SLOT]}"
    log="${LOG_DIR}/${MODEL_KEY}_${tag}_gpu${gpu}.log"
    echo "--- beta=${beta} preserve_weight=${pw} -> gpu ${gpu}  (${log})"
    GPUS="$gpu" METHODS=SDU NUM_SUBJECTS="$NUM_SUBJECTS" NO_SUMMARY=1 \
        SDU_BETA="$beta" SDU_PRESERVE_WEIGHT="$pw" RUN_TAG="$tag" \
        bash scripts/dlm/rwku/unlearn.sh "$MODEL_KEY" "$gpu" > "$log" 2>&1 &
    GPU_PID[$SLOT]=$!
    ALL_PIDS+=($!)
    ALL_NAMES+=("$tag")
done

echo
echo "running ${#CONFIGS[@]} configs across ${#GPU_POOL[@]} GPUs; tail -f ${LOG_DIR}/*.log to follow"
echo

fail=0
for i in "${!ALL_PIDS[@]}"; do
    if ! wait "${ALL_PIDS[$i]}"; then
        echo "FAILED: ${ALL_NAMES[$i]} (see ${LOG_DIR}/)"
        fail=1
    fi
done

echo
echo "=== summary (all configs, sorted by run name) ==="
python3 scripts/dlm/rwku/aggregate.py --setting "${SETTING:-train_v1}" || true

exit "$fail"
