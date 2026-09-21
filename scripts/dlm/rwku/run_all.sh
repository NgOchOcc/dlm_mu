#!/usr/bin/env bash
# Four RWKU unlearning methods in parallel, one per GPU. Each GPU runs its
# method against every model in turn -- dream first, then llada1.5 -- so the
# four methods advance together while no device ever holds two models at once.
#
#   gpu 0: SDU  dream -> SDU  llada1.5
#   gpu 1: WGA  dream -> WGA  llada1.5
#   gpu 2: NPO  dream -> NPO  llada1.5
#   gpu 3: BST  dream -> BST  llada1.5
#
# Per-run output goes to logs/rwku/ -- four processes sharing one terminal
# interleave their output into something unreadable.
#
#   scripts/dlm/rwku/run_all.sh
#   NUM_SUBJECTS=5 scripts/dlm/rwku/run_all.sh
#   MODELS="llada1.5" scripts/dlm/rwku/run_all.sh    # one model only
set -uo pipefail
# Run from the repository root. The job runner starts the script there and
# asks it not to change directory, so this only verifies the assumption instead
# of cd-ing: $0 is not a usable path when the script is sourced, piped in, or
# copied into a batch spool, and guessing from it silently landed elsewhere.
if [[ ! -f scripts/dlm/lib.sh ]]; then
    echo "error: run this from the repository root (no scripts/dlm/lib.sh in $PWD)." >&2
    exit 1
fi

MODELS="${MODELS:-dream llada1.5}"
NUM_SUBJECTS="${NUM_SUBJECTS:-3}"
LOG_DIR="${LOG_DIR:-logs/rwku}"
mkdir -p "$LOG_DIR"

# method:gpu
JOBS=(
    "SDU:0"
    "WGA:1"
    "NPO:2"
    "BST:3"
)

echo "=== RWKU: one method per GPU, models in sequence ==="
echo "  models   : ${MODELS}"
echo "  entities : ${NUM_SUBJECTS}"
echo "  logs     : ${LOG_DIR}/"
echo
for job in "${JOBS[@]}"; do
    printf '  gpu %s: %s\n' "${job##*:}" "${job%%:*}"
done
echo

# One background shell per GPU walking its own model list. Keeping the loop
# inside the background job (rather than launching every model up front) is what
# makes the models sequential per device: the next model only starts once the
# previous one has released the GPU.
PIDS=()
NAMES=()
for job in "${JOBS[@]}"; do
    method="${job%%:*}"
    gpu="${job##*:}"
    (
        for model in $MODELS; do
            log="${LOG_DIR}/${model}_${method}_gpu${gpu}.log"
            echo "[gpu ${gpu}] ${method} / ${model} -> ${log}"
            GPUS="$gpu" NUM_SUBJECTS="$NUM_SUBJECTS" METHODS="$method" NO_SUMMARY=1 \
                bash scripts/dlm/rwku/unlearn.sh "$model" "$gpu" > "$log" 2>&1 \
                || { echo "[gpu ${gpu}] FAILED: ${method} / ${model}"; exit 1; }
            echo "[gpu ${gpu}] done: ${method} / ${model}"
        done
    ) &
    PIDS+=($!)
    NAMES+=("$method")
done

echo "running ${#PIDS[@]} methods in parallel; tail -f ${LOG_DIR}/*.log to follow"
echo

fail=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "chain failed: ${NAMES[$i]} -- see ${LOG_DIR}/"; fail=1; }
done

echo
echo "=== summary ==="
python3 scripts/dlm/rwku/aggregate.py --setting "${SETTING:-train_v1}" || true

exit "$fail"
