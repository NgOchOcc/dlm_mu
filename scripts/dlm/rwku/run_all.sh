#!/usr/bin/env bash
# Four RWKU unlearning methods in parallel, one per GPU, each over the full
# entity set. Every method is an independent process writing to its own results
# directory, so they never collide.
#
# Per-method output goes to logs/rwku/ -- four processes sharing one terminal
# interleave their output into something unreadable.
#
#   scripts/dlm/rwku/run_all.sh
#   NUM_SUBJECTS=10 scripts/dlm/rwku/run_all.sh
#   MODEL=dream scripts/dlm/rwku/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

MODEL="${MODEL:-llada1.5}"
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

echo "=== RWKU: one method per GPU ==="
echo "  model    : ${MODEL}"
echo "  entities : ${NUM_SUBJECTS}"
echo "  logs     : ${LOG_DIR}/"
echo

PIDS=()
NAMES=()
for job in "${JOBS[@]}"; do
    method="${job%%:*}"
    gpu="${job##*:}"
    log="${LOG_DIR}/${MODEL}_${method}_gpu${gpu}.log"
    echo "--- ${method} -> gpu ${gpu}  (${log})"
    GPUS="$gpu" NUM_SUBJECTS="$NUM_SUBJECTS" METHODS="$method" NO_SUMMARY=1 \
        bash scripts/dlm/rwku/unlearn.sh "$MODEL" "$gpu" > "$log" 2>&1 &
    PIDS+=($!)
    NAMES+=("$method")
done

echo
echo "running ${#PIDS[@]} methods in parallel; tail -f ${LOG_DIR}/*.log to follow"
echo

fail=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "done: ${NAMES[$i]}"
    else
        echo "FAILED: ${NAMES[$i]} -- see ${LOG_DIR}/${MODEL}_${NAMES[$i]}_gpu*.log"
        fail=1
    fi
done

echo
echo "=== summary ==="
python3 scripts/dlm/rwku/aggregate.py --setting "${SETTING:-train_v1}" || true

exit "$fail"
