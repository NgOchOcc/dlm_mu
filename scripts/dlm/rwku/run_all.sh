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
RWKU_REPO_FALLBACK="/prj/corp/airesearch/lasvegas/vol20-scratch/nluu/working/open-unlearning"

# Locate the repository root without relying on $0: it is not a usable path
# when the script is sourced, piped in, or copied into a batch spool directory
# (sbatch), in which case the old `cd $(dirname $0)/../../..` silently landed
# somewhere else and `source scripts/dlm/lib.sh` failed before any log existed.
_rwku_root() {
    local src="${BASH_SOURCE[0]:-}"
    if [[ -n "$src" && "$src" != "bash" && -e "$src" ]]; then
        local dir
        dir=$(cd "$(dirname "$src")/../../.." && pwd) || return 1
        [[ -f "$dir/scripts/dlm/lib.sh" ]] && { echo "$dir"; return 0; }
    fi
    # Fall back to REPO_DIR, the known cluster checkout, then the working
    # directory the job started in. Each candidate is verified by looking for
    # scripts/dlm/lib.sh, so a stale hardcoded path is skipped rather than
    # silently used.
    local fallback
    for fallback in "${REPO_DIR:-}" "$RWKU_REPO_FALLBACK" "$PWD"; do
        [[ -n "$fallback" && -f "$fallback/scripts/dlm/lib.sh" ]] && {
            echo "$fallback"; return 0
        }
    done
    return 1
}
ROOT_DIR=$(_rwku_root) || {
    echo "error: cannot locate the repository root (no scripts/dlm/lib.sh found)." >&2
    echo "       Run this script by its path from a checkout, or set REPO_DIR." >&2
    exit 1
}
cd "$ROOT_DIR" || exit 1

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
