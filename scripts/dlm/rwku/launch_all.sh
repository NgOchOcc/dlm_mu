#!/usr/bin/env bash
# Run one unlearning method per GPU, each over the full entity set.
#
# Methods are handed out to devices in order, so with the default five methods
# and four GPUs, device 0 takes SDU first and picks up NPO once it is done:
#
#   gpu 0: SDU  -> NPO      gpu 2: WGA
#   gpu 1: GA               gpu 3: SimNPO
#
# Each method is a separate unlearn.sh process writing to its own results
# directory, so they never collide. Per-method logs go to logs/rwku/.
#
#   scripts/dlm/rwku/launch_all.sh llada1.5
#   scripts/dlm/rwku/launch_all.sh dream 0,1,2,3
#   METHODS=SDU,GA scripts/dlm/rwku/launch_all.sh llada1.5 0,1
#   NUM_SUBJECTS=3 scripts/dlm/rwku/launch_all.sh llada1.5
set -euo pipefail
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
source scripts/dlm/lib.sh

MODEL_KEY="${1:-llada1.5}"
DEVICES="${2:-$GPUS}"

GPU_POOL=()
for g in ${DEVICES//,/ }; do
    [[ -n "$g" ]] && GPU_POOL+=("$g")
done
(( ${#GPU_POOL[@]} > 0 )) || die "no GPUs given (got '$DEVICES')"

IFS=',' read -ra METHOD_LIST <<< "${METHODS:-SDU,GA,WGA,SimNPO,NPO}"
CLEAN=()
for m in "${METHOD_LIST[@]}"; do
    m="${m// /}"
    [[ -n "$m" ]] && CLEAN+=("$m")
done
METHOD_LIST=("${CLEAN[@]}")
(( ${#METHOD_LIST[@]} > 0 )) || die "no methods given"

LOG_DIR="${LOG_DIR:-logs/rwku}"
mkdir -p "$LOG_DIR"

echo "=== RWKU: one method per GPU ==="
echo "  model   : ${MODEL_KEY}"
echo "  gpus    : ${GPU_POOL[*]}"
echo "  methods : ${METHOD_LIST[*]}"
echo "  logs    : ${LOG_DIR}/"
echo

# GPU_PID[i] holds the process currently on GPU_POOL[i]; a device is free once
# that process has exited. Waiting on the pool rather than assigning round-robin
# up front keeps every GPU busy when methods finish at different times.
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

for method in "${METHOD_LIST[@]}"; do
    claim_slot
    gpu="${GPU_POOL[$SLOT]}"
    log="${LOG_DIR}/${MODEL_KEY}_${method}_gpu${gpu}.log"
    echo "--- ${method} -> gpu ${gpu}  (${log})"
    GPUS="$gpu" METHODS="$method" NO_SUMMARY=1 \
        bash scripts/dlm/rwku/unlearn.sh "$MODEL_KEY" "$gpu" > "$log" 2>&1 &
    GPU_PID[$SLOT]=$!
    # GPU_PID only remembers the process currently on each device, and a slot is
    # reused once its method finishes -- so track every launch separately to
    # collect all the exit codes.
    ALL_PIDS+=($!)
    ALL_NAMES+=("$method")
done

echo
echo "waiting for all methods..."
fail=0
for i in "${!ALL_PIDS[@]}"; do
    if ! wait "${ALL_PIDS[$i]}"; then
        echo "FAILED: ${ALL_NAMES[$i]} (see ${LOG_DIR}/)"
        fail=1
    fi
done

echo
echo "=== summary ==="
python3 scripts/dlm/rwku/aggregate.py --setting "${SETTING:-train_v1}" || true
(( fail == 0 ))
