#!/usr/bin/env bash
# Re-evaluate any per-epoch checkpoint whose TOFU_SUMMARY.json is missing or
# incomplete (fewer than $NEED metrics -- typically a CUDA OOM that only wrote
# the ROUGE metrics). Runs strictly ONE eval per GPU at a time via an 8-slot
# queue, so it never oversubscribes GPU memory the way the all-at-once parallel
# eval did.
#
#   usage: bash scripts/dlm/tofu/reeval_incomplete.sh [gpus] [methods] [split]
#   defaults: gpus=0,1,2,3,4,5,6,7  methods=GradAscent,WGA,NPO,SimNPO  split=forget10
#   env: NEED=9  minimum #metrics for a summary to count as complete
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

gpus="${1:-0,1,2,3,4,5,6,7}"
minimum_metrics="${NEED:-9}"
dlm_config llada "$gpus"
tofu_split "${3:-forget10}"
parse_methods "${2:-GradAscent,WGA,NPO,SimNPO}"

IFS=',' read -r -a gpu_list <<< "$gpus"
mkdir -p logs/eval

summary_ok() {  # <task> -> 0 if complete
    local s="saves/eval/${1}/TOFU_SUMMARY.json"
    [[ -f "$s" ]] || return 1
    local n
    n=$(python -c "import json,sys;print(len(json.load(open('$s'))))" 2>/dev/null || echo 0)
    (( n >= minimum_metrics ))
}

# Build the worklist: (checkpoint_dir, eval_task) for every incomplete epoch.
checkpoints=()
tasks=()
for method in "${METHODS_LIST[@]}"; do
    dir="saves/unlearn/${FORGET}/tofu_${MODEL}_${method}"
    [[ -d "$dir" ]] || { echo "skip ${method}: no train dir" >&2; continue; }
    ep=0
    while read -r ckpt; do
        [[ -n "$ckpt" ]] || continue
        ep=$(( ep + 1 ))
        etask=$(printf "%s/tofu_%s_%s_ep%02d" "$FORGET" "$MODEL" "$method" "$ep")
        if summary_ok "$etask"; then
            echo "ok    ${etask}"
        else
            echo "queue ${etask}  <- ${ckpt}"
            checkpoints+=("$(resolve_ckpt "$ckpt")")
            tasks+=("$etask")
        fi
    done < <(find "$dir" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | awk -F'checkpoint-' '{print $2"\t"$0}' | sort -n | cut -f2-)
done

n=${#tasks[@]}
echo "[reeval] ${n} incomplete checkpoint(s) to evaluate, 1 per GPU across: ${gpus}" >&2
(( n > 0 )) || { echo "[reeval] nothing to do"; exit 0; }

# 8-slot queue: keep at most one job per GPU running at a time.
declare -A gpu_pids=()
fail=0
launch() {  # <gpu> <idx>
    local gpu=$1 i=$2
    local log="logs/eval/${tasks[$i]//\//_}.log"
    echo "[gpu ${gpu}] ${tasks[$i]} -> ${log}" >&2
    tofu_eval_solo "$gpu" "${checkpoints[$i]}" "${tasks[$i]}" "$FORGET" "$log" &
    gpu_pids[$gpu]=$!
}

i=0
# prime: one job per GPU
for gpu in "${gpu_list[@]}"; do
    (( i < n )) || break
    launch "$gpu" "$i"; i=$((i+1)); sleep 3
done
# refill as GPUs free up
while (( i < n )); do
    for gpu in "${gpu_list[@]}"; do
        pid="${gpu_pids[$gpu]:-}"
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            [[ -n "$pid" ]] && { wait "$pid" 2>/dev/null || fail=1; }
            (( i < n )) || { gpu_pids[$gpu]=""; continue; }
            launch "$gpu" "$i"; i=$((i+1)); sleep 2
        fi
    done
    sleep 5
done
# drain
for gpu in "${gpu_list[@]}"; do
    pid="${gpu_pids[$gpu]:-}"
    [[ -n "$pid" ]] && { wait "$pid" 2>/dev/null || fail=1; }
done

echo "[reeval] done (fail=${fail})"
python scripts/dlm/show_results.py || true
exit "$fail"
