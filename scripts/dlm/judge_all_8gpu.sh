#!/usr/bin/env bash
# One-shot: serve 8 local vLLM judge servers (one per GPU 0-7, ports 8000-8007,
# TP=1, via the vllm-judge env), wait until all are ready, then judge every saved
# TOFU eval dir that is NOT yet scored -- sharded one server per GPU with
# JUDGE_WORKERS=1 for reproducible scores.
#
# Servers are setsid-detached so they survive this shell exiting. The judge step
# is also detached and logged, so the whole thing keeps running after you walk
# away. Progress:  tail -f logs/judge/sharded_run.log
#
#   usage: bash scripts/dlm/judge_all_8gpu.sh
#   env:
#     OVERWRITE (0)   set 1 to re-judge dirs that already have scores
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

VLLM_BIN="${VLLM_BIN:-/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/miniconda3/envs/vllm-judge/bin/vllm}"
PORTS=(8000 8001 8002 8003 8004 8005 8006 8007)
OVERWRITE="${OVERWRITE:-0}"
mkdir -p logs/judge

# Refuse to stomp on servers that are already up (avoids port/GPU collisions with
# a previous, still-alive launch).
if pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "error: 'vllm serve' already running; stop it first:" >&2
    echo "  bash scripts/dlm/serve_judge.sh stop" >&2
    exit 1
fi

echo "== step 1/3: launching 8 judge servers (fresh logs) =="
for i in 0 1 2 3 4 5 6 7; do
    port=$((8000 + i))
    setsid env GPUS="$i" PORT="$port" TP=1 VLLM_BIN="$VLLM_BIN" \
        bash scripts/dlm/serve_judge.sh start \
        > "logs/judge/vllm_serve_${port}.log" 2>&1 < /dev/null &
    echo "  launched GPU=$i PORT=$port -> logs/judge/vllm_serve_${port}.log"
    sleep 3
done

echo "== step 2/3: waiting for all 8 servers to become ready =="
for p in "${PORTS[@]}"; do
    if ! PORT="$p" bash scripts/dlm/serve_judge.sh wait; then
        echo "error: server on port $p never became ready; see logs/judge/vllm_serve_${p}.log" >&2
        exit 1
    fi
done
echo "all 8 servers ready."

echo "== step 3/3: judging not-yet-scored dirs (JUDGE_WORKERS=1, OVERWRITE=${OVERWRITE}) =="
setsid env JUDGE_WORKERS=1 OVERWRITE="${OVERWRITE}" \
    bash scripts/dlm/run_judge_sharded.sh "${PORTS[@]}" \
    > logs/judge/sharded_run.log 2>&1 < /dev/null &
echo "judge launched (detached). follow with:  tail -f logs/judge/sharded_run.log"
