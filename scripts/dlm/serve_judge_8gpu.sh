#!/usr/bin/env bash
# Launch 8 local vLLM judge servers, one per GPU (0-7), ports 8000-8007, TP=1,
# using the dedicated vllm-judge conda env. Each server is setsid-detached so it
# survives the launching shell (and the Claude process) exiting. Determinism is
# handled downstream by JUDGE_WORKERS=1 in run_judge_sharded.sh, not here.
#
#   usage: bash scripts/dlm/serve_judge_8gpu.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

VLLM_BIN="${VLLM_BIN:-/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/miniconda3/envs/vllm-judge/bin/vllm}"
mkdir -p logs/judge

for i in 0 1 2 3 4 5 6 7; do
    port=$((8000 + i))
    setsid env GPUS="$i" PORT="$port" TP=1 VLLM_BIN="$VLLM_BIN" \
        bash scripts/dlm/serve_judge.sh start \
        > "logs/judge/vllm_serve_${port}.log" 2>&1 < /dev/null &
    echo "launched GPU=$i PORT=$port -> logs/judge/vllm_serve_${port}.log"
    sleep 2
done
echo "all 8 launch commands issued"
