#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8000/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.8-27B}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-8}"

dirs=(
    saves/eval/tofu_LLaDA-1.5_retain99
    saves/eval/tofu_LLaDA-1.5_retain95
)

pids=()
for dir in "${dirs[@]}"; do
    name="$(basename "$dir")"
    nohup .venv-vllm/bin/python -u \
        scripts/dlm/evaluate_all_llm_judge.py \
        --eval-root "$dir" \
        --poll-seconds 60 \
        --batch-size 32 \
        > "judge_${name}.log" 2>&1 &
    pids+=($!)
    echo "started ${name} (pid $!) -> judge_${name}.log"
done

wait "${pids[@]}"
