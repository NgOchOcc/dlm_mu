#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8000/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.8-27B}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-8}"

mapfile -t dirs < <(find saves/eval -type d -iname '*DiffusionSDU*' | sort)
if (( ${#dirs[@]} == 0 )); then
    echo "no SDU eval folders found under saves/eval" >&2
    exit 1
fi

pids=()
for dir in "${dirs[@]}"; do
    name="$(echo "$dir" | tr '/' '_')"
    nohup .venv-vllm/bin/python -u \
        scripts/dlm/evaluate_all_llm_judge.py \
        --eval-root "$dir" \
        --poll-seconds 60 \
        --batch-size 32 \
        --once \
        --overwrite \
        > "judge_${name}.log" 2>&1 &
    pids+=($!)
    echo "started ${dir} (pid $!) -> judge_${name}.log"
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "judge sdu-all done (fail=${fail})"
