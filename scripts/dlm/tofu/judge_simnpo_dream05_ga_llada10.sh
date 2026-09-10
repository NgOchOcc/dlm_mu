#!/usr/bin/env bash
# Run the forget-set linguistic LLM judge for exactly two TOFU evaluations:
#   1. Dream SimNPO forget05 (forget-only)
#   2. LLaDA GA forget10 (forget-only)
#
# The judge server must already be available through JUDGE_BASE_URL.
# Existing complete judge results are skipped unless OVERWRITE=1 is set.
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8000/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.8-27B}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

targets=(
    "saves/eval/forget05/tofu_Dream-v0-Instruct-7B_DiffusionSimNPO_forgetonly"
    "saves/eval/forget10/tofu_LLaDA-8B-Instruct_DiffusionGradAscent_forgetonly"
)

for target in "${targets[@]}"; do
    if [[ ! -f "${target}/TOFU_EVAL.json" ]]; then
        echo "missing ${target}/TOFU_EVAL.json; run checkpoint evaluation first" >&2
        exit 1
    fi
done

args=("${targets[@]}")
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    args+=(--overwrite)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    args+=(--dry-run)
fi

mkdir -p logs/judge
log="logs/judge/simnpo_dream05_ga_llada10.log"

echo "[judge] model=${JUDGE_MODEL} url=${JUDGE_BASE_URL} workers=${JUDGE_WORKERS}"
echo "[judge] log=${log} overwrite=${OVERWRITE:-0} dry_run=${DRY_RUN:-0}"

"$PYTHON_BIN" -u scripts/dlm/run_linguistic_full.py "${args[@]}" 2>&1 | tee "$log"

echo "[judge] done"
