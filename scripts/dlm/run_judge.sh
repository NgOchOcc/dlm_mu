set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

# Post-hoc LLM-judge cho mọi model x forget split. Mỗi (model, split, method) là
# 1 process run_judge.py riêng, log riêng ở logs/judge/. Không dùng GPU: chỉ chấm
# lại generation đã có trong TOFU_EVAL.json. Mỗi generation chạy 3 rubric:
# quality (readability/helpfulness/truthfulness), leak (YES/NO nhị phân) và
# linguistic (fluency/relevance/hallucination/correctness).
#
# usage: bash scripts/dlm/run_judge.sh [dream llada ...]
# env:
#   SPLITS=forget01,forget05,forget10   split cần chấm
#   TARGETS=all|methods|refs            all = base + retain oracle + methods
#   METHODS=DiffusionNPO,DiffusionWGA   chỉ chấm các method này
#   JUDGE_SUITE=leak,linguistic         chỉ chạy các rubric này
#   OVERWRITE=1                         chấm lại cả rubric đã có điểm
#   JUDGE_WORKERS=8                     thread song song trong 1 target
#   JUDGE_PARALLEL=1                     số target chạy song song

MODELS=("$@")
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=(dream llada)
IFS=',' read -r -a SPLIT_LIST <<< "${SPLITS:-forget01,forget05,forget10}"
TARGETS="${TARGETS:-all}"

mkdir -p logs/judge
fail=0
for model in "${MODELS[@]}"; do
    for split in "${SPLIT_LIST[@]}"; do
        echo "=== judge ${model} ${split} (${TARGETS})"
        bash scripts/dlm/tofu/judge.sh "$model" "$split" "$TARGETS" || fail=1
    done
done

echo "run_judge done (fail=${fail})"
python scripts/dlm/show_results.py || true
