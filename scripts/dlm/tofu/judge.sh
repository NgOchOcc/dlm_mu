#!/usr/bin/env bash
# Post-hoc LLM-judge scoring of saved TOFU eval dirs, one run_judge.py invocation
# per (model, forget split, method) so each target gets its own log and can be
# re-run on its own. No GPU is used: this re-judges generations already stored in
# TOFU_EVAL.json. Three rubrics run per generation (see scripts/dlm/run_judge.py):
# quality (readability/helpfulness/truthfulness), leak (binary YES/NO) and
# linguistic (fluency/relevance/hallucination/correctness). Rubrics already
# present in a dir's summary are skipped unless OVERWRITE=1.
#
# usage: bash scripts/dlm/tofu/judge.sh <dream|llada> [split] [targets]
#   split    unlearn90 (forget10, default) | unlearn95 | unlearn99; bare forget
#            splits (forget10, ...) are accepted too
#   targets  comma-separated, default all:
#              base     the retain100 "Base" model
#              retain   the retain oracle paired with the split
#              refs     base,retain
#              methods  the unlearning methods
#              all      refs,methods
#
# env:
#   METHODS=NPO,WGA                     judge only these methods (short names ok)
#   TASK_SUFFIX=_tau0.5                 match the suffix a sweep was trained under
#   JUDGE_SUITE=leak,linguistic         judge only these rubrics
#   OVERWRITE=1                         re-judge rubrics that already have scores
#   JUDGE_WORKERS=8                     threads per target (API concurrency)
#   JUDGE_PARALLEL=1                    targets judged concurrently
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}"

FORGET_SPEC="${2:-forget10}"
TARGETS="${3:-all}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
JUDGE_PARALLEL="${JUDGE_PARALLEL:-1}"

tofu_split "$FORGET_SPEC"          # -> FORGET, RETAIN
parse_methods "${METHODS:-}"       # -> METHODS_LIST
SUFFIX="${TASK_SUFFIX:-}"
[[ "$TARGETS" == all ]] && TARGETS=refs,methods

dirs=()
for target in ${TARGETS//,/ }; do
    case "$target" in
        refs|base|retain)
            [[ "$target" != retain ]] && dirs+=("saves/eval/tofu_${MODEL}_retain100_${FORGET}")
            [[ "$target" != base ]] && dirs+=("saves/eval/tofu_${MODEL}_${RETAIN}")
            ;;
        methods)
            for method in "${METHODS_LIST[@]}"; do
                dirs+=("saves/eval/${FORGET}/tofu_${MODEL}_${method}${SUFFIX}")
            done
            ;;
        *) die "unknown target '${target}' (want all|refs|base|retain|methods)" ;;
    esac
done

mkdir -p logs/judge
echo "judge ${MODEL} ${FORGET} (${TARGETS}): ${#dirs[@]} targets, ${JUDGE_WORKERS} workers, ${JUDGE_PARALLEL} in parallel"

pids=()
fail=0
for dir in "${dirs[@]}"; do
    if [[ ! -f "${dir}/TOFU_EVAL.json" ]]; then
        echo "miss  ${dir} (no TOFU_EVAL.json)" >&2
        continue
    fi
    log="logs/judge/${FORGET}_$(basename "$dir").log"
    echo "[run] ${dir}  -> ${log}"
    # Naming a dir picks it as a target but does not force re-scoring: rubrics
    # already in its summary are skipped, so this fills in only what is missing.
    # Export OVERWRITE=1 to redo scores that are already there.
    python scripts/dlm/run_judge.py "$dir" > "$log" 2>&1 &
    pids+=($!)
    while (( $(jobs -rp | wc -l) >= JUDGE_PARALLEL )); do wait -n || fail=1; done
done

for pid in "${pids[@]}"; do wait "$pid" || fail=1; done

echo
grep -h "^ok\|^skip\|Traceback" logs/judge/${FORGET}_*.log 2>/dev/null | tail -40
echo "judge done (fail=${fail})"
