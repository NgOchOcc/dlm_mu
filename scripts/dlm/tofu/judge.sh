#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}"

export JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
parallel="${JUDGE_PARALLEL:-1}"

tofu_split "${2:-forget10}"
parse_methods "${METHODS:-}"
targets="${3:-all}"
[[ "$targets" == all ]] && targets=refs,methods

dirs=()
for target in ${targets//,/ }; do
    case "$target" in
        refs|base|retain)
            [[ "$target" != retain ]] && dirs+=("saves/eval/tofu_${MODEL}_retain100_${FORGET}")
            [[ "$target" != base ]] && dirs+=("saves/eval/tofu_${MODEL}_${RETAIN}")
            ;;
        methods)
            for method in "${METHODS_LIST[@]}"; do
                dirs+=("saves/eval/${FORGET}/tofu_${MODEL}_${method}${TASK_SUFFIX:-}")
            done
            ;;
        *) die "unknown target '${target}' (want all|refs|base|retain|methods)" ;;
    esac
done

pids=()
fail=0
for dir in "${dirs[@]}"; do
    if [[ ! -f "${dir}/TOFU_EVAL.json" ]]; then
        echo "miss  ${dir} (no TOFU_EVAL.json)" >&2
        continue
    fi
    python scripts/dlm/run_judge.py "$dir" &
    pids+=($!)
    while (( $(jobs -rp | wc -l) >= parallel )); do wait -n || fail=1; done
done

for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "judge done (fail=${fail})"
