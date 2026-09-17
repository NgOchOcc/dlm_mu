#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

dlm_config "${1:-}" "${2:-}"
tofu_split "${3:-unlearn90}"
targets="${4:-all}"
if (( $# > 4 )); then shift 4; else shift $#; fi

[[ "$targets" == all ]] && targets=base,retain,methods

eval_plan_reset
for target in ${targets//,/ }; do
    case "$target" in
        base)
            eval_plan_add "tofu_${MODEL}_retain100_${FORGET}" "$(rundir_for retain100)" "$FORGET"
            ;;
        retain)
            eval_plan_add "tofu_${MODEL}_${RETAIN}" "$(rundir_for "$RETAIN")" "$FORGET"
            ;;
        methods)
            parse_methods "${METHODS:-}"
            for method in "${METHODS_LIST[@]}"; do
                task="${FORGET}/tofu_${MODEL}_${method}${TASK_SUFFIX:-}"
                eval_plan_add "$task" "saves/unlearn/${task}" "$FORGET"
            done
            ;;
        *)
            die "unknown target '${target}' (want all|base|retain|methods)"
            ;;
    esac
done

fail=0
eval_plan_run "$@" || fail=1
python scripts/dlm/show_results.py || true
exit "$fail"
