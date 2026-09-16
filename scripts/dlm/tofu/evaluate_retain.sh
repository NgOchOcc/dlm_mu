#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

dlm_config "${1:-}" "${2:-}"
splits="${3:-all}"
if (( $# > 3 )); then shift 3; else shift $#; fi

[[ "$splits" == all ]] && splits=retain100,retain90,retain95,retain99

eval_plan_reset
for split in ${splits//,/ }; do
    case "$split" in
        retain100)
            for forget in forget10 forget05 forget01; do
                eval_plan_add "tofu_${MODEL}_retain100_${forget}" \
                    "$(rundir_for retain100)" "$forget"
            done
            ;;
        retain90|retain95|retain99)
            tofu_split "unlearn${split#retain}"
            eval_plan_add "tofu_${MODEL}_${split}" "$(rundir_for "$split")" "$FORGET"
            ;;
        *)
            die "unknown reference split '${split}' (want retain100|retain90|retain95|retain99)"
            ;;
    esac
done

fail=0
eval_plan_run "$@" || fail=1
python scripts/dlm/show_results.py || true
exit "$fail"
