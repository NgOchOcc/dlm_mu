set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

dlm_config "${1:-}" "${2:-}"
SPLITS="${3:-all}"
if (( $# > 3 )); then shift 3; else shift $#; fi

[[ "$SPLITS" == all ]] && SPLITS=retain100,retain90,retain95,retain99
ALL_SETTINGS=(forget10 forget05 forget01)

eval_plan_reset
for split in ${SPLITS//,/ }; do
    case "$split" in
        retain100)
            for forget in "${ALL_SETTINGS[@]}"; do
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

echo "[eval] model=${MODEL} refs=${SPLITS}" \
     "gpus=${EVAL_GPUS:-$GPUS} parallel=${PARALLEL:-0} n=${#PLAN_TASKS[@]}"

fail=0
eval_plan_run "$@" || fail=1
echo "[eval] done (fail=${fail})"
python scripts/dlm/show_results.py || true
exit "$fail"
