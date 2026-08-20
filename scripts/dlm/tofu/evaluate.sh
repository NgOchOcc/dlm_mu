#!/usr/bin/env bash
# (Re-)evaluate saved TOFU checkpoints with the tofu_dlm suite. Unlearn.sh and
# retain.sh already evaluate what they train, so this is for filling gaps: the
# reference models, an arm whose eval crashed, or every arm after an eval-code
# change. Everything here runs under ONE forget split; to cover all reference
# models across all three settings in one go use evaluate_retain.sh instead.
#
#   usage: bash scripts/dlm/tofu/evaluate.sh <dream|llada> [gpus] [split] [targets] [hydra overrides...]
#
#     gpus     "0,1,2,3"; default = every visible device (or $GPUS)
#     split    unlearn90 (forget10, default) | unlearn95 (forget05) | unlearn99 (forget01)
#     targets  comma-separated, default all:
#                base     the retain100 "Base" (target) model
#                retain   the retain oracle paired with the split
#                methods  the unlearned checkpoints under saves/unlearn/
#                all      base,retain,methods
#
#   examples:
#     bash scripts/dlm/tofu/evaluate.sh dream 0,1,2,3                      # everything, forget10
#     bash scripts/dlm/tofu/evaluate.sh dream 3 unlearn95 base             # base only, one GPU
#     METHODS=NPO,WGA bash scripts/dlm/tofu/evaluate.sh llada 0,1 unlearn90 methods
#     PARALLEL=1 bash scripts/dlm/tofu/evaluate.sh dream 0,1,2,3 unlearn90 methods
#
#   env:
#     METHODS=NPO,WGA  evaluate only these methods (default: the 5 baselines)
#     TASK_SUFFIX      match the suffix a sweep was trained under (e.g. _tau0.5)
#     OVERWRITE=1      re-evaluate targets that already have a summary
#     PARALLEL=1       one checkpoint per GPU at a time instead of one checkpoint
#                      across all GPUs; output goes to logs/eval/<task>.log
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

dlm_config "${1:-}" "${2:-}"
tofu_split "${3:-unlearn90}"
TARGETS="${4:-all}"
# Whatever is left after the four positional args is forwarded to every eval.
if (( $# > 4 )); then shift 4; else shift $#; fi

SUFFIX="${TASK_SUFFIX:-}"
[[ "$TARGETS" == all ]] && TARGETS=base,retain,methods

# Everything runs under the one forget split this invocation was given; see
# lib.sh (eval planning) for the skip / miss / PARALLEL rules.
eval_plan_reset
for target in ${TARGETS//,/ }; do
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
                task="${FORGET}/tofu_${MODEL}_${method}${SUFFIX}"
                eval_plan_add "$task" "saves/unlearn/${task}" "$FORGET"
            done
            ;;
        *)
            die "unknown target '${target}' (want all|base|retain|methods)"
            ;;
    esac
done

echo "[eval] model=${MODEL} split=${FORGET} targets=${TARGETS}" \
     "gpus=${EVAL_GPUS:-$GPUS} parallel=${PARALLEL:-0} n=${#PLAN_TASKS[@]}"

fail=0
eval_plan_run "$@" || fail=1
echo "[eval] done (fail=${fail})"
python scripts/dlm/show_results.py || true
exit "$fail"
