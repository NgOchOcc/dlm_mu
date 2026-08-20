#!/usr/bin/env bash
# TSD hyper-parameter sweep on TOFU (forget-only, retain-free).
#
# Runs the full grid of TSD's tunable knobs, one unlearn+eval per combination,
# each landing in its own task dir so results never collide:
#
#   tofu_<model>_DiffusionTSD_<mask>_<demotion>[_k<K>][_lr<LR>]
#
# On TOFU the answer is the whole response (A = R), so self_distill /
# balanced_weight are vacuous and the meaningful axes are mask_states x demotion
# (+ belief_top_k when demotion=belief). learning_rate is the main numeric knob
# for forget strength; sweep it via LRS.
#
#   usage: bash scripts/dlm/tofu/tsd_sweep.sh <dream|llada> [gpus] [split]
#   env (space-separated lists; defaults in []):
#     MASK_STATES_LIST="pathwise single"   mask sampler variants        [both]
#     DEMOTIONS_LIST="minimal belief"       teacher-edit variants        [both]
#     BELIEF_TOP_KS="5 10"                  top-K, only for demotion=belief [5 10]
#     LRS="1e-5 5e-6"                       learning rates; empty = config default [empty]
#     UNLEARN_EPOCHS=N                      override epochs for every arm (else split default)
#     SKIP_DONE=1                           skip arms already evaluated  [on]
#     NO_EVAL=1                             train only, skip eval        [off]
#     DRY_RUN=1                             print the plan, run nothing
#
#   Long sweeps must survive the parent shell dying:
#     setsid env LRS="1e-5 5e-6" bash scripts/dlm/tofu/tsd_sweep.sh dream 0 unlearn90 \
#       > logs/tsd_sweep_dream_unlearn90.log 2>&1 < /dev/null &
#
#   View arms (they carry a suffix, so use --mode both, not forget_only):
#     python scripts/dlm/show_results.py --mode both
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

MODEL_ARG="${1:?usage: tsd_sweep.sh <dream|llada> [gpus] [split]}"
GPUS_ARG="${2:-0}"
SPLIT_ARG="${3:-unlearn90}"

read -r -a MASK_ARR    <<< "${MASK_STATES_LIST:-pathwise single}"
read -r -a DEMOTE_ARR  <<< "${DEMOTIONS_LIST:-minimal belief}"
read -r -a TOPK_ARR    <<< "${BELIEF_TOP_KS:-5 10}"
read -r -a LR_ARR      <<< "${LRS:-}"
# No LR sweep -> one arm at the config-default lr (empty sentinel, no suffix/override).
(( ${#LR_ARR[@]} == 0 )) && LR_ARR=("")

export SKIP_DONE="${SKIP_DONE:-1}"   # resume-friendly by default

# Enumerate the grid first so we can announce it (and support DRY_RUN).
declare -a ARM_SUFFIX ARM_MASK ARM_DEMOTE ARM_TOPK ARM_LR
for mask in "${MASK_ARR[@]}"; do
    for demotion in "${DEMOTE_ARR[@]}"; do
        # belief_top_k only varies the objective under belief demotion; minimal
        # collapses the K axis to a single arm.
        if [[ "$demotion" == "belief" ]]; then
            ks=("${TOPK_ARR[@]}")
        else
            ks=("5")   # unused by minimal; kept explicit for reproducibility
        fi
        for k in "${ks[@]}"; do
            for lr in "${LR_ARR[@]}"; do
                suffix="_${mask}_${demotion}"
                [[ "$demotion" == "belief" ]] && suffix="${suffix}_k${k}"
                [[ -n "$lr" ]] && suffix="${suffix}_lr${lr}"
                ARM_SUFFIX+=("$suffix")
                ARM_MASK+=("$mask")
                ARM_DEMOTE+=("$demotion")
                ARM_TOPK+=("$k")
                ARM_LR+=("$lr")
            done
        done
    done
done

n=${#ARM_SUFFIX[@]}
echo "[sweep] TSD on TOFU: model=${MODEL_ARG} gpus=${GPUS_ARG} split=${SPLIT_ARG} arms=${n}" >&2
for i in "${!ARM_SUFFIX[@]}"; do
    echo "  [$((i + 1))/${n}] tofu_<model>_DiffusionTSD${ARM_SUFFIX[$i]}" >&2
done
if [[ -n "${DRY_RUN:-}" ]]; then
    echo "[sweep] DRY_RUN set; nothing executed." >&2
    exit 0
fi

fail=0
for i in "${!ARM_SUFFIX[@]}"; do
    mask="${ARM_MASK[$i]}"; demotion="${ARM_DEMOTE[$i]}"
    k="${ARM_TOPK[$i]}"; lr="${ARM_LR[$i]}"; suffix="${ARM_SUFFIX[$i]}"
    echo "==================== arm $((i + 1))/${n}: ${suffix#_} ====================" >&2

    overrides=(
        trainer.method_args.mask_states="${mask}"
        trainer.method_args.demotion="${demotion}"
        trainer.method_args.belief_top_k="${k}"
        trainer.method_args.self_distill=true
        trainer.method_args.balanced_weight=true
        trainer.method_args.alpha=0.0
        trainer.method_args.gamma=1.0
    )
    [[ -n "$lr" ]] && overrides+=(trainer.args.learning_rate="${lr}")

    # One arm failing (OOM, eval hiccup) must not abort the rest of the sweep.
    if ! TASK_SUFFIX="${suffix}" \
        bash scripts/dlm/tofu/unlearn.sh "${MODEL_ARG}" "${GPUS_ARG}" "${SPLIT_ARG}" TSD \
        "${overrides[@]}"; then
        echo "[warn] arm FAILED: ${suffix#_}; continuing" >&2
        fail=$((fail + 1))
    fi
done

echo "[sweep] done: $((n - fail))/${n} arms ok, ${fail} failed." >&2
echo "[sweep] table: python scripts/dlm/show_results.py --mode both" >&2
