#!/usr/bin/env bash
# Run the FULL Dream forget10 unlearning suite under the 1/t ELBO weight
# (loss_weight_type=scheduler) instead of Dream's default CART weight, and
# evaluate every arm. One self-contained command reproduces the whole table.
#
# WHY scheduler over cart -------------------------------------------------------
# On Dream, cart[geo_p:0.3] is ~0.16x the 1/t ELBO weight and tilts weight away
# from the high-mask regime the sampler starts from. Two measured consequences:
#   * NPO/SimNPO: the log-sigmoid brake beta*Delta shrinks ~6x, so beta=0.1 acts
#     like ~0.017, the brake never engages and NPO degenerates into ascent.
#   * retain anchor is starved exactly at high mask rate -> retain denoises to
#     EOS -> empty generations -> retain ROUGE collapses.
# Swapping to scheduler (w(t)=1/t, the unbiased MDLM ELBO LLaDA already trains
# under) triples NPO retain (0.118 -> 0.428) and lifts SimNPO retain to 0.814.
# See memory/project_dream_retain_collapse_cart_weighting.
#
# The weight is FORCED here (LOSS_WEIGHT_OVERRIDE=scheduler exported before
# lib.sh reads it) so no arm can silently fall back to cart -- a trap that has
# bitten this exact workflow before. Every task carries a _wsched suffix, so
# these land beside the cart baselines instead of overwriting them.
#
# Per-method hyperparameters ----------------------------------------------------
# GradDiff/GradAscent have no beta, so the ELBO weight only rescales their
# effective LR and they still diverge at lr=1e-5. GradDiff is therefore run at a
# bounded lr=1.7e-6 (~6x smaller, undoing the weight's rescale); this IS the
# canonical wsched GradDiff. GradAscent is pure unbounded ascent and collapses by
# design -- kept only as the baseline floor.
#
# Usage:
#   scripts/dlm/tofu/unlearn90_wsched.sh [dream] [gpu-list] [extra hydra args...]
#   ARMS="DiffusionNPO DiffusionSimNPO" scripts/dlm/tofu/unlearn90_wsched.sh dream 0,1,2,3
#   SKIP_DONE=1 scripts/dlm/tofu/unlearn90_wsched.sh   # resume: skip arms whose
#                                                      # eval summary already exists
#   GRADDIFF_LR=1e-6 EPOCHS=10 scripts/dlm/tofu/unlearn90_wsched.sh
#
# Long run -- launch detached so it survives the shell/session:
#   setsid scripts/dlm/tofu/unlearn90_wsched.sh dream 0,1,2,3,4,5,6,7 \
#     > logs/unlearn90_wsched_all.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

# Force the ELBO weight before dlm_config reads the LOSS_WEIGHT_OVERRIDE hook, so
# every arm runs under scheduler regardless of the caller's environment.
export LOSS_WEIGHT_OVERRIDE=scheduler
source scripts/dlm/lib.sh
dlm_config "${1:-dream}" "${2:-}"
# Drop the args dlm_config consumed (backbone, gpu-list); anything left is a
# hydra override forwarded to every arm, e.g. trainer.method_args.beta=0.4.
(( $# )) && shift $(( $# < 2 ? $# : 2 ))

[[ "$LOSS_WEIGHT" == scheduler ]] || {
    echo "error: expected scheduler weight, got '$LOSS_WEIGHT'" >&2; exit 1; }

FORGET=forget10
RETAIN=retain90
HOLDOUT=holdout10
# dlm_config just set EPOCHS=150 (that is the *finetune* length); reading
# ${EPOCHS:-10} here would silently inherit 150. Use a distinct env var so an
# unlearn run has its own default and stays overridable.
#
# Epoch budget is backbone-specific: Dream diverges faster under the 1/t ELBO
# weight (GradAscent/GradDiff are unbounded ascent), so it gets 5; LLaDA is
# stable and gets the full 10. Override for either via UNLEARN_EPOCHS.
case "$MODEL" in
    *Dream*) DEFAULT_EPOCHS=5 ;;
    *LLaDA*) DEFAULT_EPOCHS=10 ;;
    *)       DEFAULT_EPOCHS=5 ;;
esac
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-$DEFAULT_EPOCHS}"
SUFFIX="${TASK_SUFFIX:-_wsched}"
GRADDIFF_LR="${GRADDIFF_LR:-1.7e-6}"

# Full baseline suite; override with e.g. ARMS="DiffusionNPO DiffusionSimNPO".
read -r -a ARMS <<< "${ARMS:-DiffusionGradAscent DiffusionGradDiff DiffusionNPO DiffusionSimNPO DiffusionWGA}"

echo "[wsched] backbone=${MODEL} weight=${LOSS_WEIGHT} epochs=${UNLEARN_EPOCHS}" \
     "gpus=${GPUS} arms=${ARMS[*]}" >&2

for trainer in "${ARMS[@]}"; do
    task="${FORGET}/tofu_${MODEL}_${trainer}${SUFFIX}"
    summary="saves/eval/${task}/TOFU_SUMMARY.json"

    # Opt-in crash-resume. NOTE: not recipe-aware -- it only checks that some
    # eval exists, so clear a stale summary (or leave SKIP_DONE unset) if you
    # changed hyperparameters for an arm since the last run.
    if [[ -n "${SKIP_DONE:-}" && -f "$summary" ]]; then
        echo "[skip] ${task} already evaluated (unset SKIP_DONE to redo)" >&2
        continue
    fi

    # Per-method extra overrides, appended after the shared ones so they win.
    extra=()
    case "$trainer" in
        DiffusionGradDiff) extra=(trainer.args.learning_rate="${GRADDIFF_LR}") ;;
    esac

    tofu_unlearn "$trainer" "$task" "$FORGET" "$RETAIN" "$HOLDOUT" "$UNLEARN_EPOCHS" \
        "${extra[@]}" "$@"
    tofu_eval "saves/unlearn/${task}" "$task" "$FORGET"
done

echo "[wsched] done. Table: python scripts/dlm/show_results.py" >&2
