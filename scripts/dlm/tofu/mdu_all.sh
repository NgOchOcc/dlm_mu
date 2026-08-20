#!/usr/bin/env bash
# MDU over the whole TOFU matrix: both DLMs x all three forget settings, one tau.
# Each arm trains and then evaluates (this calls mdu.sh -> unlearn.sh, and
# unlearn.sh already evaluates whatever it trains), so one invocation fills a
# complete MDU row of the results table.
#
# Arms run strictly one after another: a single arm already uses every GPU it is
# handed, so there is nothing to overlap. To use two GPU groups at once, launch
# this twice with disjoint gpus and disjoint models/splits.
#
#   usage: bash scripts/dlm/tofu/mdu_all.sh [gpus] [models] [splits]
#
#     gpus    "0,1,2,3"; default = every visible device (or $GPUS)
#     models  comma-separated, default "dream,llada"
#     splits  comma-separated, default "unlearn90,unlearn95,unlearn99"
#             (= forget10, forget05, forget01)
#
#   examples:
#     bash scripts/dlm/tofu/mdu_all.sh 4,5,6,7                        # 6 arms, tau=0.5
#     bash scripts/dlm/tofu/mdu_all.sh 0,1,2,3 llada                  # llada, all splits
#     bash scripts/dlm/tofu/mdu_all.sh 4,5,6,7 dream unlearn95,unlearn99
#     TAUS=0.5 SKIP_DONE=0 bash scripts/dlm/tofu/mdu_all.sh 4,5,6,7   # force redo
#
#   env:
#     TAUS="0.5"      tau values forwarded to mdu.sh (default 0.5)
#     SKIP_DONE=1     skip arms that already have an eval summary; ON by default
#                     here so a rerun after a crash resumes instead of redoing.
#                     Note it keys off the *eval* summary, so an arm that trained
#                     but never evaluated is still retrained -- evaluate those
#                     first with evaluate.sh (METHODS=MDU TASK_SUFFIX=_tau0.5).
#     NO_EVAL=1       train only, no eval
#     UNLEARN_EPOCHS  override the per-split default; unset means lib.sh decides
#                     (tofu_split's SPLIT_EPOCHS: 10 / 20 / 50 for
#                     forget10 / forget05 / forget01)
#     ...plus anything unlearn.sh / lib.sh read (TRAIN_BS, GRAD_ACCUM, OFFLOAD, ...)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../.." || exit 1

GPUS_ARG="${1:-}"
MODELS="${2:-dream,llada}"
SPLITS="${3:-unlearn90,unlearn95,unlearn99}"
export SKIP_DONE="${SKIP_DONE:-1}"

for model in ${MODELS//,/ }; do
    case "$model" in
        dream|llada) ;;
        *) echo "error: unknown model '${model}' (want dream|llada)" >&2; exit 1 ;;
    esac
done
for split in ${SPLITS//,/ }; do
    case "$split" in
        unlearn90|unlearn95|unlearn99|forget10|forget05|forget01) ;;
        *) echo "error: unknown split '${split}'" >&2; exit 1 ;;
    esac
done

echo "[mdu_all] tau=${TAUS:-0.5} models=${MODELS} splits=${SPLITS}" \
     "gpus=${GPUS_ARG:-${GPUS:-auto}} skip_done=${SKIP_DONE}" >&2

# One failing arm must not take the rest of the matrix down with it.
ok=() bad=()
for model in ${MODELS//,/ }; do
    for split in ${SPLITS//,/ }; do
        echo "########## MDU tau=${TAUS:-0.5} ${model} ${split} ##########" >&2
        if bash "$HERE/mdu.sh" "$model" "$GPUS_ARG" "$split"; then
            ok+=("${model}/${split}")
        else
            bad+=("${model}/${split}")
            echo "[mdu_all] FAILED ${model}/${split} -- continuing" >&2
        fi
    done
done

echo "[mdu_all] done: ${#ok[@]} ok${ok[*]+ (${ok[*]})}, ${#bad[@]} failed${bad[*]+ (${bad[*]})}" >&2
python scripts/dlm/show_results.py || true
(( ${#bad[@]} == 0 ))
