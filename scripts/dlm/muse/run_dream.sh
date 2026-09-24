#!/usr/bin/env bash
# Full MUSE run for Dream: the five baselines, then the SelectiveUnlearning
# sweep, over both corpora.
#
# The SFT phases are assumed done -- this script only unlearns. Each method
# trains for UNLEARN_EPOCHS (5) with a checkpoint kept after every epoch, and
# every one of those checkpoints is evaluated into its own directory:
#
#   saves/eval/muse_<MODEL>_<split>_<Method>_ep1/MUSE_SUMMARY.json
#   saves/eval/muse_<MODEL>_<split>_<Method>_ep2/...
#   ...
#   saves/eval/muse_<MODEL>_<split>_<Method>_ep5/...
#
# so a method's quality can be tracked across epochs rather than only at the end.
#
#   scripts/dlm/muse/run_dream.sh
#   scripts/dlm/muse/run_dream.sh 0,1,2,3
#   DATA_SPLITS=News scripts/dlm/muse/run_dream.sh 0,1,2,3
#   SKIP_BASELINES=1 scripts/dlm/muse/run_dream.sh      # sweep only
#   SKIP_SWEEP=1 scripts/dlm/muse/run_dream.sh          # baselines only
set -uo pipefail
if [[ ! -f scripts/dlm/lib.sh ]]; then
    echo "error: run this from the repository root (no scripts/dlm/lib.sh in $PWD)." >&2
    exit 1
fi

MODEL="${MODEL:-dream}"
GPUS_ARG="${1:-${GPUS:-0,1,2,3}}"
DATA_SPLITS="${DATA_SPLITS:-News Books}"

BASELINES="${BASELINES:-DiffusionGA,DiffusionWGA,DiffusionNPO,DiffusionSimNPO,DiffusionBST}"

# Three epochs for every method, checkpointed and evaluated after each one.
# The SFT checkpoints already exist, so both SFT phases are skipped.
export KNOWMEM_SFT="${KNOWMEM_SFT:-0}"

# Read the datasets from disk instead of the Hub. download_data.py saves them
# under DLM_DATASET_DIR; anything missing there still falls back to a Hub load,
# so an incomplete download degrades rather than fails.
export DLM_DATASET_DIR="${DLM_DATASET_DIR:-data/muse}"
if [[ ! -d "$DLM_DATASET_DIR" ]]; then
    echo "note: ${DLM_DATASET_DIR} not found -- datasets will come from the Hub." >&2
    echo "      Run: python3 scripts/dlm/muse/download_data.py" >&2
fi
# A beta sweep at pw=0.1 plus a preserve sweep at beta=10 -- not a full grid,
# hence PAIRS rather than BETAS x PRESERVE_WEIGHTS.
PAIRS="${PAIRS:-1:0.1 2:0.1 5:0.1 10:0.1 20:0.1 10:0.2 10:0.4 10:0.6 10:1.0}"

echo "=== MUSE: ${MODEL} ==="
echo "  gpus      : ${GPUS_ARG}"
echo "  splits    : ${DATA_SPLITS}"
echo "  epochs    : ${UNLEARN_EPOCHS:-5} (evaluated after each)"
echo "  lr        : ${UNLEARN_LR:-3e-6}"
echo "  baselines : ${BASELINES//,/ }"
echo "  pairs     : ${PAIRS}"
echo

fail=0

if [[ "${SKIP_BASELINES:-0}" == 0 ]]; then
    echo "######## step 1: baselines ########"
    METHODS="$BASELINES" \
        bash scripts/dlm/muse/run.sh "$MODEL" "$GPUS_ARG" "$DATA_SPLITS" \
        || { echo "[warn] baselines FAILED" >&2; fail=1; }
    echo
fi

if [[ "${SKIP_SWEEP:-0}" == 0 ]]; then
    echo "######## step 2: SelectiveUnlearning sweep ########"
    PAIRS="$PAIRS" \
        bash scripts/dlm/muse/selective_sweep.sh "$MODEL" "$GPUS_ARG" "$DATA_SPLITS" \
        || { echo "[warn] sweep FAILED" >&2; fail=1; }
    echo
fi

echo "=== results ==="
python3 scripts/dlm/show_results.py --dataset muse --model "$MODEL" || true

exit "$fail"
