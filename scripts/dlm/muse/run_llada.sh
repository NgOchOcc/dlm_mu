#!/usr/bin/env bash
# Full MUSE run for LLaDA-1.5: the five baselines, then the SelectiveUnlearning
# sweep, over both corpora. Every training run is evaluated as soon as it
# finishes, and the knowmem SFT phase is trained first if it is missing.
#
#   scripts/dlm/muse/run_llada.sh
#   scripts/dlm/muse/run_llada.sh 0,1,2,3
#   DATA_SPLITS=News scripts/dlm/muse/run_llada.sh 0,1,2,3
#   SKIP_BASELINES=1 scripts/dlm/muse/run_llada.sh      # sweep only
#   SKIP_SWEEP=1 scripts/dlm/muse/run_llada.sh          # baselines only
set -uo pipefail
if [[ ! -f scripts/dlm/lib.sh ]]; then
    echo "error: run this from the repository root (no scripts/dlm/lib.sh in $PWD)." >&2
    exit 1
fi

MODEL="${MODEL:-llada1.5}"
GPUS_ARG="${1:-${GPUS:-0,1,2,3}}"
DATA_SPLITS="${DATA_SPLITS:-News Books}"

BASELINES="${BASELINES:-DiffusionGA,DiffusionWGA,DiffusionNPO,DiffusionSimNPO,DiffusionBST}"
# A beta sweep at pw=0.1 plus a preserve sweep at beta=10 -- not a full grid,
# hence PAIRS rather than BETAS x PRESERVE_WEIGHTS.
PAIRS="${PAIRS:-1:0.1 2:0.1 5:0.1 10:0.1 20:0.1 10:0.2 10:0.4 10:0.6 10:1.0}"

echo "=== MUSE: ${MODEL} ==="
echo "  gpus      : ${GPUS_ARG}"
echo "  splits    : ${DATA_SPLITS}"
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
