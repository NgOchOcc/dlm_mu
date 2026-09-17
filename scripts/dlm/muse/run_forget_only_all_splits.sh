#!/usr/bin/env bash
# SFT finetune (if missing) + forget-only MUSE unlearning + eval for both data
# splits (News, Books), over GA/NPO/SimNPO/WGA. Runs finetune.sh first for each
# split (it skips training if saves/finetune/muse_<MODEL>_<split>_full already
# exists), then run_forget_only.sh for that same split, sequentially.
#
# Usage: run_forget_only_all_splits.sh <dream|llada> <gpus> [methods]
#   run_forget_only_all_splits.sh llada 0,1,2,3
#   METHODS="DiffusionNPO,DiffusionWGA" run_forget_only_all_splits.sh llada 0,1,2,3
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export METHODS="${METHODS:-DiffusionGA,DiffusionNPO,DiffusionSimNPO,DiffusionWGA}"

for split in News Books; do
    bash scripts/dlm/muse/finetune.sh "${1:-}" "${2:-}" "${split}"
    bash scripts/dlm/muse/run_forget_only.sh "${1:-}" "${2:-}" "${split}"
done
