#!/usr/bin/env bash
# Forget-only MUSE unlearning + eval for both data splits (News, Books), over
# GA/NPO/SimNPO/WGA. Assumes the SFT targets already exist at
# saves/finetune/muse_<MODEL>_{News,Books}_full (train with finetune.sh first).
# Thin wrapper around run_forget_only.sh: runs it once per split, sequentially.
#
# Usage: run_forget_only_all_splits.sh <dream|llada> <gpus> [methods]
#   run_forget_only_all_splits.sh llada 0,1,2,3
#   METHODS="DiffusionNPO,DiffusionWGA" run_forget_only_all_splits.sh llada 0,1,2,3
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export METHODS="${METHODS:-DiffusionGA,DiffusionNPO,DiffusionSimNPO,DiffusionWGA}"

for split in News Books; do
    bash scripts/dlm/muse/run_forget_only.sh "${1:-}" "${2:-}" "${split}"
done
