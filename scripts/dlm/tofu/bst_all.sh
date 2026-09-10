#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/dlm/tofu/bst_all.sh [dream_epochs]
# Example: bash scripts/dlm/tofu/bst_all.sh 2,4

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-$USER}"
mkdir -p "$TRITON_CACHE_DIR"

epochs_dream="${1:-${DREAM_EPOCHS:-3,4}}"
splits="forget10,forget05"
gpus="0,1,2,3"
launcher="$(dirname "$0")/unlearn.sh"

for epochs in "$epochs_dream"; do
    [[ "$epochs" =~ ^[0-9]+,[0-9]+$ ]] || {
        echo "Epochs must contain two integers: forget10,forget05" >&2
        exit 1
    }
done

echo "[BST] Dream: splits=${splits}, epochs=${epochs_dream}, gpus=${gpus}"

EPOCHS="$epochs_dream" bash "$launcher" \
    dream "$gpus" "$splits" BST \
    trainer.method_args.lambda_bst="${LAMBDA_BST:-0.2}" \
    trainer.method_args.top_k="${TOP_K:-10}"