#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

FORGET="${3:-forget10}"
tofu_eval "$SFT_CKPT" "tofu_${MODEL}_retain100_${FORGET}" "$FORGET"
