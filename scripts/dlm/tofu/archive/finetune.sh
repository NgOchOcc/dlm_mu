#!/usr/bin/env bash
# Full-finetune the "Base" (target) DLM on TOFU_QA_full.
#   usage: scripts/dlm/tofu/finetune.sh <dream|llada> [gpus] [extra hydra overrides...]
#   e.g.:  GPUS=0,1,2,3,4,5,6,7 scripts/dlm/tofu/finetune.sh dream
#          scripts/dlm/tofu/finetune.sh llada 0,1,2,3,4,5,6,7 trainer.method_args.eos_pad_to=128
# Args after the gpu list are forwarded verbatim to the trainer (e.g. to enable
# eos_pad_to, the fixed-block EOS-termination fix).
#
# Crash-resume: checkpoints are resumable by default (optimizer/scheduler/RNG saved,
# save_only_model=false), kept SAVE_TOTAL_LIMIT (default 2) at a time. To resume an
# interrupted run, relaunch the SAME command with RESUME=auto — it picks up the
# latest checkpoint under the run's output_dir, or starts fresh if none exists:
#   RESUME=auto GPUS=0,1,2,3,4,5,6,7 scripts/dlm/tofu/finetune.sh dream
# Tunables: SAVE_STRATEGY (epoch|steps, default epoch), SAVE_STEPS (default 500,
# used when SAVE_STRATEGY=steps for finer-grained resume), SAVE_TOTAL_LIMIT,
# SAVE_ONLY_MODEL (set true to disable resume and save disk).
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

task="$(basename "$SFT_CKPT")"
tofu_train "$task" "$EPOCHS" \
    forget_split=forget10 \
    holdout_split=holdout10 \
    "${@:3}"
