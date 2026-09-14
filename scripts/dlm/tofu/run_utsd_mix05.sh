#!/bin/bash
# DiffusionUTSD mix_beta=0.5 rescue sweep: Dream x forget05/forget10 (where uniform
# unlearning collapsed retain), all 8 GPUs, eval after each. TASK_SUFFIX=_mix05.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export TASK_SUFFIX="${TASK_SUFFIX:-_mix05}"
echo "[utsd-mix05] Dream forget05,forget10  mix_beta=0.5  gpus=${GPUS}"
bash scripts/dlm/tofu/unlearn.sh dream "$GPUS" "forget05,forget10" UTSD \
    trainer.method_args.mix_beta=0.5
echo "======= ALL DONE ======="
