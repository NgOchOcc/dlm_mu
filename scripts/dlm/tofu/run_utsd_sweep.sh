#!/bin/bash
# DiffusionUTSD (uniform-target forget) sweep: llada + dream x forget01/05/10, all on
# GPUS 0-7, one arm at a time (each arm uses all 8 GPUs). Eval runs after each arm.
# Launched detached via setsid so it survives the parent shell exiting.
set -uo pipefail
cd "$(dirname "$0")/../../.."

export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export TASK_SUFFIX="${TASK_SUFFIX:-_uniform}"
SPLITS="${SPLITS:-forget01,forget05,forget10}"

for model in llada dream; do
    echo "======================================================================="
    echo "[utsd-sweep] model=${model} splits=${SPLITS} suffix=${TASK_SUFFIX} gpus=${GPUS}"
    echo "======================================================================="
    bash scripts/dlm/tofu/unlearn.sh "$model" "$GPUS" "$SPLITS" UTSD || {
        echo "[utsd-sweep] model=${model} FAILED (rc=$?), continuing to next model" >&2
    }
done

echo "======= ALL DONE ======="
