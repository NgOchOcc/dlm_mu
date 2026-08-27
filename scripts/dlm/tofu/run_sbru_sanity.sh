#!/bin/bash
# Wait until GPUs 4-7 are truly idle, then launch a 1-epoch SBRU sanity run on
# LLaDA/forget10 (NO_EVAL) to probe the kernel bandwidth sigma: watch sbru/dist_c
# and sbru/L_rep in the log. Gated on free memory so it never fights the mix05
# eval or the cron fluency job for VRAM. Launch detached:
#   setsid bash scripts/dlm/tofu/run_sbru_sanity.sh >logs/sbru_sanity.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.."

GATE_GPUS="${GATE_GPUS:-4 5 6 7}"
RUN_GPUS="${RUN_GPUS:-4,5,6,7}"
FREE_MIN_MIB="${FREE_MIN_MIB:-70000}"   # per-GPU free memory required to call it idle
MAX_WAIT_MIN="${MAX_WAIT_MIN:-180}"

idle() {
    for g in $GATE_GPUS; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
        [[ -n "$free" && "$free" -ge "$FREE_MIN_MIB" ]] || return 1
    done
    return 0
}

echo "[sbru-sanity] waiting for GPUs [$GATE_GPUS] to reach >=${FREE_MIN_MIB}MiB free (max ${MAX_WAIT_MIN}min)"
waited=0
until idle; do
    sleep 60
    waited=$((waited + 1))
    if (( waited % 5 == 0 )); then
        echo "[sbru-sanity] still waiting (${waited}min): $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader -i "${RUN_GPUS}")"
    fi
    if (( waited >= MAX_WAIT_MIN )); then
        echo "[sbru-sanity] gave up after ${MAX_WAIT_MIN}min; GPUs never freed" >&2
        exit 1
    fi
done

echo "[sbru-sanity] GPUs free after ${waited}min. Launching SBRU 1-epoch sanity on ${RUN_GPUS}."
export TASK_SUFFIX="${TASK_SUFFIX:-_sbru_sanity}"
export NO_EVAL=1
bash scripts/dlm/tofu/unlearn.sh llada "$RUN_GPUS" forget10 SBRU \
    trainer.args.num_train_epochs=1
echo "======= SBRU SANITY DONE ======="
