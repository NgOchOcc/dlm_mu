#!/bin/bash
# SBRU sigma sweep: LLaDA / forget10, full epochs + eval, one arm per sigma.
# The sanity run showed sigma=1.0 saturates the kernel (embedding distances ~0.01-0.3);
# this sweeps sigma matched to that scale. Each arm gets its own TASK_SUFFIX so the
# save/eval dirs never clobber. Launch detached:
#   setsid bash scripts/dlm/tofu/run_sbru_sweep.sh >logs/sbru_sweep.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.."

GPUS="${GPUS:-4,5,6,7}"
SIGMAS="${SIGMAS:-0.05 0.1 0.2}"

for sig in $SIGMAS; do
    tag=$(echo "$sig" | tr '.' 'p')   # 0.05 -> 0p05
    echo "======================================================================="
    echo "[sbru-sweep] sigma=${sig} tag=${tag} gpus=${GPUS} (LLaDA forget10, full + eval)"
    echo "======================================================================="
    TASK_SUFFIX="_sbru_s${tag}" bash scripts/dlm/tofu/unlearn.sh llada "$GPUS" forget10 SBRU \
        trainer.method_args.sigma="${sig}" || {
        echo "[sbru-sweep] sigma=${sig} FAILED (rc=$?), continuing" >&2
    }
done

echo "======= SBRU SWEEP ALL DONE ======="
echo "[sbru-sweep] table: python scripts/dlm/show_results.py" >&2
