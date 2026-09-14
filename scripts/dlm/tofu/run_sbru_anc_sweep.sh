#!/bin/bash
# SBRU anchor-bandwidth sweep: fix the repulsion sigma small (0.05) and widen the
# ANCHOR sigma so L_anc keeps a broad attraction basin after repulsion pushes e_hat
# off the forget centre. The tied-sigma retain-free sweep collapsed retain at every
# sigma (retain ROUGE 0.96 -> 0.00-0.03) because the anchor died at the same scale
# the repulsion did; this decouples them. One arm per sigma_anc, each its own
# TASK_SUFFIX so save/eval dirs never clobber. Launch detached:
#   setsid bash scripts/dlm/tofu/run_sbru_anc_sweep.sh >logs/sbru_anc_sweep.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.."

GPUS="${GPUS:-4,5,6,7}"
SIGMA_REP="${SIGMA_REP:-0.05}"
SIGMA_ANCS="${SIGMA_ANCS:-0.5 1.0 2.0}"

for sa in $SIGMA_ANCS; do
    tag=$(echo "$sa" | tr '.' 'p')   # 0.5 -> 0p5
    echo "======================================================================="
    echo "[sbru-anc] sigma=${SIGMA_REP} sigma_anc=${sa} tag=${tag} gpus=${GPUS} (LLaDA forget10, full + eval)"
    echo "======================================================================="
    TASK_SUFFIX="_sbru_sa${tag}" bash scripts/dlm/tofu/unlearn.sh llada "$GPUS" forget10 SBRU \
        trainer.method_args.sigma="${SIGMA_REP}" \
        trainer.method_args.sigma_anc="${sa}" || {
        echo "[sbru-anc] sigma_anc=${sa} FAILED (rc=$?), continuing" >&2
    }
done

echo "======= SBRU ANC SWEEP ALL DONE ======="
echo "[sbru-anc] table: python scripts/dlm/show_results.py" >&2
