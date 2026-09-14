#!/bin/bash
# Run SBRU unlearning (train + eval) on TOFU.
#
# Usage (foreground):
#   bash scripts/dlm/tofu/run_sbru.sh
#
# Usage (detached, survives the shell/Claude session -- recommended for real runs):
#   setsid bash scripts/dlm/tofu/run_sbru.sh > logs/sbru_run1.log 2>&1 < /dev/null &
#   # watch:  tr '\r' '\n' < logs/sbru_run1.log | grep -aE '^ep |L_rep|ALL DONE'
#   # table:  python scripts/dlm/show_results.py
#
# Override any of these on the command line, e.g.:
#   GPUS=0,1 MODEL=dream SPLIT=forget01 LAM_RET=1.0 bash scripts/dlm/tofu/run_sbru.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."

MODEL="${MODEL:-llada}"          # llada | dream
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"          # GPU ids
SPLIT="${SPLIT:-forget10}"       # forget01 | forget05 | forget10
SUFFIX="${TASK_SUFFIX:-_sbru_run1}"   # unique save dir per run (avoids clobbering)
OUTPUT_ROOT="${OUTPUT_ROOT:-$(pwd)/outputs}"
mkdir -p "${OUTPUT_ROOT}"/{unlearn,eval}

# --- SBRU hyperparameters (see configs/trainer/DiffusionSBRU.yaml for defaults) ---
SIGMA="${SIGMA:-0.2}"           # repulsion ball radius
SIGMA_ANC="${SIGMA_ANC:-0.2}"    # anchor ball radius (wider than SIGMA)
GAMMA_ANC="${GAMMA_ANC:-1.0}"    # anchor attraction weight
LAM_RET="${LAM_RET:-0.0}"        # retain-NLL weight. 0.0 = retain-FREE (default);
                                 #   set 1.0 for the retain-USING mode that holds utility.

echo "[sbru] model=${MODEL} gpus=${GPUS} split=${SPLIT} suffix=${SUFFIX}"
echo "[sbru] sigma=${SIGMA} sigma_anc=${SIGMA_ANC} gamma_anc=${GAMMA_ANC} lam_ret=${LAM_RET}"
echo "[sbru] outputs=${OUTPUT_ROOT}"

TASK_SUFFIX="${SUFFIX}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
    bash scripts/dlm/tofu/unlearn.sh "${MODEL}" "${GPUS}" "${SPLIT}" SBRU \
    paths.output_dir="${OUTPUT_ROOT}/\${mode}/\${task_name}" \
    trainer.method_args.sigma="${SIGMA}" \
    trainer.method_args.sigma_anc="${SIGMA_ANC}" \
    trainer.method_args.gamma_anc="${GAMMA_ANC}" \
    trainer.method_args.lam_ret="${LAM_RET}"

echo "[sbru] done. Table: python scripts/dlm/show_results.py"

echo pwd
