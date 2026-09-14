#!/usr/bin/env bash
# Train unlearning with DiffusionSBRUTSD: SBRU repulsion at PCSS-selected forget
# positions + inherited SimpleTSD self-distillation at retain positions R = M\A_hat
# (retain-free utility via the alpha composition; no training on the retain corpus).
#
# Usage:
#   bash scripts/dlm/tofu/run_sbrutsd.sh                 # defaults below
#   MODEL=dream GPUS=0,1 SPLIT=forget05 bash scripts/dlm/tofu/run_sbrutsd.sh
#   SIGMA=0.1 SIGMA_ANC=2.0 GAMMA_ANC=1.0 ALPHA=0.5 bash scripts/dlm/tofu/run_sbrutsd.sh
#
# Env vars (all optional):
#   MODEL       llada | dream                      (default: llada)
#   GPUS        comma-separated device ids         (default: 0,1,2,3)
#   SPLIT       forget10 | forget05 | forget01     (default: forget10)
#   TASK_SUFFIX unique suffix for the save dir     (default: _sbrutsd_run1)
#   SIGMA       repulsion kernel bandwidth         (default: median)
#                 median = SVGD heuristic h=med^2/log(n) each step;
#                 a positive float = fixed exp(-||.||^2 / 2 sigma^2) kernel
#   SIGMA_FALLBACK fixed sigma when heuristic degenerates (default: 0.05)
#   SIGMA_ANC   anchor kernel bandwidth (fixed)     (default: 1.0)
#   GAMMA_ANC   anchor attraction weight           (default: 1.0)
#   ALPHA       forget vs self-distill trade-off   (default: 0.5)
#   CENTRE_MODE gold | ref_soft                    (default: gold)
# Any extra args are passed straight through to unlearn.sh / hydra.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODEL="${MODEL:-llada}"
GPUS="${GPUS:-0,1,2,3}"
SPLIT="${SPLIT:-forget10}"
SUFFIX="${TASK_SUFFIX:-_sbrutsd_run2}"
SIGMA="${SIGMA:-median}"
SIGMA_FALLBACK="${SIGMA_FALLBACK:-0.05}"
SIGMA_ANC="${SIGMA_ANC:-1.0}"
GAMMA_ANC="${GAMMA_ANC:-5.0}"
ALPHA="${ALPHA:-0.8}"
CENTRE_MODE="${CENTRE_MODE:-gold}"

mkdir -p logs
LOG="logs/sbrutsd${SUFFIX}_no_distil.log"

echo "[run_sbrutsd] model=${MODEL} gpus=${GPUS} split=${SPLIT} suffix=${SUFFIX}" >&2
echo "[run_sbrutsd] sigma=${SIGMA} sigma_fallback=${SIGMA_FALLBACK} sigma_anc=${SIGMA_ANC} gamma_anc=${GAMMA_ANC} alpha=${ALPHA} centre=${CENTRE_MODE}" >&2
echo "[run_sbrutsd] log -> ${LOG}" >&2

TASK_SUFFIX="${SUFFIX}" setsid bash scripts/dlm/tofu/unlearn.sh \
    "${MODEL}" "${GPUS}" "${SPLIT}" SBRUTSD \
    trainer.method_args.sigma="${SIGMA}" \
    trainer.method_args.sigma_fallback="${SIGMA_FALLBACK}" \
    trainer.method_args.sigma_anc="${SIGMA_ANC}" \
    trainer.method_args.gamma_anc="${GAMMA_ANC}" \
    trainer.method_args.alpha="${ALPHA}" \
    trainer.method_args.centre_mode="${CENTRE_MODE}" \
    "$@" > "${LOG}" 2>&1 < /dev/null &

echo "[run_sbrutsd] launched pid=$! (detached via setsid); tail -f ${LOG}" >&2
