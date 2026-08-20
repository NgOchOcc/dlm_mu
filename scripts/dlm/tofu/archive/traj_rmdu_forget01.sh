#!/usr/bin/env bash
# TrajRMDU forget01 (unlearn99) sweep for both DLMs, sized from measured step times.
#
# Why this differs from `traj_rmdu.sh <model> 0,1 forget01`:
#
# The TOFU accelerate default (configs/accelerate/dlm_config.yaml) runs ZeRO-3 with
# both the optimizer and the params offloaded to the CPU. For a 7-8B model on 80 GB
# H100s that offload is pure overhead -- the GPUs sit at ~0% util waiting on PCIe,
# and a step costs ~77 s. Measured on forget01 (scripts/dlm/probe_step_time.sh):
#
#   config                                  Dream       LLaDA
#   2 GPU + CPU offload                     77 s/it     78 s/it
#   2 GPU, no offload                       17.6 s/it   OOM at 77.6/79 GB
#   4 GPU, no offload, GRAD_ACCUM=2         12.8 s/it   16.0 s/it (peak 66.6 GB)
#
# LLaDA cannot drop the offload on 2 GPUs: the fp32 optimizer state sharded over
# two ranks alone is ~48 GB/GPU. Sharding over 4 halves it, so each model takes all
# four GPUs and the two run back to back rather than side by side.
#
# GRAD_ACCUM=2 is what keeps this comparable to the 2-GPU setup: 4 GPUs would
# otherwise double the effective batch (4x4x4=64). At accum 2 it stays 4x2x4=32,
# so the step count and optimisation trajectory match the 2-GPU configuration and
# only the wall-clock changes. Dropping the offload and enabling checkpointing are
# both mathematically inert, so these runs stay comparable to the other splits.
#
# usage: bash scripts/dlm/tofu/traj_rmdu_forget01.sh
# env:   BETAS ("0 0.25 0.5 1 2"), GPUS ("0,1,2,3")
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export BETAS="${BETAS:-0 0.25 0.5 1 2}"
export UNLEARN_ACCEL=configs/accelerate/dlm_config_nooffload_4gpu.yaml
export GRAD_ACCUM=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS_ALL="${GPUS:-0,1,2,3}"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/traj_rmdu

for model in dream llada; do
    log="logs/traj_rmdu/${model}_forget01_nooffload_${TS}.log"
    # LLaDA's activations only fit once checkpointing is on (see table above).
    ckpt=$([[ "$model" == llada ]] && echo true || echo "")
    echo "=== ${model} forget01 betas='${BETAS}' gpus=${GPUS_ALL} -> ${log}"
    GRAD_CKPT_OVERRIDE="$ckpt" MAIN_PORT=29500 EVAL_PORT=29600 \
        bash scripts/dlm/tofu/traj_rmdu.sh "$model" "$GPUS_ALL" forget01 > "$log" 2>&1
    echo "=== ${model} exit=$?"
done
