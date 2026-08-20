#!/usr/bin/env bash
# Measure TrajRMDU step time / peak GPU memory under a given accelerate config,
# so a full sweep is only launched once its real ETA is known.
#
# The offload config that the TOFU scripts default to puts both optimizer and
# params on the CPU, which leaves the H100s idle (~0% util, 77 s/it). This probe
# runs a handful of steps so the no-offload config can be compared on the same
# data before committing GPU-days to a sweep.
#
# usage: bash scripts/dlm/probe_step_time.sh <dream|llada> <gpus> [steps]
# env:
#   UNLEARN_ACCEL   accelerate config to measure (defaults to the offload one)
#   GRAD_CKPT_OVERRIDE  force gradient checkpointing on/off (llada defaults off)
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

STEPS="${3:-12}"
FORGET=forget01
RETAIN=retain99
HOLDOUT=holdout01
TAG="probe_$(basename "${UNLEARN_ACCEL}" .yaml)"
TASK="${TAG}/tofu_${MODEL}_DiffusionTrajRMDU_probe"

echo "probe: model=${MODEL} gpus=${GPUS} steps=${STEPS} accel=${UNLEARN_ACCEL} grad_ckpt=${GRAD_CKPT}"

# Sample GPU memory in the background; the max over the run is the peak we care
# about (torch's own counter is per-rank and not printed by the trainer).
MEMLOG=$(mktemp)
( while true; do
    nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits >> "$MEMLOG"
    sleep 5
  done ) & sampler=$!
trap 'kill $sampler 2>/dev/null; rm -f "$MEMLOG"' EXIT

EXP=unlearn/tofu/dlm_traj_rmdu \
    tofu_unlearn DiffusionTrajRMDU "$TASK" "$FORGET" "$RETAIN" "$HOLDOUT" 25 \
    trainer.method_args.beta=1 \
    trainer.method_args.mask_source=trajectory \
    trainer.method_args.target_mode=residual \
    trainer.method_args.trajectory_steps=0 \
    trainer.method_args.alpha=1.0 \
    trainer.method_args.gamma=1.0 \
    +trainer.args.max_steps="$STEPS"
rc=$?

kill $sampler 2>/dev/null
peak=$(sort -n "$MEMLOG" | tail -1)
echo "probe result: model=${MODEL} accel=${UNLEARN_ACCEL} grad_ckpt=${GRAD_CKPT} rc=${rc} peak_mem_mib=${peak:-unknown}"
