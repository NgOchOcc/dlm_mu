#!/usr/bin/env bash
# WHP baseline (scripts/dlm/tofu/run_whp.sh) on one 4-GPU node, models one after another.
# Effective batch 4 GPUs x 4 x 4 = 64, as in the Selective runs. Stages whose outputs
# already exist (reinforced model, label file) are reused.
#
# usage: as the command of a 4-GPU runai training job (runs in the foreground), or inside a pod:
#   setsid nohup bash scripts/dlm/tofu/run_whp_4gpu.sh [llada llada1.5 dream] > logs/whp/4gpu.log 2>&1 < /dev/null &
# logs: logs/whp/<model>_4gpu.log; logs/whp/done_4gpu is written at the end.
cd "$(dirname "$0")/../../.."
export PATH=/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/miniconda3/envs/unlearning/bin:$PATH
export HF_HOME=/prj/corp/airesearch/lasvegas/vol20-scratch/gnguyenn/.cache/hf_cache  # TOFU is cached here; jobs don't source ~/.bashrc
export HF_DATASETS_OFFLINE=1
export TRAIN_ACCEL=${TRAIN_ACCEL:-configs/accelerate/dlm_config_nooffload_4gpu.yaml}
export UNLEARN_ACCEL=${UNLEARN_ACCEL:-configs/accelerate/dlm_config_nooffload_4gpu.yaml}
export EVAL_ACCEL=configs/accelerate/eval_4gpu.yaml
export TRAIN_BS=4 GRAD_ACCUM=4
mkdir -p logs/whp

models=("$@")
(( ${#models[@]} )) || models=(llada llada1.5 dream)
for m in "${models[@]}"; do
    echo "[$(date -u +%FT%TZ)] start $m"
    bash scripts/dlm/tofu/run_whp.sh "$m" 0,1,2,3 > "logs/whp/${m}_4gpu.log" 2>&1 || echo "[$(date -u +%FT%TZ)] $m failed, see logs/whp/${m}_4gpu.log"
done
echo "[$(date -u +%FT%TZ)] ALL_DONE" | tee logs/whp/done_4gpu
