#!/usr/bin/env bash
# TOFU forget10 evaluation of the WHP models (saves/unlearn/forget10/tofu_<model>_DiffusionWHP_alpha5) on a
# 4-GPU node, one model per GPU in parallel: the DLM evaluator does not shard across ranks, so a single process
# per model is fastest (~50 min each). Writes saves/eval/forget10/tofu_<model>_DiffusionWHP_alpha5. Metrics
# already in its TOFU_EVAL.json are kept (eval.tofu_dlm.overwrite=false), so a preempted job resumes where it stopped.
#
# usage: as the command of a 4-GPU runai training job (runs in the foreground)
# logs: logs/whp/<model>_eval_4gpu.log; logs/whp/done_eval_4gpu is written at the end.
cd "$(dirname "$0")/../../.."
export PATH=/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/miniconda3/envs/unlearning/bin:$PATH
export HF_HOME=/prj/corp/airesearch/lasvegas/vol20-scratch/gnguyenn/.cache/hf_cache  # TOFU is cached here; jobs don't source ~/.bashrc
export HF_DATASETS_OFFLINE=1
export EVAL_ACCEL=configs/accelerate/eval_1gpu.yaml METHODS=WHP TASK_SUFFIX=_alpha5 OVERWRITE=1
umask 0000
mkdir -p logs/whp

models=(llada llada1.5 dream)
for i in "${!models[@]}"; do
    m=${models[$i]}
    EVAL_PORT=$((29600 + 10 * i)) bash scripts/dlm/tofu/evaluate.sh "$m" "$i" forget10 methods \
        eval.tofu_dlm.overwrite=false >> "logs/whp/${m}_eval_4gpu.log" 2>&1 &
done
wait
echo "[$(date -u +%FT%TZ)] ALL_DONE" | tee logs/whp/done_eval_4gpu
