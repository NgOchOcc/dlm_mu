#!/usr/bin/env bash
# LLM-judge the WHP TOFU forget10 evals (Fluency/Relevance/Correctness, needed for
# EQ and LQ) with the judge used for the eval_v2 results: Qwen/Qwen3.8-27B on a
# local vLLM server (weights are downloaded into $HF_HOME on first use).
# Run after scripts/dlm/tofu/run_whp.sh has written the TOFU_EVAL.json files.
#
# usage: bash scripts/dlm/tofu/judge_whp.sh [GPU]      # one free 80 GB GPU
set -euo pipefail
cd "$(dirname "$0")/../../.."
ENVS=/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/miniconda3/envs
export PATH=$ENVS/unlearning/bin:$PATH
mkdir -p logs/judge

GPUS=${1:-0} VLLM_BIN=$ENVS/vllm-judge/bin/vllm bash scripts/dlm/serve_judge.sh > logs/judge/vllm_serve_8000.log 2>&1 &
trap 'bash scripts/dlm/serve_judge.sh stop' EXIT
bash scripts/dlm/serve_judge.sh wait

for m in LLaDA-8B-Instruct LLaDA-1.5 Dream-v0-Instruct-7B; do
    python scripts/dlm/evaluate_all_llm_judge.py --once --eval-root saves/eval/forget10/tofu_${m}_DiffusionWHP_alpha5
done
python scripts/dlm/show_results.py --eval-dir saves/eval --split forget10 --method WHP --detail
