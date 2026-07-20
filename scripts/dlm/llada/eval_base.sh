#!/bin/bash
# Evaluate the LLaDA-8B-Instruct "Base" (TOFU-finetuned target) model with the
# tofu_dlm suite: rougeL + probability on forget10 / retain90 / real_authors /
# world_facts.
#
# The "Base" row must be evaluated on the TOFU-SFT checkpoint, NOT the raw
# instruct model (which has never seen TOFU's fictitious authors). Point CKPT at
# your SFT checkpoint (e.g. saves/finetune/tofu_LLaDA-8B-Instruct_full from
# scripts/dlm/llada/finetune.sh).
set -e
cd "$(dirname "$0")/../../.." || exit 1  # -> repo root (relative paths below)

MODEL="LLaDA-8B-Instruct"
EVAL_GPU=2
RIGHT_SHIFT=False   # LLaDA: no logit shift
forget_split=forget10

# TOFU-SFT checkpoint (Base/target model).
CKPT="/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/working/open-unlearning/saves/finetune/tofu_LLaDA-8B-Instruct_full_cont/checkpoint-4725"

task_name=tofu_${MODEL}_base
echo "==================== ${task_name} (right_shift=${RIGHT_SHIFT}) ===================="
echo "checkpoint: ${CKPT}"

CUDA_VISIBLE_DEVICES=${EVAL_GPU} python src/eval.py \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${CKPT} \
    model.tokenizer_args.pretrained_model_name_or_path=${CKPT} \
    eval=tofu_dlm \
    eval.tofu_dlm.overwrite=true \
    eval.tofu_dlm.mc_num=128 \
    eval.tofu_dlm.forget_split=${forget_split} \
    eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT} \
    task_name=${task_name}
