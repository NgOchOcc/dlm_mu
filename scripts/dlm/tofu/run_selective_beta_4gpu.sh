#!/usr/bin/env bash
# Selective unlearning, LLaDA-8B forget10, beta 10, preserve 0.1, with the DiRU target q* applied at every
# answer-text position except copied subjects (forget_all; the EOS pad and terminator keep p_cond), training
# timesteps t ~ Beta(3, 1) (t = 1e-3 + (1 - 1e-3) * Beta(3, 1); drop the time_beta line for U(1e-3, 1)).
# Trained and evaluated like the beta 1/3/10 runs on one 4-GPU node: effective batch 4 GPUs x 4 x 4 = 64,
# 20 epochs = 140 steps.
# outputs: saves/{unlearn,eval}/forget10/tofu_LLaDA-8B-Instruct_DiffusionSelectiveUnlearning_selective_beta10_preserve0.1_tbeta3-1_forgetall
#
# usage: as the command of a 4-GPU runai training job (runs in the foreground)
# log: logs/selective/beta10_tbeta3-1_forgetall_4gpu.log
cd "$(dirname "$0")/../../.."
export PATH=/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/miniconda3/envs/unlearning/bin:$PATH
export HF_HOME=/prj/corp/airesearch/lasvegas/vol20-scratch/gnguyenn/.cache/hf_cache  # TOFU is cached here; jobs don't source ~/.bashrc
export HF_DATASETS_OFFLINE=1
export UNLEARN_ACCEL=configs/accelerate/dlm_config_nooffload_4gpu.yaml UNLEARN_EPOCHS=20
export EXP=unlearn/tofu/dlm_selective_unlearning TASK_SUFFIX=_selective_beta=100_preserve5.0_tbeta3-1_forgetall
mkdir -p logs/selective

bash scripts/dlm/tofu/unlearn.sh llada 0,1,2,3 forget10 SelectiveUnlearning \
    trainer.method_args.beta=100 trainer.method_args.preserve_weight=5.0 \
    '+trainer.method_args.time_beta=[3,1]' \
    '+trainer.method_args.forget_all=true' \
    2>&1 | tee -a logs/selective/beta100_tbeta3-1_forgetall_4gpu.log
