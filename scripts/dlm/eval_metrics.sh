#!/bin/bash
# Evaluate all unlearned DLM checkpoints in saves/unlearn on TOFU with the
# tofu_dlm suite: model_utility + forget_Truth_Ratio (truth_ratio), plus the
# per-split probability / rougeL that feed them. forget_quality is intentionally
# excluded (no retain-only reference model available).
set -e
cd "$(dirname "$0")/../.." || exit 1  # -> repo root

EVAL_GPU=0
forget_split=forget10

trainers=(
    "DiffusionGradAscent"
    "DiffusionGradDiff"
    "DiffusionNPO"
    "DiffusionSimNPO"
    "DiffusionWGA"
    "DiffusionAMTU"
)

# model : right_shift_logits (LLaDA=False, Dream=True)
run_model() {
    local MODEL=$1 RIGHT_SHIFT=$2
    for trainer in "${trainers[@]}"; do
        task_name=tofu_${MODEL}_${forget_split}_${trainer}
        save_dir=saves/unlearn/${task_name}
        echo "==================== ${task_name} (right_shift=${RIGHT_SHIFT}) ===================="
        CUDA_VISIBLE_DEVICES=${EVAL_GPU} python src/eval.py \
            model=${MODEL} \
            model.model_args.pretrained_model_name_or_path=${save_dir} \
            model.tokenizer_args.pretrained_model_name_or_path=${save_dir} \
            eval=tofu_dlm \
            eval.tofu_dlm.overwrite=true \
            eval.tofu_dlm.forget_split=${forget_split} \
            eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT} \
            task_name=${task_name}
    done
}

run_model "LLaDA-8B-Instruct" False
run_model "Dream-v0-Instruct-7B" True
