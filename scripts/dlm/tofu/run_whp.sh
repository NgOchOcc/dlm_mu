#!/usr/bin/env bash
# WHP baseline (Eldan & Russinovich 2023, arXiv:2310.02238) on TOFU forget10 for
# a masked-diffusion target model:
#   1. reinforced model: the target fine-tuned 3 more epochs on forget10
#   2. generic labels:   Eq. (1) + anchored-term translations + consistency rules
#   3. unlearning:       fine-tune the target on forget10 towards the generic labels
#   4. TOFU evaluation   (inside unlearn.sh)
# usage: bash scripts/dlm/tofu/run_whp.sh <dream|llada|llada1.5> [gpus]
# env: SFT_ROOT (repo whose saves/finetune holds the targets), ALPHA (5), UNLEARN_EPOCHS (20)
set -euo pipefail
umask 0000
cd "$(dirname "$0")/../../.."
source scripts/dlm/lib.sh
dlm_config "$1" "${2:-}"

SFT_ROOT="${SFT_ROOT:-/prj/corp/airesearch/lasvegas/vol20-scratch/nluu/working/open-unlearning}"
ALPHA="${ALPHA:-5}"
SFT_CKPT=$(resolve_ckpt "${SFT_ROOT}/${SFT_DIR}")
REINF_TASK="tofu_${MODEL}_forget10_reinforced"
LABELS="saves/whp/tofu_${MODEL}_forget10_alpha${ALPHA}.json"

[[ -f "saves/finetune/${REINF_TASK}/config.json" ]] || TRAIN_EXP=finetune/tofu/dlm_reinforce SAVE_STRATEGY=no \
    tofu_train "${REINF_TASK}" 3 \
    model.model_args.pretrained_model_name_or_path="${SFT_CKPT}" \
    model.tokenizer_args.pretrained_model_name_or_path="${SFT_CKPT}"

shift_flag=()
[[ "${RIGHT_SHIFT}" == True ]] && shift_flag=(--right-shift)
[[ -f "${LABELS}" ]] || CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python scripts/dlm/whp_generic_labels.py \
    --baseline "${SFT_CKPT}" --reinforced "saves/finetune/${REINF_TASK}" \
    --alpha "${ALPHA}" "${shift_flag[@]}" --out "${LABELS}"

START_MODEL="${SFT_CKPT}" TASK_SUFFIX="_alpha${ALPHA}" UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-20}" \
    bash scripts/dlm/tofu/unlearn.sh "$1" "${GPUS}" forget10 WHP \
    trainer.method_args.generic_labels_path="${LABELS}"
