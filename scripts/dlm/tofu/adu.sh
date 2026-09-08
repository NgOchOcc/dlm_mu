#!/usr/bin/env bash
# DiffusionADU: selective anti-conditional KL plus secret-free generative
# distillation. Retain-free and trained only on the TOFU forget split.
#
# Usage: bash scripts/dlm/tofu/adu.sh llada [gpus] [splits]
# Default splits: forget10,forget05,forget01 with 10,20,40 epochs.
#
# Method environment variables:
#   AUTO_WEIGHT=0.5        LAM=0.5
#   SELECT_METHOD=topk     TOPK_FRAC=0.8     SEL_MC=16
#   DISTILL_MODE=sft       DISTILL_TOPK=64
#   GEN_STEPS=128          KEEP_ANSWER=False
#   QUERY_BLANK_IDENTITY=True   KEEP_SUBJECT_NAME=False
#   SUPPRESS_GOLD=True     LOSS_D_AHAT_WEIGHT=1.0
#   LOSS_MC=1              GAMMA=1.0
#
# Other variables supported by unlearn.sh remain available, including GPUS,
# TRAIN_BS, GRAD_ACCUM, UNLEARN_EPOCHS, TASK_SUFFIX, SKIP_DONE, and NO_EVAL.
# DiffusionADU uses answer-aligned logits; use LLaDA rather than Dream.
set -euo pipefail

splits="${3:-forget10,forget05,forget01}"
epochs="${EPOCHS:-}"
if [[ -z "$epochs" ]]; then
    for split in ${splits//,/ }; do
        case "$split" in
            forget10|unlearn90) count=10 ;;
            forget05|unlearn95) count=20 ;;
            forget01|unlearn99) count=40 ;;
            *) echo "Unknown split: $split" >&2; exit 1 ;;
        esac
        epochs="${epochs:+$epochs,}$count"
    done
fi

EPOCHS="$epochs" \
bash "$(dirname "$0")/unlearn.sh" \
    "${1:-llada}" "${2:-}" "$splits" DiffusionADU \
    trainer.method_args.auto_weight="${AUTO_WEIGHT:-1.0}" \
    trainer.method_args.lam="${LAM:-0.5}" \
    trainer.method_args.select_method="${SELECT_METHOD:-topk}" \
    trainer.method_args.topk_frac="${TOPK_FRAC:-0.8}" \
    trainer.method_args.sel_mc="${SEL_MC:-16}" \
    trainer.method_args.distill_mode="${DISTILL_MODE:-sft}" \
    trainer.method_args.distill_topk="${DISTILL_TOPK:-64}" \
    trainer.method_args.gen_steps="${GEN_STEPS:-128}" \
    trainer.method_args.keep_answer="${KEEP_ANSWER:-True}" \
    trainer.method_args.query_blank_identity="${QUERY_BLANK_IDENTITY:-True}" \
    trainer.method_args.keep_subject_name="${KEEP_SUBJECT_NAME:-True}" \
    trainer.method_args.suppress_gold="${SUPPRESS_GOLD:-True}" \
    trainer.method_args.loss_d_ahat_weight="${LOSS_D_AHAT_WEIGHT:-1.0}" \
    trainer.method_args.loss_mc="${LOSS_MC:-1}" \
    trainer.method_args.gamma="${GAMMA:-1.0}"
