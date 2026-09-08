#!/usr/bin/env bash
# RepelGenDistill (masked-diffusion unlearning): repel forget positions off the gold
# secret + distil the whole answer toward a teacher-generated SECRET-FREE completion.
# Retain-FREE, forget-set only.
#
#   usage: bash scripts/dlm/tofu/repel_gendistill.sh <dream|llada> [gpus] [split]
#   e.g.:  bash scripts/dlm/tofu/repel_gendistill.sh llada 0,1,2,3 unlearn90
#
# PCSS forget-selection defaults are pinned here to the MDU ks_repel setting
# (select_method=topk, topk_frac=0.8, sel_mc=16): rank-based selection never leaves
# A_hat empty and stays stable when scores are uniformly high, unlike pure Otsu which
# left A_hat empty on ~6/10 forget10 samples. Override any of these via the env vars
# below; anything unlearn.sh reads (GPUS, TRAIN_BS, GRAD_ACCUM, UNLEARN_EPOCHS, ...)
# still applies.
#
#   env (method knobs, with their config defaults):
#     SELECT_METHOD=topk     TOPK_FRAC=0.8     SEL_MC=16      # PCSS A_hat selection
#     REPEL_ALPHA=0.5                                         # loss_f vs loss_d weight
#     DISTILL_MODE=sft       DISTILL_TOPK=64                  # secret-free distillation
#     GEN_STEPS=128          KEEP_ANSWER=False                # teacher generation
#     QUERY_BLANK_IDENTITY=True   KEEP_SUBJECT_NAME=False
#     SUPPRESS_GOLD=True     LOSS_D_AHAT_WEIGHT=1.0
#     GAMMA_ANC=1.0          SIGMA=median      SIGMA_ANC=1.0
#     LOSS_MC=1
#
# NOTE: RepelGenDistill reads answer-aligned logit/embedding spans, so right_shift_logits
# is unsupported -- use the `llada` config (Dream is not applicable to this method).
set -euo pipefail
bash "$(dirname "$0")/unlearn.sh" \
    "${1:-llada}" "${2:-}" "${3:-unlearn90}" DiffusionRepelGenDistill \
    trainer.method_args.select_method="${SELECT_METHOD:-topk}" \
    trainer.method_args.topk_frac="${TOPK_FRAC:-0.8}" \
    trainer.method_args.sel_mc="${SEL_MC:-16}" \
    trainer.method_args.repel_alpha="${REPEL_ALPHA:-0.0}" \
    trainer.method_args.distill_mode="${DISTILL_MODE:-sft}" \
    trainer.method_args.distill_topk="${DISTILL_TOPK:-64}" \
    trainer.method_args.gen_steps="${GEN_STEPS:-128}" \
    trainer.method_args.keep_answer="${KEEP_ANSWER:-True}" \
    trainer.method_args.query_blank_identity="${QUERY_BLANK_IDENTITY:-True}" \
    trainer.method_args.keep_subject_name="${KEEP_SUBJECT_NAME:-True}" \
    trainer.method_args.suppress_gold="${SUPPRESS_GOLD:-True}" \
    trainer.method_args.loss_d_ahat_weight="${LOSS_D_AHAT_WEIGHT:-1.0}" \
    trainer.method_args.gamma_anc="${GAMMA_ANC:-1.0}" \
    trainer.method_args.sigma="\"${SIGMA:-median}\"" \
    trainer.method_args.sigma_anc="${SIGMA_ANC:-1.0}" \
    trainer.method_args.loss_mc="${LOSS_MC:-1}"
