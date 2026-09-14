#!/usr/bin/env bash
# DIAGNOSTIC: force DiffusionSimpleTSD to reduce EXACTLY to DiffusionTSD, to test
# the hypothesis that SimpleTSD's failure comes from its two departures from TSD
# (a PCSS-selected subset A_hat + self-distillation on the rest of the answer),
# NOT from a porting bug. If this arm matches a real DiffusionTSD run, the method
# design -- not the code -- is what breaks unlearning on TOFU.
#
# The reduction (see src/trainer/dlm/unlearn/{simple_tsd,tsd}.py):
#   select_method=tau, tau=-1e9   -> _threshold_scores returns the whole answer:
#                                    A_hat = A, so R = A \ A_hat = empty.
#   drop_question_overlap=false   -> keep question-echo tokens in A_hat (TSD does).
#   alpha=1.0                     -> pure forget; and with R empty the self-distill
#                                    term never fires anyway (loss = L_f).
#   mask_mode=stratified,loss_mc=1-> _sample_mask == TSD pathwise, one draw/step.
#   demotion=minimal              -> demote only the gold token (TSD default).
# => per-position: demote gold + renorm at EVERY masked answer token; nothing
#    anchors the model back to the answer. That is DiffusionTSD.
#
#     bash scripts/dlm/tofu/simpletsd_tsd_equiv.sh
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

# hydra/omegaconf need antlr4-python3-runtime==4.9.*; lm_eval[math]/math-verify
# pull 4.11, which breaks `import hydra` (ATN deserialize). Self-heal before run.
if ! python -c "import hydra" >/dev/null 2>&1; then
    echo "[tsdequiv] antlr4 mismatch -> pinning antlr4-python3-runtime==4.9.3" >&2
    pip install -q 'antlr4-python3-runtime==4.9.3'
fi

MODEL="${MODEL:-llada}"
GPUS="${GPUS:-0,1,2,3}"
SPLIT="${SPLIT:-forget10}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
BS="${BS:-4}"
ACCUM="${ACCUM:-4}"
LR="${LR:-1e-5}"

SUFFIX="${TASK_SUFFIX:-_tsdequiv_fullahat_a1}"

echo "[tsdequiv] model=$MODEL gpus=$GPUS split=$SPLIT epochs=$NUM_EPOCHS bs=$BS accum=$ACCUM lr=$LR suffix=$SUFFIX"

TASK_SUFFIX="$SUFFIX" \
UNLEARN_EPOCHS="$NUM_EPOCHS" TRAIN_BS="$BS" GRAD_ACCUM="$ACCUM" \
EPOCH_IN_NAME="${EPOCH_IN_NAME:-1}" \
UNLEARN_SAVE_STRATEGY="${UNLEARN_SAVE_STRATEGY:-epoch}" SKIP_DONE="${SKIP_DONE:-1}" \
    bash scripts/dlm/tofu/unlearn.sh "$MODEL" "$GPUS" "$SPLIT" SimpleTSD \
    trainer.args.learning_rate=$LR \
    trainer.method_args.select_method=tau \
    trainer.method_args.tau=-1e9 \
    trainer.method_args.drop_question_overlap=false \
    trainer.method_args.alpha=1.0 \
    trainer.method_args.mask_mode=stratified \
    trainer.method_args.loss_mc=1 \
    trainer.method_args.demotion=minimal
rc=$?
if (( rc == 0 )); then
    echo "[tsdequiv] OK -- compare saves/eval/${SPLIT}/tofu_${MODEL}_*SimpleTSD${SUFFIX}_${NUM_EPOCHS}epochs against a DiffusionTSD run"
else
    echo "[tsdequiv] FAILED rc=${rc}" >&2
fi
