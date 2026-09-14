#!/usr/bin/env bash
# Run SimpleTSD in the "pathwise" masking regime -- the SimpleTSD analogue of
# DiffusionTSD's mask_states=pathwise.
#
#   mask_mode=stratified  ==  TSD pathwise:  each draw masks k ~ Uniform{1..|A|}
#                             answer positions (forces >=1 A_hat position), i.e. a
#                             random point along the whole answer-masking trajectory
#                             -- NOT the cover-mode timestep-rho + q-split scheme.
#   loss_mc=1             ==  one mask draw per step, exactly like TSD (no multi-draw
#                             averaging). Raise it for variance reduction while staying
#                             pathwise/stratified (each draw is still k~U{1..|A|}).
#
# Retain-free (forget set only). Own save dir per arm via TASK_SUFFIX so runs never
# clobber each other; setsid so the run survives this shell exiting.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

# ===================== HYPERPARAMS (edit here) =====================
MODEL="${MODEL:-llada}"                 # llada | dream
SPLIT="${SPLIT:-forget10}"              # forget01 | forget05 | forget10
GPUS="${GPUS:-0,1,2,3}"                 # comma-separated device ids
NUM_EPOCHS="${NUM_EPOCHS:-9}"
BS="${BS:-4}"                           # per-device batch size
ACCUM="${ACCUM:-4}"                     # gradient accumulation
LR="${LR:-1e-5}"

# --- SimpleTSD pathwise (= stratified) selection/objective knobs ---
LOSS_MC="${LOSS_MC:-1}"                 # 1 = TSD-style single draw/step
SELECT_METHOD="${SELECT_METHOD:-otsu}"  # otsu | topk | tau  (A_hat thresholding)
OTSU_RELAX="${OTSU_RELAX:-0.5}"         # loosen Otsu cut -> wider A_hat
SEL_MC="${SEL_MC:-8}"                   # partial-mask draws when scoring PCSS s_i
DEMOTION="${DEMOTION:-minimal}"         # minimal | belief
BELIEF_TOP_K="${BELIEF_TOP_K:-5}"       # only used when demotion=belief
ALPHA="${ALPHA:-0.5}"                   # L = alpha*L_forget + (1-alpha)*L_distill
# ===================================================================

NAME="${TASK_SUFFIX:-_pathwise_${SELECT_METHOD}_relax${OTSU_RELAX}_mc${LOSS_MC}_${DEMOTION}}"
LOG_DIR="${LOG_DIR:-logs/simpletsd_pathwise}"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${NAME#_}.log"

echo "[pathwise] model=$MODEL split=$SPLIT gpus=$GPUS epochs=$NUM_EPOCHS bs=$BS accum=$ACCUM lr=$LR"
echo "[pathwise] mask_mode=stratified loss_mc=$LOSS_MC select=$SELECT_METHOD otsu_relax=$OTSU_RELAX sel_mc=$SEL_MC demotion=$DEMOTION alpha=$ALPHA"
echo "[pathwise] task_suffix=$NAME  log=$LOG"

TASK_SUFFIX="$NAME" \
UNLEARN_EPOCHS="$NUM_EPOCHS" TRAIN_BS="$BS" GRAD_ACCUM="$ACCUM" \
UNLEARN_SAVE_STRATEGY="${UNLEARN_SAVE_STRATEGY:-epoch}" SKIP_DONE="${SKIP_DONE:-1}" \
setsid bash scripts/dlm/tofu/unlearn.sh "$MODEL" "$GPUS" "$SPLIT" SimpleTSD \
    trainer.args.learning_rate=$LR \
    trainer.method_args.mask_mode=stratified \
    trainer.method_args.loss_mc=$LOSS_MC \
    trainer.method_args.select_method=$SELECT_METHOD \
    trainer.method_args.otsu_relax=$OTSU_RELAX \
    trainer.method_args.sel_mc=$SEL_MC \
    trainer.method_args.demotion=$DEMOTION \
    trainer.method_args.belief_top_k=$BELIEF_TOP_K \
    trainer.method_args.alpha=$ALPHA \
    > "$LOG" 2>&1 &

echo "[pathwise] launched pid=$!  (tail -f $LOG)"
