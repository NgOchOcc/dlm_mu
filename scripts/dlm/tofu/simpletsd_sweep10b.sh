#!/usr/bin/env bash
# SimpleTSD 10-arm tuning sweep (SET B) on TOFU -- explores axes NOT covered by
# simpletsd_sweep10.sh: A_hat thresholding (topk / tau vs Otsu), MC draw counts
# (loss_mc, sel_mc), the objective mix (alpha), and question-overlap masking.
# Runs arms SEQUENTIALLY; every arm gets its own TASK_SUFFIX -> its own save/eval
# dir (with _<N>epochs), so nothing clobbers -- including the SET A dirs.
#
#     bash scripts/dlm/tofu/simpletsd_sweep10b.sh
#
# Baseline (config defaults): mask_mode=cover, loss_mc=4, retain_frac=0.5,
# select_temp=1.0, otsu_relax=0.5, select_method=otsu, sel_mc=8, demotion=minimal,
# alpha=0.5, drop_question_overlap=true. Each arm departs along ONE axis.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

# hydra/omegaconf need antlr4-python3-runtime==4.9.*; lm_eval[math]/math-verify
# pull 4.11, which breaks `import hydra` (ATN deserialize). Self-heal before run.
if ! python -c "import hydra" >/dev/null 2>&1; then
    echo "[sweep10b] antlr4 mismatch -> pinning antlr4-python3-runtime==4.9.3" >&2
    pip install -q 'antlr4-python3-runtime==4.9.3'
fi

MODEL="${MODEL:-llada}"
GPUS="${GPUS:-0,1,2,3}"
SPLIT="${SPLIT:-forget10}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
BS="${BS:-4}"
ACCUM="${ACCUM:-4}"
LR="${LR:-1e-5}"

# "suffix | method_args overrides that DIFFER from the config default".
ARMS=(
  "_topk_frac050  | trainer.method_args.select_method=topk trainer.method_args.topk_frac=0.5"   # 1. fixed-fraction A_hat (top 50% PCSS)
  "_topk_frac025  | trainer.method_args.select_method=topk trainer.method_args.topk_frac=0.25"  # 2. tighter fixed fraction (top 25%)
  "_tau0          | trainer.method_args.select_method=tau trainer.method_args.tau=0.0"          # 3. threshold A_hat at PCSS>0 (prompt-dependent)
  "_loss_mc2      | trainer.method_args.loss_mc=2"                                               # 4. fewer cover draws / step
  "_loss_mc8      | trainer.method_args.loss_mc=8"                                               # 5. more cover draws (variance reduction)
  "_selmc4        | trainer.method_args.sel_mc=4"                                                # 6. noisier PCSS scoring (fewer partial masks)
  "_selmc16       | trainer.method_args.sel_mc=16"                                               # 7. smoother PCSS scoring (more partial masks)
  "_alpha025      | trainer.method_args.alpha=0.25"                                              # 8. lean toward self-distill (preserve)
  "_alpha075      | trainer.method_args.alpha=0.75"                                              # 9. lean toward forget
  "_noqoverlap    | trainer.method_args.drop_question_overlap=false"                             # 10. keep question-overlapping tokens in A_hat
)

echo "[sweep10b] start $(date) | model=$MODEL gpus=$GPUS split=$SPLIT epochs=$NUM_EPOCHS bs=$BS accum=$ACCUM lr=$LR arms=${#ARMS[@]}"
for arm in "${ARMS[@]}"; do
    suffix="${arm%%|*}";   suffix="$(echo "$suffix" | xargs)"
    overrides="${arm#*|}"; overrides="$(echo "$overrides" | xargs)"

    echo "==================================================================="
    echo "[sweep10b] arm=${suffix} $(date)"
    echo "[sweep10b]   overrides: ${overrides:-<baseline defaults>}"

    TASK_SUFFIX="$suffix" \
    UNLEARN_EPOCHS="$NUM_EPOCHS" TRAIN_BS="$BS" GRAD_ACCUM="$ACCUM" \
    EPOCH_IN_NAME="${EPOCH_IN_NAME:-1}" \
    UNLEARN_SAVE_STRATEGY="${UNLEARN_SAVE_STRATEGY:-epoch}" SKIP_DONE="${SKIP_DONE:-1}" \
        bash scripts/dlm/tofu/unlearn.sh "$MODEL" "$GPUS" "$SPLIT" SimpleTSD \
        trainer.args.learning_rate=$LR $overrides
    rc=$?
    if (( rc == 0 )); then
        echo "[sweep10b] arm=${suffix} OK $(date)"
    else
        echo "[sweep10b] arm=${suffix} FAILED rc=${rc} $(date) -- continuing" >&2
    fi
done

echo "[sweep10b] done $(date). Results table: python scripts/dlm/show_results.py"
