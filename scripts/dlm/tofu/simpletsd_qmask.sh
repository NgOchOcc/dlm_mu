set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

if ! python -c "import hydra" >/dev/null 2>&1; then
    echo "[qmask] antlr4 mismatch -> pinning antlr4-python3-runtime==4.9.3" >&2
    pip install -q 'antlr4-python3-runtime==4.9.3'
fi

MODEL="${MODEL:-llada}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SPLIT="${SPLIT:-forget10}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
BS="${BS:-4}"
ACCUM="${ACCUM:-4}"
LR="${LR:-1e-5}"

FULL="trainer.method_args.select_method=tau trainer.method_args.tau=-1e9 trainer.method_args.drop_question_overlap=false"
ARMS=(
  "_qmask_cover_a05  | "                                                                                     # 1. defaults (alpha=0.5, cover, otsu)
  "_qmask_a10        | trainer.method_args.alpha=1.0"                                                        # 2. pure forget, no self-distill
  "_qmask_a10_full   | trainer.method_args.alpha=1.0 ${FULL}"                                                # 3. A_hat = whole answer
  "_qmask_a10_erase  | trainer.method_args.alpha=1.0 trainer.method_args.erase_gold_in_target=true"          # 4. also demote gold from promptless target
  "_qmask_full_mw2   | trainer.method_args.alpha=1.0 ${FULL} trainer.method_args.mask_weight_power=2.0"      # 5. full + high-mask emphasis (p=2)
  "_qmask_full_mw3   | trainer.method_args.alpha=1.0 ${FULL} trainer.method_args.mask_weight_power=3.0"      # 6. full + sharper high-mask emphasis (p=3)
)

echo "[qmask] start $(date) | model=$MODEL gpus=$GPUS split=$SPLIT epochs=$NUM_EPOCHS bs=$BS accum=$ACCUM lr=$LR arms=${#ARMS[@]}"
for arm in "${ARMS[@]}"; do
    suffix="${arm%%|*}";   suffix="$(echo "$suffix" | xargs)"
    overrides="${arm#*|}"; overrides="$(echo "$overrides" | xargs)"

    echo "==================================================================="
    echo "[qmask] arm=${suffix} $(date)"
    echo "[qmask]   overrides: ${overrides:-<baseline defaults>}"

    TASK_SUFFIX="$suffix" \
    UNLEARN_EPOCHS="$NUM_EPOCHS" TRAIN_BS="$BS" GRAD_ACCUM="$ACCUM" \
    EPOCH_IN_NAME="${EPOCH_IN_NAME:-1}" \
    UNLEARN_SAVE_STRATEGY="${UNLEARN_SAVE_STRATEGY:-epoch}" SKIP_DONE="${SKIP_DONE:-1}" \
        bash scripts/dlm/tofu/unlearn.sh "$MODEL" "$GPUS" "$SPLIT" SimpleTSDQMask \
        trainer.args.learning_rate=$LR $overrides
    rc=$?
    if (( rc == 0 )); then
        echo "[qmask] arm=${suffix} OK $(date)"
    else
        echo "[qmask] arm=${suffix} FAILED rc=${rc} $(date) -- continuing" >&2
    fi
done

echo "[qmask] done $(date). Results table: python scripts/dlm/show_results.py"
