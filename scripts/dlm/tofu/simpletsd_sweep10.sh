set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

# hydra/omegaconf need antlr4-python3-runtime==4.9.*; lm_eval[math]/math-verify
# pull 4.11, which breaks `import hydra` (ATN deserialize). Self-heal before run.
if ! python -c "import hydra" >/dev/null 2>&1; then
    echo "[sweep10] antlr4 mismatch -> pinning antlr4-python3-runtime==4.9.3" >&2
    pip install -q 'antlr4-python3-runtime==4.9.3'
fi

MODEL="${MODEL:-llada}"
GPUS="${GPUS:-4,5,6,7}"
SPLIT="${SPLIT:-forget10}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
BS="${BS:-4}"
ACCUM="${ACCUM:-4}"
LR="${LR:-1e-5}"

# "suffix | method_args overrides that DIFFER from the config default".
# Empty override => the pure baseline arm (all config defaults).
ARMS=(
  "_cover_base                |"                                                                  # 1. baseline: cover q0.5 t1.0 relax0.5 mc4 minimal a0.5
  "_pathwise_mc1              | trainer.method_args.mask_mode=stratified trainer.method_args.loss_mc=4"  # 2. TSD-style pathwise, single draw
  "_cover_q025               | trainer.method_args.retain_frac=0.25"                              # 3. less self-distill, more forget mass
  "_cover_q075               | trainer.method_args.retain_frac=0.75"                              # 4. more self-distill coverage
  "_cover_temp03             | trainer.method_args.select_temp=0.3"                               # 5. sharp: focus high-PCSS forget tokens
  "_cover_temp30             | trainer.method_args.select_temp=3.0"                               # 6. near-uniform forget selection
  "_relax00_pureotsu         | trainer.method_args.otsu_relax=0.0"                                # 7. tight A_hat (pure Otsu)
  "_relax09_wide             | trainer.method_args.otsu_relax=0.9"                                # 8. very wide A_hat
  "_belief_top5              | trainer.method_args.demotion=belief trainer.method_args.belief_top_k=5"  # 9. broader belief demotion
  "_alpha10_nodistill        | trainer.method_args.alpha=1.0"                                     # 10. pure demotion, drop self-distill
)

echo "[sweep10] start $(date) | model=$MODEL gpus=$GPUS split=$SPLIT epochs=$NUM_EPOCHS bs=$BS accum=$ACCUM lr=$LR arms=${#ARMS[@]}"
for arm in "${ARMS[@]}"; do
    suffix="${arm%%|*}";   suffix="$(echo "$suffix" | xargs)"
    overrides="${arm#*|}"; overrides="$(echo "$overrides" | xargs)"

    echo "==================================================================="
    echo "[sweep10] arm=${suffix} $(date)"
    echo "[sweep10]   overrides: ${overrides:-<baseline defaults>}"

    TASK_SUFFIX="$suffix" \
    UNLEARN_EPOCHS="$NUM_EPOCHS" TRAIN_BS="$BS" GRAD_ACCUM="$ACCUM" \
    EPOCH_IN_NAME="${EPOCH_IN_NAME:-1}" \
    UNLEARN_SAVE_STRATEGY="${UNLEARN_SAVE_STRATEGY:-epoch}" SKIP_DONE="${SKIP_DONE:-1}" \
        bash scripts/dlm/tofu/unlearn.sh "$MODEL" "$GPUS" "$SPLIT" SimpleTSD \
        trainer.args.learning_rate=$LR $overrides
    rc=$?
    if (( rc == 0 )); then
        echo "[sweep10] arm=${suffix} OK $(date)"
    else
        echo "[sweep10] arm=${suffix} FAILED rc=${rc} $(date) -- continuing" >&2
    fi
done

echo "[sweep10] done $(date). Results table: python scripts/dlm/show_results.py"
