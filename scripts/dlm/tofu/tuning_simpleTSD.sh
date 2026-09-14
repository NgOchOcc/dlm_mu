  # ===== HYPERPARAMS =====
  MODEL=llada                 # llada | dream
  SPLIT=forget10              # forget01 | forget05 | forget10
  GPUS=0,1,2,3,4,5,6,7        # node này chỉ có 4 GPU
  NUM_EPOCHS=20
  BS=4                        # per-device batch size
  ACCUM=4                     # grad accum
  LR=1e-5

  # --- SimpleTSD cover-mode ---
  LOSS_MC=8                   # số draw Monte-Carlo / sample
  Q=0.75                       # retain_frac = q (retain share; 1-q vào forget)
  TEMP=1.0                    # select_temp (nhỏ -> sharp theo score; lớn -> uniform)
  OTSU_RELAX=0.5              # nới ngưỡng Otsu -> A_hat rộng hơn
  DEMOTION=minimal            # minimal | belief
  ALPHA=0.8                   # L = alpha*L_forget + (1-alpha)*L_distill
  # ===================================

  NAME="_cover_q${Q}_t${TEMP}_relax${OTSU_RELAX}_demotion${DEMOTION}_alpha${ALPHA}"   # save dir riêng cho arm

  TASK_SUFFIX="$NAME" \
  UNLEARN_EPOCHS="$NUM_EPOCHS" TRAIN_BS="$BS" GRAD_ACCUM="$ACCUM" \
  UNLEARN_SAVE_STRATEGY=epoch SKIP_DONE=1 \
  setsid bash scripts/dlm/tofu/unlearn.sh "$MODEL" "$GPUS" "$SPLIT" SimpleTSD \
      trainer.args.learning_rate=$LR \
      trainer.method_args.mask_mode=cover \
      trainer.method_args.loss_mc=$LOSS_MC \
      trainer.method_args.retain_frac=$Q \
      trainer.method_args.select_temp=$TEMP \
      trainer.method_args.otsu_relax=$OTSU_RELAX \
      trainer.method_args.demotion=$DEMOTION \
      trainer.method_args.alpha=$ALPHA