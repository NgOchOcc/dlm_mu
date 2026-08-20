# DLM unlearning — run guide
#
# Cách gọi chung:  bash <script> <dream|llada> [gpus] [...]
#   $1 = model (dream | llada)
#   $2 = gpus  (vd "0,1,2,3"); không truyền thì lấy env GPUS, mặc định = tất cả GPU thấy được
# Override bằng env (tương đương $2):  GPUS=6,7 bash <script> dream
# GPU riêng cho bước eval:              EVAL_GPUS=7 bash <script> dream 0,1,2,3
#
# TOFU chỉ còn 4 script (chi tiết đọc header từng file):
#   tofu/retain.sh          <model> [gpus] [retain100|retain90|retain95|retain99]
#   tofu/unlearn.sh         <model> [gpus] [unlearn90|unlearn95|unlearn99] [methods] [hydra...]
#   tofu/evaluate.sh        <model> [gpus] [split] [base|retain|methods|all] [hydra...]
#   tofu/evaluate_retain.sh <model> [gpus] [retain100,retain90,retain95,retain99] [hydra...]
# Script cũ (unlearn90.sh, retrain*.sh, eval_*.sh, ...) nằm ở tofu/archive/ — xem
# tofu/archive/README.md để biết cái nào thay bằng cái nào.

########################################
# TOFU — full pipeline
########################################

# 1. Base model / retain oracle (150 epoch, chạy nhiều ngày → nên setsid)
bash scripts/dlm/tofu/retain.sh dream 0,1,2,3            # retain100 = Base (target)
bash scripts/dlm/tofu/retain.sh dream 0,1,2,3 retain90   # oracle cho forget10
bash scripts/dlm/tofu/retain.sh dream 0,1,2,3 retain95   # oracle cho forget05
bash scripts/dlm/tofu/retain.sh dream 0,1,2,3 retain99   # oracle cho forget01
setsid env RESUME=auto bash scripts/dlm/tofu/retain.sh dream 0,1,2,3,4,5,6,7 \
    > logs/retain100_dream.log 2>&1 &                    # chạy detached + tự resume

# 2. Unlearn (train + eval từng method) theo split
bash scripts/dlm/tofu/unlearn.sh dream 0,1,2,3 unlearn90            # forget10, 5 baseline
bash scripts/dlm/tofu/unlearn.sh dream 0,1,2,3 unlearn95            # forget05
bash scripts/dlm/tofu/unlearn.sh dream 0,1,2,3 unlearn99            # forget01
bash scripts/dlm/tofu/unlearn.sh llada 0,1 unlearn90 NPO,WGA        # chỉ 2 method
bash scripts/dlm/tofu/unlearn.sh llada 0,1 unlearn90 MDU trainer.method_args.tau=0.5
SKIP_DONE=1 bash scripts/dlm/tofu/unlearn.sh dream 0,1 unlearn90    # bỏ arm đã eval
NO_EVAL=1 bash scripts/dlm/tofu/unlearn.sh dream 0,1 unlearn90      # train, chưa eval
UNLEARN_EPOCHS=5 bash scripts/dlm/tofu/unlearn.sh dream 0,1 unlearn90
OFFLOAD=0 bash scripts/dlm/tofu/unlearn.sh dream 0,1,2,3 unlearn90  # ~5x nhanh, cần >=4 GPU
setsid bash scripts/dlm/tofu/unlearn.sh dream 0,1,2,3 unlearn90 \
    > logs/unlearn90_dream.log 2>&1 &

#   Sweep hyperparam (wrapper mỏng gọi unlearn.sh, tự thêm suffix vào task name)
TAUS="0 0.5 1.0" bash scripts/dlm/tofu/mdu.sh dream 0,1
BETAS="0 0.5 2" bash scripts/dlm/tofu/traj_rmdu.sh dream 0,1 unlearn90
ARMS="knowledge kappa0.5" bash scripts/dlm/tofu/traj_iproj.sh dream 0,1

# 3. Eval (unlearn.sh/retain.sh đã eval cái nó train; script này để vá chỗ thiếu)
bash scripts/dlm/tofu/evaluate.sh dream 0,1,2,3                      # base+retain+methods
bash scripts/dlm/tofu/evaluate.sh dream 3 unlearn95 base             # chỉ base, 1 GPU
METHODS=NPO,WGA bash scripts/dlm/tofu/evaluate.sh llada 0,1 unlearn90 methods
PARALLEL=1 bash scripts/dlm/tofu/evaluate.sh dream 0,1,2,3 unlearn90 methods  # 1 ckpt/GPU
OVERWRITE=1 bash scripts/dlm/tofu/evaluate.sh dream 0,1 unlearn90    # ép eval lại

# 3b. Eval trọn bộ model tham chiếu (Base + 3 oracle) đúng setting của từng cái:
#     retain100 -> forget10 + forget05 + forget01 (nó là target của cả 3 bảng)
#     retain90 -> forget10, retain95 -> forget05, retain99 -> forget01
#     => tối đa 6 lần eval; cùng 1 ckpt chỉ khác forget_split và task_name.
bash scripts/dlm/tofu/evaluate_retain.sh dream 0,1,2,3               # đủ 6 eval
bash scripts/dlm/tofu/evaluate_retain.sh llada 0,1 retain100         # chỉ Base, 3 setting
PARALLEL=1 bash scripts/dlm/tofu/evaluate_retain.sh dream 0,1,2,3    # 1 ckpt/GPU
OVERWRITE=1 bash scripts/dlm/tofu/evaluate_retain.sh dream 3 retain90

# 4. Fluency (chấm hậu kỳ, ghi vào summary cho show_results)
#   skip dir đã chấm; OVERWRITE=1 hoặc truyền dir cụ thể để ép lại
bash scripts/dlm/run_fluency.sh                             # GPU = GPUS/FLU_GPU, mặc định 0
GPUS=5 bash scripts/dlm/run_fluency.sh
OVERWRITE=1 bash scripts/dlm/run_fluency.sh

# 5. LLM judge (chấm hậu kỳ qua API — không cần GPU)
#   Chấm lại generation đã lưu trong TOFU_EVAL.json, ghi vào summary cho show_results.
#   Mỗi (model, split, method) 1 process, log ở logs/judge/.
bash scripts/dlm/run_judge.sh                                   # cả 2 model x 3 split
bash scripts/dlm/run_judge.sh dream                             # chỉ Dream
SPLITS=forget10 bash scripts/dlm/run_judge.sh dream             # Dream, chỉ forget10
JUDGE_PARALLEL=2 bash scripts/dlm/run_judge.sh                  # 2 target song song
#   Chạy lẻ 1 model + 1 split:  <model> [split] [base|retain|refs|methods|all]
bash scripts/dlm/tofu/judge.sh dream unlearn90                  # base + retain + methods
bash scripts/dlm/tofu/judge.sh dream unlearn90 methods          # chỉ method
bash scripts/dlm/tofu/judge.sh dream unlearn90 refs             # chỉ base + retain oracle
METHODS=NPO bash scripts/dlm/tofu/judge.sh dream unlearn90      # đúng 1 method
JUDGE_WORKERS=16 bash scripts/dlm/tofu/judge.sh dream unlearn90 # tăng thread API (default 8)
#   Chạy trực tiếp python: truyền dir = luôn chấm lại; không truyền = chỉ dir chưa chấm
python scripts/dlm/run_judge.py                                 # điền chỗ còn thiếu
OVERWRITE=1 python scripts/dlm/run_judge.py                     # chấm lại tất cả
python scripts/dlm/run_judge.py saves/eval/forget10/tofu_Dream-v0-Instruct-7B_DiffusionNPO

# 6. Tổng hợp bảng
python scripts/dlm/show_results.py
python scripts/dlm/aggregate_results.py

########################################
# MUSE — SFT + 3 scenario (News/Books), có eval bên trong
########################################
bash scripts/dlm/muse/run.sh dream
SFT_EPOCHS=5 UNLEARN_EPOCHS=5 bash scripts/dlm/muse/run.sh dream 0,1,2,3

########################################
# RWKU — forget-only GradAscent + eval
########################################
bash scripts/dlm/rwku/run.sh dream
UNLEARN_EPOCHS=3 bash scripts/dlm/rwku/run.sh llada 0,1

########################################
# Ghi chú quan trọng
########################################
# * unlearn.sh ép loss_weight_type=scheduler (1/t ELBO) cho CẢ 2 backbone. Dream
#   mặc định trong config là cart[geo_p:0.3] (~0.16x ELBO) làm retain collapse và
#   brake của NPO/SimNPO không ăn. Muốn chạy lại đúng số cũ của Dream (cart):
#     LOSS_WEIGHT_OVERRIDE='cart[geo_p:0.3]' bash scripts/dlm/tofu/unlearn.sh dream 0,1 unlearn90
#   retain.sh KHÔNG ép — finetune vẫn dùng weight mặc định của từng backbone.
# * Batch toàn cục = ngpu * TRAIN_BS * GRAD_ACCUM (mặc định 4*4), tức là tăng theo
#   số GPU. Muốn giữ batch cố định khi tăng GPU thì giảm GRAD_ACCUM tương ứng:
#     GRAD_ACCUM=2 bash scripts/dlm/tofu/unlearn.sh dream 0,1,2,3 unlearn90
# * Chạy dài phải dùng setsid, không thì shell chết là mất run (save_strategy=no
#   ⇒ không cứu được gì).
