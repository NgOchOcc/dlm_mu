set -e
cd "$(dirname "$0")/../../.." || exit 1  

MODEL="Dream-v0-Instruct-7B"
EVAL_GPU=2
RIGHT_SHIFT=True   
forget_split=forget10

CKPT="saves/finetune/tofu_${MODEL}_full_cont/checkpoint-4725"

if [ ! -e "${CKPT}" ]; then
    echo "ERROR: Dream TOFU-SFT checkpoint not found at ${CKPT}" >&2
    echo "Run scripts/dlm/dream/finetune.sh first, or set CKPT to an existing SFT checkpoint." >&2
    exit 1
fi

task_name=tofu_${MODEL}_base
echo "==================== ${task_name} (right_shift=${RIGHT_SHIFT}) ===================="
echo "checkpoint: ${CKPT}"

CUDA_VISIBLE_DEVICES=${EVAL_GPU} python src/eval.py \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${CKPT} \
    model.tokenizer_args.pretrained_model_name_or_path=${CKPT} \
    eval=tofu_dlm \
    eval.tofu_dlm.overwrite=true \
    eval.tofu_dlm.mc_num=128 \
    eval.tofu_dlm.forget_split=${forget_split} \
    eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT} \
    task_name=${task_name}
