#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
dlm_config "${1:-}" "${2:-}"

FORGET="${3:-forget10}"
IFS=',' read -r -a GPU_LIST <<< "${EVAL_GPUS:-$GPUS}"
NGPU=${#GPU_LIST[@]}
mkdir -p logs/eval

run_one() {
    local gpu=$1 task=$2 ckpt=$3
    CUDA_VISIBLE_DEVICES=${gpu} python src/eval.py \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path="${ckpt}" \
        model.tokenizer_args.pretrained_model_name_or_path="${ckpt}" \
        eval=tofu_dlm \
        eval.tofu_dlm.overwrite=true \
        eval.tofu_dlm.forget_split=${FORGET} \
        eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT} \
        task_name="${task}" > "logs/eval/${task//\//_}.log" 2>&1
}

pids=()
i=0
for trainer in "${TOFU_TRAINERS[@]}"; do
    task="${FORGET}/tofu_${MODEL}_${trainer}"
    ckpt="saves/unlearn/${task}"
    if [[ ! -e "$ckpt" ]]; then
        echo "miss ${task} (checkpoint not found)" >&2
        i=$((i + 1)); continue
    fi
    gpu=${GPU_LIST[$((i % NGPU))]}
    echo "[gpu ${gpu}] ${task}"
    run_one "$gpu" "$task" "$ckpt" &
    pids+=($!)
    sleep 3
    i=$((i + 1))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "eval_all done (fail=${fail})"
python scripts/dlm/show_results.py || true
