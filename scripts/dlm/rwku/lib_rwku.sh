RWKU_SUBJECTS_FILE="${RWKU_SUBJECTS_FILE:-scripts/dlm/rwku/subjects.txt}"

# RWKU runs on a single GPU for both training and evaluation, so the whole
# pipeline is pinned to one device via GPUS.
EVAL_ACCEL_1GPU="${EVAL_ACCEL_1GPU:-configs/accelerate/eval_1gpu.yaml}"
TRAIN_ACCEL_1GPU="${TRAIN_ACCEL_1GPU:-configs/accelerate/dlm_1gpu.yaml}"

# Distinct port ranges per GPU.
#
# free_port only checks whether something is already listening, so parallel
# launches that all start probing at the same base pick the same port: with four
# processes starting together, every one of them sees 29600 free. Offsetting the
# base by device index keeps the ranges disjoint from the start.
#
# The stride must leave room for free_port's own probing, which steps by 10 --
# 200 gives each device 20 retries before it could reach the next device's base.
# Training and evaluation are interleaved within a device's block rather than
# given separate bases: with separate bases, enough devices would push the
# training range into the evaluation one (device 5's training base would land on
# device 0's evaluation base).
PORT_STRIDE="${PORT_STRIDE:-200}"

port_base() {
    local kind=$1 dev base off
    dev="${GPUS%%,*}"
    [[ "$dev" =~ ^[0-9]+$ ]] || dev=0
    base="${RWKU_PORT_BASE:-29000}"
    case "$kind" in
        train) off=0 ;;
        eval)  off=100 ;;
        *) die "port_base: unknown kind '$kind'" ;;
    esac
    echo $(( base + dev * PORT_STRIDE + off ))
}

# Pin the whole pipeline (training and evaluation) to one device.
#
# GPUS may arrive as a list -- either from the caller or from lib.sh's
# auto-detect -- so narrow it to the first entry and say so, rather than
# silently using a device the caller did not expect.
pin_single_gpu() {
    if [[ "$(ngpus "$GPUS")" -gt 1 ]]; then
        echo "note: RWKU runs on one GPU; using device ${GPUS%%,*} of '${GPUS}'" >&2
    fi
    GPUS="${GPUS%%,*}"
    [[ -n "$GPUS" ]] || die "no GPU available (GPUS is empty)"
}

load_subjects() {
    RWKU_SUBJECTS=()
    [[ -f "$RWKU_SUBJECTS_FILE" ]] || die "missing $RWKU_SUBJECTS_FILE"
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        RWKU_SUBJECTS+=("$line")
    done < "$RWKU_SUBJECTS_FILE"
    if [[ -n "${SUBJECTS:-}" ]]; then
        local sel=() s
        IFS='|' read -ra sel <<< "$SUBJECTS"
        RWKU_SUBJECTS=("${sel[@]}")
    fi
    local n="${NUM_SUBJECTS:-10}"
    (( ${#RWKU_SUBJECTS[@]} > n )) && RWKU_SUBJECTS=("${RWKU_SUBJECTS[@]:0:n}")
    (( ${#RWKU_SUBJECTS[@]} > 0 )) || die "no subjects selected"
}

# "Prince Harry, Duke of Sussex" -> "Prince_Harry_Duke_of_Sussex"
slug() { echo "$1" | tr -cs '[:alnum:]' '_' | sed 's/^_//; s/_$//'; }

# eval/{dataset}/{model_method}/{setting}/{NN_entity}
eval_dir() {
    local model_method=$1 setting=$2 idx=$3 subject=$4
    printf 'eval/RWKU/%s/%s/%d_%s' "$model_method" "$setting" "$idx" "$(slug "$subject")"
}

rwku_eval() {
    local ckpt=$1 outdir=$2 subject=$3
    mkdir -p "$outdir"
    local port; port=$(free_port "$(port_base eval)")
    # Single-GPU evaluation. The evaluators do not shard data across ranks --
    # every rank recomputes the whole metric and only rank 0 writes results
    # (evals/base.py:save_logs) -- so extra processes only duplicate the work
    # and hold extra model copies.
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${EVAL_ACCEL_1GPU}" --num_processes 1 \
        --main_process_port "${port}" \
        src/eval.py \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path="${ckpt}" \
        model.tokenizer_args.pretrained_model_name_or_path="${ckpt}" \
        eval=rwku_dlm \
        eval.rwku_dlm.overwrite=true \
        eval.rwku_dlm.right_shift_logits=${RIGHT_SHIFT} \
        eval.rwku_dlm.batch_size=${EVAL_BS:-8} \
        eval.rwku_dlm.mmlu_batch_size=${MMLU_BS:-$(( ${EVAL_BS:-8} < 4 ? ${EVAL_BS:-8} : 4 ))} \
        eval.rwku_dlm.target_subject="\"${subject}\"" \
        paths.output_dir="${outdir}" \
        task_name="$(basename "$outdir")"
}

rwku_base_ckpt() {
    case "$MODEL" in
        Dream-v0-Instruct-7B) echo "${DREAM_BASE:-Dream-org/Dream-v0-Instruct-7B}" ;;
        LLaDA-1.5)            echo "${LLADA15_BASE:-GSAI-ML/LLaDA-1.5}" ;;
        LLaDA-8B-Instruct)    echo "${LLADA_BASE:-GSAI-ML/LLaDA-8B-Instruct}" ;;
        *) die "no RWKU base checkpoint known for MODEL='$MODEL'" ;;
    esac
}
