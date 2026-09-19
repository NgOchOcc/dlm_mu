RWKU_SUBJECTS_FILE="${RWKU_SUBJECTS_FILE:-scripts/dlm/rwku/subjects.txt}"

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
    local port; port=$(free_port "${EVAL_PORT_BASE:-29800}")
    # The evaluators do not shard data across ranks: every rank recomputes the
    # whole metric and only rank 0 writes results (evals/base.py:save_logs), so
    # extra processes just duplicate the work and hold extra model copies.
    # Default to the first visible GPU; set EVAL_GPUS to override.
    local egpus="${EVAL_GPUS:-${GPUS%%,*}}"
    local accel="${EVAL_ACCEL_OVERRIDE:-configs/accelerate/eval_1gpu.yaml}"
    if [[ -z "${EVAL_ACCEL_OVERRIDE:-}" && "$(ngpus "$egpus")" -gt 1 ]]; then
        accel="${EVAL_ACCEL}"
    fi
    CUDA_VISIBLE_DEVICES=${egpus} accelerate launch \
        --config_file "${accel}" --num_processes "$(ngpus "$egpus")" \
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
