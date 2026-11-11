export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1

# $HOME is NFS-mounted on this cluster; concurrent multi-process launches
# hammering the Triton/Inductor autotune cache there causes DeepSpeed imports
# to stall or hang on exit. Redirect to local (non-NFS) disk unless the caller
# already set these.
# $USER is not exported in every batch environment (Slurm sets SLURM_JOB_USER
# instead), and callers run with `set -u`, so referencing it bare aborts the
# script here -- before any log file exists, which looks like the job completing
# instantly with no output.
_CACHE_USER="${USER:-${SLURM_JOB_USER:-${LOGNAME:-$(id -un 2>/dev/null || echo user)}}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${_CACHE_USER}/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/${_CACHE_USER}/inductor_cache}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true

die() { echo "error: $*" >&2; exit 1; }
flag_on() {
    case "${1:-}" in
        ''|0|false|False|FALSE|no|No|off|Off) return 1 ;;
        *) return 0 ;;
    esac
}

detect_gpus() {
    local n
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        # CUDA renumbers the visible devices to 0..N-1 inside the process, so
        # the physical ids in CUDA_VISIBLE_DEVICES are not usable as indices:
        # under a Slurm allocation of GPUs 2,3,6,7 the device to address as 0 is
        # physical 2. Count them and emit logical ids instead.
        n=$(awk -F',' '{c=0; for (i=1; i<=NF; i++) if ($i ~ /[0-9]/) c++; print c}' \
            <<< "$CUDA_VISIBLE_DEVICES")
    else
        n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    fi
    (( n > 0 )) || { echo 0; return; }
    # BSD seq leaves a trailing separator, which would end up inside
    # CUDA_VISIBLE_DEVICES; build the list directly instead.
    local i out=""
    for ((i = 0; i < n; i++)); do
        out+="${out:+,}$i"
    done
    echo "$out"
}
GPUS="${GPUS:-$(detect_gpus)}"
ngpus() { awk -F',' '{n=0; for (i=1; i<=NF; i++) if ($i ~ /[0-9]/) n++; print n}' <<< "$1"; }

assert_gpus() {
    local want have idx
    want=$(ngpus "$GPUS")
    (( want > 0 )) || die "GPUS is empty (got '$GPUS')"
    have=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    (( have > 0 )) || return 0
    for idx in ${GPUS//,/ }; do
        (( idx < have )) || die "GPUS='$GPUS' asks for device $idx but only $have GPU(s) present"
    done
}

free_port() {
    local port=$1
    while python -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',$port))==0 else 1)" 2>/dev/null; do
        port=$((port + 10))
    done
    echo "$port"
}

if [[ "${OFFLOAD:-1}" == 1 ]]; then
    _ACCEL_DEFAULT=configs/accelerate/dlm_config.yaml
else
    _ACCEL_DEFAULT=configs/accelerate/dlm_config_nooffload.yaml
fi
TRAIN_ACCEL="${TRAIN_ACCEL:-$_ACCEL_DEFAULT}"
UNLEARN_ACCEL="${UNLEARN_ACCEL:-$_ACCEL_DEFAULT}"
EVAL_ACCEL="${EVAL_ACCEL:-configs/accelerate/eval_4gpu.yaml}"
TRAIN_BS="${TRAIN_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EXP_OVERRIDE="${EXP:-}"

dlm_config() {
    case "${1:-}" in
        dream)
            MODEL="Dream-v0-Instruct-7B"
            RIGHT_SHIFT=True
            LOSS_WEIGHT='cart[geo_p:0.3]'
            GRAD_CKPT=true
            FINETUNE_EPOCHS=150
            SFT_DIR="saves/finetune/tofu_${MODEL}_retain100_150_epochs"
            ;;
        llada)
            MODEL="LLaDA-8B-Instruct"
            RIGHT_SHIFT=False
            LOSS_WEIGHT=scheduler
            GRAD_CKPT=false
            FINETUNE_EPOCHS=150
            SFT_DIR="saves/finetune/tofu_${MODEL}_retain100_150epochs"
            ;;
        llada1.5)
            MODEL="LLaDA-1.5"
            RIGHT_SHIFT=False
            LOSS_WEIGHT=scheduler
            GRAD_CKPT=false
            FINETUNE_EPOCHS=150
            SFT_DIR="saves/finetune/tofu_${MODEL}_retain100_150epochs"
            ;;
        *)
            die "usage: <script> <dream|llada|llada1.5> [gpus] [...]"
            ;;
    esac
    RETAIN90_DIR="saves/finetune/tofu_${MODEL}_retain90_150epochs"
    RETAIN95_DIR="saves/finetune/tofu_${MODEL}_retain95_150epochs"
    RETAIN99_DIR="saves/finetune/tofu_${MODEL}_retain99_150epochs"

    SFT_CKPT=$(resolve_ckpt "$SFT_DIR")
    GPUS="${2:-$GPUS}"
    GRAD_CKPT="${GRAD_CKPT_OVERRIDE:-$GRAD_CKPT}"
    LOSS_WEIGHT="${LOSS_WEIGHT_OVERRIDE:-$LOSS_WEIGHT}"
}

resolve_ckpt() {
    local dir=$1 latest
    [[ -f "$dir/config.json" ]] && { echo "$dir"; return; }
    latest=$(find "$dir" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null \
        | sort -t- -k2 -n | tail -1)
    echo "${latest:-$dir}"
}

rundir_for() {
    local var
    case "$1" in
        retain100)                  var=SFT_DIR ;;
        retain90|retain95|retain99) var="RETAIN${1#retain}_DIR" ;;
        *) die "unknown reference split '$1' (want retain100|retain90|retain95|retain99)" ;;
    esac
    echo "${!var}"
}

ckpt_for() { resolve_ckpt "$(rundir_for "$1")"; }
tofu_split() {
    case "${1:-}" in
        unlearn90|forget10) FORGET=forget10; RETAIN=retain90; HOLDOUT=holdout10; SPLIT_EPOCHS=5 ;;
        unlearn95|forget05) FORGET=forget05; RETAIN=retain95; HOLDOUT=holdout05; SPLIT_EPOCHS=10 ;;
        unlearn99|forget01) FORGET=forget01; RETAIN=retain99; HOLDOUT=holdout01; SPLIT_EPOCHS=20 ;;
        *) die "unknown split '${1:-}' (want unlearn90|unlearn95|unlearn99)" ;;
    esac
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-$SPLIT_EPOCHS}"
}

DEFAULT_METHODS=(
    DiffusionGA
    DiffusionNPO
    DiffusionSimNPO
    DiffusionWGA
    DiffusionBST
    DiffusionGD
    DiffusionMDU
    DiffusionSDU
    DiffusionSelectiveUnlearning
)

parse_methods() {
    local raw="${1:-}" m
    METHODS_LIST=()
    if [[ -z "$raw" ]]; then
        METHODS_LIST=("${DEFAULT_METHODS[@]}")
        return
    fi
    for m in ${raw//,/ }; do
        [[ "$m" == Diffusion* ]] || m="Diffusion${m}"
        [[ -f "configs/trainer/${m}.yaml" ]] || die "no such trainer: configs/trainer/${m}.yaml"
        METHODS_LIST+=("$m")
    done
}

method_setup() {
    local method=$1
    METHOD_ARGS=()
    # RETAIN_ALREADY_NULL=1 marks experiment configs that already set
    # `data.retain` to null in their own defaults (forget-only by design) -
    # for these, unlearn.sh's automatic `~data.retain` delete override would
    # hit a key that's already absent and Hydra errors ("does not exist").
    RETAIN_ALREADY_NULL=0
    case "$method" in
        DiffusionGD)
            [[ -n "${GRADDIFF_LR:-}" ]] && METHOD_ARGS=(trainer.args.learning_rate="${GRADDIFF_LR}")
            EXP=unlearn/tofu/dlm
            ;;
        DiffusionMDU)       EXP=unlearn/tofu/dlm_mdu ;;
        DiffusionBST)       EXP=unlearn/tofu/dlm_bst ; RETAIN_ALREADY_NULL=1 ;;
        DiffusionSDU)       EXP=unlearn/tofu/dlm_sdu ; RETAIN_ALREADY_NULL=1 ;;
        DiffusionSelectiveUnlearning) EXP=unlearn/tofu/dlm_selective_unlearning ; RETAIN_ALREADY_NULL=1 ;;
        DiffusionWHP)       EXP=unlearn/tofu/dlm_whp ; RETAIN_ALREADY_NULL=1 ;;
        *)                  EXP=unlearn/tofu/dlm ;;
    esac
    [[ -n "$EXP_OVERRIDE" ]] && EXP="$EXP_OVERRIDE"
    return 0
}

tofu_train() {
    local task=$1 epochs=$2; shift 2
    assert_gpus
    local nproc port
    nproc=$(ngpus "$GPUS")
    port=$(free_port "${MAIN_PORT:-29600}")
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${TRAIN_ACCEL}" --num_processes "${nproc}" --main_process_port "${port}" \
        src/train.py \
        experiment=${TRAIN_EXP:-finetune/tofu/dlm} \
        model=${MODEL} \
        task_name=${task} \
        trainer.args.num_train_epochs=${epochs} \
        trainer.args.per_device_train_batch_size=${TRAIN_BS} \
        trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
        trainer.args.save_strategy=${SAVE_STRATEGY:-epoch} \
        +trainer.args.save_total_limit=${SAVE_TOTAL_LIMIT:-2} \
        +trainer.args.save_steps=${SAVE_STEPS:-500} \
        trainer.args.save_only_model=${SAVE_ONLY_MODEL:-false} \
        resume_from_checkpoint=${RESUME:-null} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=${GRAD_CKPT} \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT} \
        "$@"
}

tofu_unlearn() {
    local trainer=$1 task=$2 forget=$3 retain=$4 holdout=$5 epochs=$6; shift 6
    assert_gpus
    local nproc port
    nproc=$(ngpus "$GPUS")
    port=$(free_port "${MAIN_PORT:-29600}")
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${UNLEARN_ACCEL}" --num_processes "${nproc}" --main_process_port "${port}" \
        src/train.py --config-name=unlearn.yaml \
        experiment=${EXP:-unlearn/tofu/dlm} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${START_MODEL:-$SFT_CKPT} \
        trainer=${trainer} \
        task_name=${task} \
        forget_split=${forget} \
        retain_split=${retain} \
        holdout_split=${holdout} \
        trainer.args.num_train_epochs=${epochs} \
        trainer.args.per_device_train_batch_size=${TRAIN_BS} \
        trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
        trainer.args.save_strategy=${UNLEARN_SAVE_STRATEGY:-no} \
        +trainer.args.save_total_limit=${UNLEARN_SAVE_TOTAL_LIMIT:-2} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=${GRAD_CKPT} \
        trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT} \
        "$@"
}

tofu_eval_args() {
    local ckpt=$1 task=$2 forget=$3
    EVAL_ARGS=(
        model=${MODEL}
        model.model_args.pretrained_model_name_or_path="${ckpt}"
        model.tokenizer_args.pretrained_model_name_or_path="${ckpt}"
        eval=tofu_dlm
        eval.tofu_dlm.overwrite=true
        eval.tofu_dlm.forget_split="${forget}"
        eval.tofu_dlm.right_shift_logits=${RIGHT_SHIFT}
        task_name="${task}"
    )
}

tofu_eval() {
    local ckpt=$1 task=$2 forget=$3; shift 3
    local gpus="${EVAL_GPUS:-$GPUS}" nproc port
    nproc=$(ngpus "$gpus")
    (( nproc > 0 )) || die "no GPU for eval (EVAL_GPUS='${EVAL_GPUS:-}' GPUS='$GPUS')"
    port=$(free_port "${EVAL_PORT:-29600}")
    tofu_eval_args "$ckpt" "$task" "$forget"
    CUDA_VISIBLE_DEVICES=${gpus} accelerate launch \
        --config_file "${EVAL_ACCEL}" --num_processes "${nproc}" --main_process_port "${port}" \
        src/eval.py "${EVAL_ARGS[@]}" "$@"
}

tofu_eval_solo() {
    local gpu=$1 ckpt=$2 task=$3 forget=$4; shift 4
    tofu_eval_args "$ckpt" "$task" "$forget"
    CUDA_VISIBLE_DEVICES=${gpu} python src/eval.py "${EVAL_ARGS[@]}" "$@"
}

eval_plan_reset() { PLAN_TASKS=(); PLAN_CKPTS=(); PLAN_FORGETS=(); }
eval_plan_add() {
    local task=$1 dir=$2 forget=$3
    if ! flag_on "${OVERWRITE:-}" && [[ -f "saves/eval/${task}/TOFU_SUMMARY.json" ]]; then
        echo "skip  ${task} (already evaluated; OVERWRITE=1 to redo)"
        return 0
    fi
    if [[ ! -e "$dir" ]]; then
        echo "miss  ${task} (no checkpoint at ${dir})" >&2
        return 0
    fi
    PLAN_TASKS+=("$task")
    PLAN_CKPTS+=("$(resolve_ckpt "$dir")")
    PLAN_FORGETS+=("$forget")
}

eval_plan_run() {
    (( ${#PLAN_TASKS[@]} )) || { echo "nothing to evaluate"; return 0; }
    local fail=0 i gpu
    if flag_on "${PARALLEL:-}"; then
        local gpu_list pids=()
        IFS=',' read -r -a gpu_list <<< "${EVAL_GPUS:-$GPUS}"
        (( ${#gpu_list[@]} )) || die "no GPU for eval (EVAL_GPUS='${EVAL_GPUS:-}' GPUS='$GPUS')"
        for i in "${!PLAN_TASKS[@]}"; do
            gpu="${gpu_list[$(( i % ${#gpu_list[@]} ))]}"
            tofu_eval_solo "$gpu" "${PLAN_CKPTS[$i]}" "${PLAN_TASKS[$i]}" \
                "${PLAN_FORGETS[$i]}" "$@" &
            pids+=($!)
            sleep 3
        done
        for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    else
        for i in "${!PLAN_TASKS[@]}"; do
            tofu_eval "${PLAN_CKPTS[$i]}" "${PLAN_TASKS[$i]}" "${PLAN_FORGETS[$i]}" "$@" || fail=1
        done
    fi
    return $fail
}
