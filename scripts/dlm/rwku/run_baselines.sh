#!/usr/bin/env bash
# Train and evaluate masked-diffusion baselines on the full RWKU forget corpus.
#
# Usage:
#   bash scripts/dlm/rwku/run_baselines.sh \
#       <dream|llada|llada1.5> [gpus] [methods] [hydra overrides...]
#
# Examples:
#   bash scripts/dlm/rwku/run_baselines.sh llada1.5 0,1,2,3
#   bash scripts/dlm/rwku/run_baselines.sh dream 0,1 "GA NPO WGA"
#
# Environment:
#   UNLEARN_EPOCHS=1   Number of training epochs.
#   SKIP_DONE=1        Skip methods with an existing evaluation summary.
#   SKIP_TRAIN=1       Evaluate existing checkpoints without retraining.
#   NO_EVAL=1          Train without evaluation.
#   OVERWRITE=1        Retrain even when a checkpoint already exists.
#   DRY_RUN=1          Print commands without running them.
set -euo pipefail
umask 0000

cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

model_key="${1:-}"
gpus="${2:-}"
methods="${3:-GA NPO SimNPO WGA}"
if (( $# > 3 )); then shift 3; else shift $#; fi

dlm_config "$model_key" "$gpus"
parse_methods "$methods"

epochs="${UNLEARN_EPOCHS:-1}"
nproc=$(ngpus "$GPUS")
if ! flag_on "${DRY_RUN:-}"; then
    assert_gpus
fi

log_dir="logs/rwku/${model_key}"
mkdir -p "$log_dir"

run_command() {
    local log=$1
    shift
    if flag_on "${DRY_RUN:-}"; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@" 2>&1 | tee "$log"
    fi
}

rwku_eval() {
    local task=$1
    local eval_gpus="${EVAL_GPUS:-$GPUS}"
    local eval_nproc eval_port
    eval_nproc=$(ngpus "$eval_gpus")
    eval_port=$(free_port "${EVAL_PORT:-29600}")

    run_command "${log_dir}/eval_${task}.log" \
        env CUDA_VISIBLE_DEVICES="$eval_gpus" \
        accelerate launch \
        --config_file "$EVAL_ACCEL" \
        --num_processes "$eval_nproc" \
        --main_process_port "$eval_port" \
        src/eval.py \
        model="$MODEL" \
        model.model_args.pretrained_model_name_or_path="saves/unlearn/${task}" \
        model.tokenizer_args.pretrained_model_name_or_path="saves/unlearn/${task}" \
        eval=rwku_dlm \
        eval.rwku_dlm.overwrite=true \
        eval.rwku_dlm.right_shift_logits="$RIGHT_SHIFT" \
        task_name="$task" \
        "paths.output_dir=saves/eval/${task}"
}

fail=0
for method in "${METHODS_LIST[@]}"; do
    task="rwku_${MODEL}_${method}"
    checkpoint="saves/unlearn/${task}"
    summary="saves/eval/${task}/RWKU_SUMMARY.json"

    if flag_on "${SKIP_DONE:-}" && [[ -f "$summary" ]]; then
        echo "[skip] ${task} already evaluated" >&2
        continue
    fi

    echo "==================== RWKU ${task} (${epochs} epochs, gpus=${GPUS}) ====================" >&2
    if ! flag_on "${SKIP_TRAIN:-}"; then
        if [[ -f "${checkpoint}/config.json" ]] && ! flag_on "${OVERWRITE:-}"; then
            echo "[train] skip ${task}; checkpoint exists (OVERWRITE=1 to redo)" >&2
        else
            train_port=$(free_port "${MAIN_PORT:-29500}")
            if ! run_command "${log_dir}/train_${task}.log" \
                    env CUDA_VISIBLE_DEVICES="$GPUS" \
                    accelerate launch \
                    --config_file "$UNLEARN_ACCEL" \
                    --num_processes "$nproc" \
                    --main_process_port "$train_port" \
                    src/train.py --config-name=unlearn.yaml \
                    experiment=unlearn/rwku/dlm \
                    model="$MODEL" \
                    trainer="$method" \
                    task_name="$task" \
                    trainer.args.num_train_epochs="$epochs" \
                    trainer.args.gradient_accumulation_steps="$GRAD_ACCUM" \
                    trainer.args.save_strategy=no \
                    trainer.args.ddp_find_unused_parameters=true \
                    trainer.args.gradient_checkpointing="$GRAD_CKPT" \
                    trainer.method_args.right_shift_logits="$RIGHT_SHIFT" \
                    trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
                    eval.rwku_dlm.right_shift_logits="$RIGHT_SHIFT" \
                    "paths.output_dir=${checkpoint}" \
                    "$@"; then
                echo "[failed] training ${task}" >&2
                fail=1
                continue
            fi
        fi
    elif [[ ! -f "${checkpoint}/config.json" ]] && ! flag_on "${DRY_RUN:-}"; then
        echo "[failed] SKIP_TRAIN=1 but checkpoint is missing: ${checkpoint}" >&2
        fail=1
        continue
    fi

    if ! flag_on "${NO_EVAL:-}" && ! rwku_eval "$task"; then
        echo "[failed] evaluation ${task}" >&2
        fail=1
    fi
done

echo "[done] fail=${fail}; logs=${log_dir}" >&2
exit "$fail"
