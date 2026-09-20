#!/usr/bin/env bash
# Per-entity RWKU unlearning, matching the MDU reference setting: no SFT stage,
# forget-only (no retain term), lr 1e-5, per-device batch 4. SDU trains for 10
# epochs and every other method for 1 (SDU_EPOCHS / UNLEARN_EPOCHS). One
# training run per target entity -- RWKU's own evaluation filters every forget
# and neighbor probe by `subject`, so a checkpoint that forgot all entities at
# once could not be scored the way RWKU reports results.
#
# For each method, all entities are trained and evaluated before moving to the
# next method. Results land in
#   eval/RWKU/{model}_{Method}/{setting}/{NN_entity}/RWKU_SUMMARY.json
#
# Training and evaluation both run on ONE GPU, named by the second argument.
# To use several GPUs, launch several copies of this script -- one per device --
# each with its own entity subset (see SUBJECTS / NUM_SUBJECTS).
#
#   scripts/dlm/rwku/unlearn.sh llada1.5 0
#   scripts/dlm/rwku/unlearn.sh dream 2
#   METHODS=SDU,GA scripts/dlm/rwku/unlearn.sh llada1.5 0
#   NUM_SUBJECTS=3 scripts/dlm/rwku/unlearn.sh llada1.5 0
#   DELETE_CKPT=1 scripts/dlm/rwku/unlearn.sh llada1.5 0  # free ckpts after eval
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh
source scripts/dlm/rwku/lib_rwku.sh

MODEL_KEY="${1:-llada1.5}"
dlm_config "$MODEL_KEY" "${2:-$GPUS}"
pin_single_gpu
load_subjects

SETTING="${SETTING:-train_v1}"
# SDU needs more passes than the gradient-based methods: its loss only shapes
# the selected entity positions, so a single epoch barely moves the model.
SDU_EPOCHS="${SDU_EPOCHS:-10}"
EPOCHS="${UNLEARN_EPOCHS:-1}"

epochs_for() {
    case "$1" in
        DiffusionSDU) echo "$SDU_EPOCHS" ;;
        *)            echo "$EPOCHS" ;;
    esac
}
LR="${UNLEARN_LR:-1e-5}"
BS="${UNLEARN_BS:-4}"
ACCUM="${UNLEARN_ACCUM:-8}"
BASE_CKPT=$(rwku_base_ckpt)

IFS=',' read -ra RAW_METHODS <<< "${METHODS:-SDU,GA,WGA,SimNPO,NPO}"
METHOD_LIST=()
for m in "${RAW_METHODS[@]}"; do
    m="${m// /}"
    [[ -z "$m" ]] && continue
    [[ "$m" == Diffusion* ]] || m="Diffusion${m}"
    [[ -f "configs/trainer/${m}.yaml" ]] || die "no such trainer: configs/trainer/${m}.yaml"
    METHOD_LIST+=("$m")
done

# SDU reads its query mask from an offline Monte Carlo cache. The cache is keyed
# by dataset index, assigned AFTER the subject filter, so each entity needs its
# own cache built with that same filter.
sdu_cache_for() {
    local subject=$1 cache
    cache="saves/sdu_cache/rwku_${MODEL}_$(slug "$subject").json"
    if [[ -f "$cache" ]] && ! flag_on "${OVERWRITE_CACHE:-0}"; then
        echo "$cache"; return
    fi
    mkdir -p "$(dirname "$cache")"
    echo "--- precomputing SDU mask cache: $cache" >&2
    CUDA_VISIBLE_DEVICES=${GPUS} python scripts/dlm/sdu_precompute.py \
        --model-dir "$BASE_CKPT" \
        --dataset-path jinzhuoran/RWKU \
        --dataset-name train_pair_llama3 \
        --split train \
        --filter-column subject \
        --filter-value "$subject" \
        --question-key prompt \
        --answer-key response \
        --answer-index 1 \
        --k-percent-query "${SDU_K_PERCENT:-25}" \
        --entity-boost "${SDU_ENTITY_BOOST:-1.6}" \
        --out "$cache" >&2
    echo "$cache"
}

run_one() {
    local method=$1 subject=$2 task=$3
    local exp=unlearn/rwku/dlm
    local epochs; epochs=$(epochs_for "$method")
    local extra=()
    if [[ "$method" == "DiffusionSDU" ]]; then
        exp=unlearn/rwku/dlm_sdu
        local cache; cache=$(sdu_cache_for "$subject")
        extra+=("trainer.method_args.mask_cache_path=${cache}"
                "trainer.method_args.beta=${SDU_BETA:-10.0}"
                "trainer.method_args.preserve_weight=${SDU_PRESERVE_WEIGHT:-0.1}")
    fi

    assert_gpus
    pin_single_gpu
    local port; port=$(free_port "$(port_base train)")
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --config_file "${TRAIN_ACCEL_1GPU}" --num_processes 1 \
        --main_process_port "${port}" \
        src/train.py --config-name=unlearn.yaml \
        experiment="${exp}" \
        model=${MODEL} \
        trainer="${method}" \
        task_name="${task}" \
        data.forget.RWKU_forget_pair.args.hf_args.filter_value="\"${subject}\"" \
        trainer.args.num_train_epochs=${epochs} \
        trainer.args.learning_rate=${LR} \
        trainer.args.per_device_train_batch_size=${BS} \
        trainer.args.gradient_accumulation_steps=${ACCUM} \
        trainer.args.save_strategy=no \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=${GRAD_CKPT} \
        trainer.args.eval_strategy=no \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        eval.rwku_dlm.target_subject="\"${subject}\"" \
        trainer.method_args.right_shift_logits=${RIGHT_SHIFT} \
        trainer.method_args.loss_weight_type="\"${LOSS_WEIGHT}\"" \
        eval.rwku_dlm.right_shift_logits=${RIGHT_SHIFT} \
        ${extra[@]+"${extra[@]}"}
}

echo "=== RWKU per-entity unlearning ==="
echo "  model    : ${MODEL_KEY} (${MODEL}) <- ${BASE_CKPT}"
echo "  methods  : ${METHOD_LIST[*]}"
echo "  entities : ${#RWKU_SUBJECTS[@]}"
echo "  setting  : ${SETTING}  (lr ${LR}, bs ${BS})"
echo "  epochs   : ${EPOCHS} (${SDU_EPOCHS} for SDU)"
echo "  runs     : $(( ${#METHOD_LIST[@]} * ${#RWKU_SUBJECTS[@]} ))"
echo

# Each entity is trained and then evaluated before moving to the next, so only
# one checkpoint is being produced at a time. Note this is unrelated to the
# in-training evaluation that `eval_strategy=no` disables above: that one ran
# the whole RWKU suite after every epoch, unfiltered.
#
# EVAL=0 trains without evaluating; EVAL_ONLY=1 evaluates checkpoints already
# on disk without retraining.
for method in "${METHOD_LIST[@]}"; do
    echo "######## ${method} ########"
    idx=0
    for subject in "${RWKU_SUBJECTS[@]}"; do
        idx=$((idx + 1))
        outdir=$(eval_dir "${MODEL_KEY}_${method}" "$SETTING" "$idx" "$subject")
        task="rwku_${MODEL}_${method}_$(slug "$subject")"
        label="[$idx/${#RWKU_SUBJECTS[@]}] ${method} / ${subject}"

        if [[ -f "$outdir/RWKU_SUMMARY.json" ]] && ! flag_on "${OVERWRITE:-0}"; then
            echo "--- ${label}: evaluated already, skip"
            continue
        fi

        if flag_on "${EVAL_ONLY:-0}"; then
            if [[ ! -d "saves/unlearn/${task}" ]]; then
                echo "--- ${label}: no checkpoint, skip"
                continue
            fi
        elif [[ -d "saves/unlearn/${task}" ]] && ! flag_on "${OVERWRITE:-0}"; then
            echo "--- ${label}: checkpoint exists, skip training"
        else
            echo "--- ${label}: train"
            run_one "$method" "$subject" "$task"
        fi

        if flag_on "${EVAL:-1}"; then
            echo "--- ${label}: eval"
            rwku_eval "saves/unlearn/${task}" "$outdir" "$subject"
            # Checkpoints are kept by default. Every entity trains its own, so
            # they add up quickly (~15GB each); remove them yourself once you no
            # longer need to re-evaluate, e.g. rm -rf saves/unlearn/rwku_*.
            if flag_on "${DELETE_CKPT:-0}"; then
                rm -rf "saves/unlearn/${task}"
            fi
        fi
    done
    echo
done

# NO_SUMMARY=1 when several methods run in parallel: aggregate.py reports every
# method it finds, so a per-method call would print a table of everyone else's
# half-finished results. The launcher prints one summary once all are done.
if ! flag_on "${NO_SUMMARY:-0}"; then
    echo "=== summary ==="
    python3 scripts/dlm/rwku/aggregate.py --setting "$SETTING" || true
fi
