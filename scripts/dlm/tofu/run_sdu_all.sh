set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export LOSS_WEIGHT_OVERRIDE="${LOSS_WEIGHT_OVERRIDE:-scheduler}"
source scripts/dlm/lib.sh

gpus="${1:-0,1,2,3}"
if (( $# > 0 )); then
    shift
fi

IFS=' ' read -r -a models <<< "${MODELS:-dream llada llada1.5}"
IFS=' ' read -r -a splits <<< "${SPLITS:-forget01 forget05 forget10}"

CACHE_DIR="${SDU_CACHE_DIR:-saves/sdu_cache}"
mkdir -p "$CACHE_DIR" logs/sdu

BETA="${BETA:-10.0}"
PRESERVE_WEIGHT="${PRESERVE_WEIGHT:-0.1}"

precompute_cache() {
    local model_key=$1 split=$2 cache=$3
    if [[ -f "$cache" ]] && ! flag_on "${FORCE_PRECOMPUTE:-}"; then
        echo "[precompute] skip ${model_key}/${split}; cache exists at ${cache}" >&2
        return 0
    fi
    if flag_on "${SKIP_PRECOMPUTE:-}"; then
        die "SKIP_PRECOMPUTE=1 but no cache at ${cache} for ${model_key}/${split}"
    fi
    local first_gpu="${gpus%%,*}"
    echo "[precompute] ${model_key}/${split} base=${SFT_CKPT} -> ${cache}" >&2
    CUDA_VISIBLE_DEVICES="${first_gpu}" python scripts/dlm/sdu_precompute.py \
        --model-dir "${SFT_CKPT}" \
        --split "${split}" \
        --num-mc-trials "${NUM_MC_TRIALS:-32}" \
        --k-percent-query "${K_PERCENT_QUERY:-30}" \
        --max-length "${PRECOMPUTE_MAX_LENGTH:-256}" \
        --out "${cache}" \
        2>&1 | tee "logs/sdu/precompute_${model_key}_${split}.log"
}

fail=0
for model_key in "${models[@]}"; do
    dlm_config "$model_key" "$gpus"
    for split in "${splits[@]}"; do
        tofu_split "$split"
        epochs="${SDU_EPOCHS:-$SPLIT_EPOCHS}"
        cache="${CACHE_DIR}/${model_key}_${FORGET}.json"
        task="${FORGET}/tofu_${MODEL}_DiffusionSDU_beta${BETA}_preserve${PRESERVE_WEIGHT}"

        echo "==================== ${model_key} / ${FORGET} (${epochs} epochs) ====================" >&2

        if flag_on "${SKIP_DONE:-}" && [[ -f "saves/eval/${task}/TOFU_SUMMARY.json" ]]; then
            echo "[skip] ${task} already evaluated" >&2
            continue
        fi

        if ! precompute_cache "$model_key" "$split" "$cache"; then
            echo "[failed] precompute ${model_key}/${split}" >&2
            fail=1
            continue
        fi

        method_setup DiffusionSDU
        echo "[train] ${task}" >&2
        if ! tofu_unlearn DiffusionSDU "$task" "$FORGET" "$RETAIN" "$HOLDOUT" "$epochs" \
                "${METHOD_ARGS[@]}" \
                trainer.method_args.beta="${BETA}" \
                trainer.method_args.preserve_weight="${PRESERVE_WEIGHT}" \
                trainer.method_args.mask_cache_path="${cache}" \
                "paths.output_dir=saves/unlearn/${task}" \
                "$@" \
                2>&1 | tee "logs/sdu/train_${model_key}_${FORGET}.log"; then
            echo "[failed] training ${task}" >&2
            fail=1
            continue
        fi

        if ! flag_on "${NO_EVAL:-}"; then
            if ! tofu_eval "saves/unlearn/${task}" "$task" "$FORGET" \
                    "paths.output_dir=saves/eval/${task}" \
                    2>&1 | tee "logs/sdu/eval_${model_key}_${FORGET}.log"; then
                echo "[failed] evaluation ${task}" >&2
                fail=1
            fi
        fi
    done
done

echo "[done] fail=${fail}; results: python scripts/dlm/show_results.py" >&2
python scripts/dlm/show_results.py || true
exit "$fail"
