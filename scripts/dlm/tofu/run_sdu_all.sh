#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

export LOSS_WEIGHT_OVERRIDE=scheduler
source scripts/dlm/lib.sh

gpus="${1:-0,1,2,3}"
(( $# == 0 )) || shift

mkdir -p saves/sdu_cache

precompute_cache() {
    local split=$1 cache=$2
    CUDA_VISIBLE_DEVICES="${gpus%%,*}" \
        python scripts/dlm/sdu_precompute.py \
            --model-dir "$SFT_CKPT" \
            --split "$split" \
            --num-mc-trials 16 \
            --k-percent-query 15 \
            --max-length 256 \
            --out "$cache"
}

for model in dream llada llada1.5; do
    dlm_config "$model" "$gpus"

    for split in forget01 forget05 forget10; do
        tofu_split "$split"

        task="$FORGET/tofu_${MODEL}_DiffusionSDU_beta10.0_preserve0.1"
        cache="saves/sdu_cache/${model}_${FORGET}.json"

        if flag_on "${SKIP_DONE:-}" && [[ -f "saves/eval/$task/TOFU_SUMMARY.json" ]]; then
            echo "[skip] $task already evaluated"
            continue
        fi

        precompute_cache "$split" "$cache"
        method_setup DiffusionSDU

        tofu_unlearn \
            DiffusionSDU \
            "$task" \
            "$FORGET" \
            "$RETAIN" \
            "$HOLDOUT" \
            "$SPLIT_EPOCHS" \
            "${METHOD_ARGS[@]}" \
            trainer.method_args.beta=10.0 \
            trainer.method_args.preserve_weight=0.1 \
            "trainer.method_args.mask_cache_path=$cache" \
            "paths.output_dir=saves/unlearn/$task" \
            "$@"

        flag_on "${NO_EVAL:-}" || tofu_eval \
            "saves/unlearn/$task" \
            "$task" \
            "$FORGET" \
            "paths.output_dir=saves/eval/$task"
    done
done

python scripts/dlm/show_results.py || true
