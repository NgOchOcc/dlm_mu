#!/usr/bin/env bash
# Sweep SDU's query-mask size K on TOFU forget05 with LLaDA-1.5.
#
# K selects the top-K% most answer-leaking query word groups for the
# counterfactual mask. Unlike beta and preserve_weight it is consumed offline by
# sdu_precompute.py, so each value needs its own mask cache -- the sweep
# rebuilds one per K rather than sharing a single cache.
#
# Note that topk_word_indices() rounds to at least one word, so K=0 and K=10
# both mask a single group on short queries; K=100 masks every group, after
# which drop_stopwords still removes function words.
#
# Results land in
#   saves/eval/forget05/tofu_LLaDA-1.5_DiffusionSDU_topk<K>/TOFU_SUMMARY.json
#
#   scripts/dlm/tofu/tune_sdu_topk.sh
#   scripts/dlm/tofu/tune_sdu_topk.sh 0,1,2,3
#   TOPKS="20 25 30" scripts/dlm/tofu/tune_sdu_topk.sh 0,1,2,3
#   MODEL_KIND=dream SETTING=forget01 scripts/dlm/tofu/tune_sdu_topk.sh
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

export LOSS_WEIGHT_OVERRIDE="${LOSS_WEIGHT_OVERRIDE:-scheduler}"
source scripts/dlm/lib.sh

gpus="${1:-${GPUS:-0,1,2,3}}"
MODEL_KIND="${MODEL_KIND:-llada1.5}"
SETTING="${SETTING:-forget05}"

IFS=' ' read -r -a topks <<< "${TOPKS:-0 10 20 30 40 50 60 70 80 90 100}"
beta="${SDU_BETA:-10.0}"
preserve="${SDU_PRESERVE_WEIGHT:-0.1}"
mc_trials="${SDU_MC_TRIALS:-32}"

dlm_config "$MODEL_KIND" "$gpus"
tofu_split "$SETTING"
mkdir -p saves/sdu_cache

echo "=== TOFU SDU top-K sweep ==="
echo "  model   : ${MODEL_KIND} (${MODEL})"
echo "  split   : ${FORGET} (${SPLIT_EPOCHS} epochs)"
echo "  gpus    : ${gpus}"
echo "  K values: ${topks[*]}"
echo "  beta    : ${beta}   preserve_weight: ${preserve}"
echo "  runs    : ${#topks[@]}"
echo

fail=0
index=0
for k in "${topks[@]}"; do
    index=$((index + 1))
    task="${FORGET}/tofu_${MODEL}_DiffusionSDU_topk${k}"
    cache="saves/sdu_cache/${MODEL_KIND}_${FORGET}_topk${k}.json"

    echo "--- [${index}/${#topks[@]}] K=${k}%"

    if [[ -f "saves/eval/$task/TOFU_SUMMARY.json" ]] && ! flag_on "${OVERWRITE:-}"; then
        echo "    skip (already evaluated; OVERWRITE=1 to redo)"
        continue
    fi

    # The cache depends only on K, so it survives a rerun of the training step.
    if [[ -f "$cache" ]] && ! flag_on "${OVERWRITE_CACHE:-}"; then
        echo "    cache exists: ${cache}"
    else
        echo "    precomputing ${cache}"
        if ! CUDA_VISIBLE_DEVICES="${gpus%%,*}" \
            python scripts/dlm/sdu_precompute.py \
                --model-dir "$SFT_CKPT" \
                --dataset-name "$FORGET" \
                --num-mc-trials "$mc_trials" \
                --k-percent-query "$k" \
                --max-length 256 \
                --out "$cache"
        then
            echo "    FAILED: precompute" >&2
            fail=1
            continue
        fi
    fi

    method_setup DiffusionSDU
    if ! tofu_unlearn \
        DiffusionSDU \
        "$task" \
        "$FORGET" \
        "$RETAIN" \
        "$HOLDOUT" \
        "$SPLIT_EPOCHS" \
        ${METHOD_ARGS[@]+"${METHOD_ARGS[@]}"} \
        "trainer.method_args.beta=${beta}" \
        "trainer.method_args.preserve_weight=${preserve}" \
        "trainer.method_args.mask_cache_path=$cache" \
        "paths.output_dir=saves/unlearn/$task"
    then
        echo "    FAILED: unlearn" >&2
        fail=1
        continue
    fi

    if ! flag_on "${NO_EVAL:-}"; then
        tofu_eval \
            "saves/unlearn/$task" \
            "$task" \
            "$FORGET" \
            "paths.output_dir=saves/eval/$task" \
            || { echo "    FAILED: eval" >&2; fail=1; }
    fi
done

echo
echo "=== summary ==="
python3 scripts/dlm/show_results.py --split "$SETTING" --method SDU || true

exit "$fail"
