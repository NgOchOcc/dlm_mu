#!/usr/bin/env bash
# Evaluate the two RWKU base models (Dream-v0-Instruct-7B, LLaDA-1.5) per target
# entity. RWKU has no SFT stage -- real-world entities are already in
# pretraining -- so these runs are the pre-unlearning reference: forget scores
# should be HIGH here and drop after unlearning, while neighbor and utility
# scores should stay put.
#
# Results land in eval/RWKU/{model}_base/{setting}/{NN_entity}/RWKU_SUMMARY.json
#
# Evaluation runs on ONE GPU (the first of GPUS): the evaluators do not shard
# across ranks. To use several GPUs, launch several copies with different GPUS
# and entity subsets.
#
#   GPUS=0 scripts/dlm/rwku/evaluate_base.sh                 # both models
#   GPUS=0 MODELS=llada1.5 scripts/dlm/rwku/evaluate_base.sh # one model
#   GPUS=0 NUM_SUBJECTS=3 scripts/dlm/rwku/evaluate_base.sh  # first 3 entities
#   GPUS=0 SETTING=train_v2 scripts/dlm/rwku/evaluate_base.sh
set -euo pipefail
# Run from the repository root. The job runner starts the script there and
# asks it not to change directory, so this only verifies the assumption instead
# of cd-ing: $0 is not a usable path when the script is sourced, piped in, or
# copied into a batch spool, and guessing from it silently landed elsewhere.
if [[ ! -f scripts/dlm/lib.sh ]]; then
    echo "error: run this from the repository root (no scripts/dlm/lib.sh in $PWD)." >&2
    exit 1
fi
source scripts/dlm/lib.sh
source scripts/dlm/rwku/lib_rwku.sh

SETTING="${SETTING:-train_v1}"
MODELS="${MODELS:-dream llada1.5}"

for model_key in $MODELS; do
    dlm_config "$model_key" "${GPUS}"
    require_gpus
    load_subjects
    ckpt=$(rwku_base_ckpt)
    echo "=== base eval: ${model_key} (${MODEL}) <- ${ckpt} ==="

    idx=0
    for subject in "${RWKU_SUBJECTS[@]}"; do
        idx=$((idx + 1))
        outdir=$(eval_dir "${model_key}_base" "$SETTING" "$idx" "$subject")
        if [[ -f "$outdir/RWKU_SUMMARY.json" ]] && ! flag_on "${OVERWRITE:-0}"; then
            echo "--- [$idx/${#RWKU_SUBJECTS[@]}] $subject: exists, skip (OVERWRITE=1 to redo)"
            continue
        fi
        echo "--- [$idx/${#RWKU_SUBJECTS[@]}] $subject -> gpu $(eval_gpu) -> $outdir"
        rwku_eval "$ckpt" "$outdir" "$subject"
    done
done

echo
echo "=== summary ==="
python3 scripts/dlm/rwku/aggregate.py --setting "$SETTING" || true
