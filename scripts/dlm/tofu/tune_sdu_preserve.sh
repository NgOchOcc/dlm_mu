#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.." || exit 1

setting=""
model=""
while (( $# > 0 )); do
    case "$1" in
        --setting)
            [[ $# -ge 2 ]] || { echo "error: --setting requires a value" >&2; exit 2; }
            setting=$2
            shift 2
            ;;
        --model)
            [[ $# -ge 2 ]] || { echo "error: --model requires a value" >&2; exit 2; }
            model=$2
            shift 2
            ;;
        *)
            echo "error: unknown argument '$1'" >&2
            exit 2
            ;;
    esac
done

case "$setting" in
    forget01|forget05|forget10) split=$setting ;;
    *)
        echo "error: --setting must be forget01, forget05, or forget10" >&2
        exit 2
        ;;
esac

case "$model" in
    llada|llada1.5|dream) ;;
    *)
        echo "error: --model must be llada, llada1.5, or dream" >&2
        exit 2
        ;;
esac

: "${MASK_CACHE:?MASK_CACHE must point to a scripts/dlm/sdu_precompute.py cache JSON}"

# beta held fixed at its DiffusionSDU baseline (see run_sdu_all.sh) while
# preserve_weight is swept across 5 settings.
IFS=' ' read -r -a preserves <<< "${PRESERVES:-0.01 0.05 0.1 0.5 1.0}"
beta="${BETA:-10.0}"
gpus="${GPUS:-0,1,2,3}"

fail=0
for preserve in "${preserves[@]}"; do
    echo "=== tuning preserve_weight=${preserve} (beta fixed at ${beta}) ==="
    command=(
        env
        MODEL="$model"
        GPUS="$gpus"
        SPLIT="$split"
        BETA="$beta"
        PRESERVE_WEIGHT="$preserve"
        MASK_CACHE="$MASK_CACHE"
        bash scripts/dlm/tofu/run_sdu.sh
    )

    if [[ "${DRY_RUN:-0}" != 0 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    elif ! "${command[@]}"; then
        echo "[failed] preserve_weight=${preserve}" >&2
        fail=1
    fi
done

exit "$fail"
