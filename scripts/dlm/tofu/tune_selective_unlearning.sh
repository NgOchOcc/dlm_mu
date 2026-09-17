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
    forge01|forget01) split=forget01 ;;
    forget05|forget10) split=$setting ;;
    *)
        echo "error: --setting must be forge01, forget01, forget05, or forget10" >&2
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

IFS=' ' read -r -a betas <<< "${BETAS:-1 2 3 4 5 6 7 8 9 10}"
IFS=' ' read -r -a preserves <<< "${PRESERVES:-0.1 1.0}"

gpus="${GPUS:-0,1,2,3}"

fail=0
for beta in "${betas[@]}"; do
    for preserve in "${preserves[@]}"; do
        suffix="_selective_beta${beta}_preserve${preserve}"
        command=(
            env
            EXP=unlearn/tofu/dlm_selective_unlearning
            TASK_SUFFIX="$suffix"
            bash scripts/dlm/tofu/unlearn.sh
            "$model" "$gpus" "$split" SelectiveUnlearning
            "trainer.method_args.beta=${beta}"
            "trainer.method_args.preserve_weight=${preserve}"
        )

        if [[ "${DRY_RUN:-0}" != 0 ]]; then
            printf '%q ' "${command[@]}"
            printf '\n'
        elif ! "${command[@]}"; then
            echo "[failed] beta=${beta} preserve_weight=${preserve}" >&2
            fail=1
        fi
    done
done

exit "$fail"
