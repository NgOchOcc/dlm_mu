#!/usr/bin/env bash
# Tune SelectiveUnlearning over beta and preserve_weight for one TOFU setting.
#
# Usage:
#   bash scripts/dlm/tofu/tune_selective_unlearning.sh \
#       --setting forget05 --model llada1.5
#
# Required flags:
#   --setting  forge01 | forget01 | forget05 | forget10
#   --model    llada | llada1.5 | dream
#
# Environment:
#   BETAS="1 2 3 4 5 6 7 8 9 10" Space-separated beta values.
#   PRESERVES="0.1 1.0"           Space-separated preserve_weight values.
#   GPUS=0,1,2,3                  Devices used by each sequential run.
#   DRY_RUN=1                     Print commands without running them.
#   SKIP_DONE=1                   Skip completed train/eval runs.
#   NO_EVAL=1                     Train without evaluation.
set -euo pipefail

cd "$(dirname "$0")/../../.." || exit 1

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

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
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument '$1'" >&2
            usage >&2
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
(( ${#betas[@]} > 0 )) || { echo "error: BETAS must not be empty" >&2; exit 2; }
(( ${#preserves[@]} > 0 )) || { echo "error: PRESERVES must not be empty" >&2; exit 2; }

gpus="${GPUS:-0,1,2,3}"
log_dir="logs/selective_tuning/${model}/${split}"
mkdir -p "$log_dir"

echo "[sweep] model=${model} setting=${split} gpus=${gpus}" >&2
echo "[sweep] betas=${betas[*]} preserves=${preserves[*]}" >&2

fail=0
for beta in "${betas[@]}"; do
    for preserve in "${preserves[@]}"; do
        suffix="_selective_beta${beta}_preserve${preserve}"
        log="${log_dir}/beta${beta}_preserve${preserve}.log"
        command=(
            env
            EXP=unlearn/tofu/dlm_selective_unlearning
            TASK_SUFFIX="$suffix"
            bash scripts/dlm/tofu/unlearn.sh
            "$model" "$gpus" "$split" SelectiveUnlearning
            "trainer.method_args.beta=${beta}"
            "trainer.method_args.preserve_weight=${preserve}"
        )

        echo "[run] beta=${beta} preserve_weight=${preserve} log=${log}" >&2
        if [[ "${DRY_RUN:-0}" != 0 ]]; then
            printf '%q ' "${command[@]}"
            printf '\n'
        elif ! "${command[@]}" 2>&1 | tee "$log"; then
            echo "[failed] beta=${beta} preserve_weight=${preserve}" >&2
            fail=1
        fi
    done
done

echo "[done] fail=${fail}; logs=${log_dir}" >&2
exit "$fail"
