
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SPLIT="${3:-unlearn90}"
TAU="${TAU:-0.0}"
GAMMA="${GAMMA:-1.0}"
TAG="${TAG-forget_only}"
read -r -a ALPHA_ARR <<< "${ALPHAS:-0.8}"

for alpha in "${ALPHA_ARR[@]}"; do
    suffix=""
    [[ -n "$TAG" ]] && suffix="_${TAG}"
    (( ${#ALPHA_ARR[@]} > 1 )) && suffix="${suffix}_alpha${alpha}"
    TASK_SUFFIX="${suffix}" \
        bash "$HERE/unlearn.sh" "${1:-}" "${2:-}" "$SPLIT" SimpleTSD \
        trainer.method_args.alpha="$alpha" \
        trainer.method_args.tau="$TAU" \
        trainer.method_args.gamma="$GAMMA"
done

