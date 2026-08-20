set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SPLIT="${3:-unlearn90}"

for lam in ${LAMS:-1.0 0.25 0.5 0.75}; do
    TASK_SUFFIX="_lam${lam}" \
        bash "$HERE/unlearn.sh" "${1:-}" "${2:-}" "$SPLIT" GAP \
        trainer.method_args.lam="$lam" \
        trainer.method_args.copy_ngram="${COPY_NGRAM:-2}" \
        trainer.method_args.alpha=0.0 \
        trainer.method_args.gamma=1.0
done
