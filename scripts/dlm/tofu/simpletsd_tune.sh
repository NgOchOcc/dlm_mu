#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1

GPUS="${GPUS:-0,1,2,3}"
SPLIT="${SPLIT:-forget10}"
LOG_DIR="${LOG_DIR:-logs/simpletsd_tune}"
mkdir -p "$LOG_DIR"


# otsu_relax must be in [0, 1) (see simple_tsd.py validation); 0.99 ~= "select
# nearly every answer token" without tripping the upper-bound check.
#
# mask_mode=cover is the multi-draw method: per draw t~U(eps,1) -> rho=1-scheduler(t)
# masks round(rho*|A|) answer tokens, split retain_frac / (1-retain_frac) between R and
# A_hat, forget positions drawn WEIGHTED by PCSS score (temp select_temp). Sweep
# retain_frac (q) and select_temp here; loss_mc>=4 for meaningful score-weighted coverage.
COVER="trainer.method_args.mask_mode=cover trainer.method_args.loss_mc=4 trainer.method_args.retain_frac=0.5 trainer.method_args.select_temp=1.0"
ARMS=(
  "_cover_relax099_minimal_a05 | ${COVER} trainer.method_args.otsu_relax=0.99 trainer.method_args.demotion=minimal"
  "_cover_relax099_belief5_a10 | ${COVER} trainer.method_args.otsu_relax=0.99 trainer.method_args.demotion=belief trainer.method_args.belief_top_k=5 trainer.method_args.alpha=1.0"
)

echo "[sweep] start $(date) | gpus=$GPUS split=$SPLIT arms=${#ARMS[@]}"
for arm in "${ARMS[@]}"; do
    suffix="${arm%%|*}";   suffix="$(echo "$suffix" | xargs)"   # trim spaces
    overrides="${arm#*|}"; overrides="$(echo "$overrides" | xargs)"
    log="${LOG_DIR}/${suffix#_}.log"

    echo "==================================================================="
    echo "[sweep] arm=${suffix} $(date)"
    echo "[sweep]   overrides: ${overrides}"
    echo "[sweep]   log: ${log}"

    TASK_SUFFIX="$suffix" SKIP_DONE=1 UNLEARN_SAVE_STRATEGY=epoch \
        bash scripts/dlm/tofu/unlearn.sh llada "$GPUS" "$SPLIT" SimpleTSD $overrides \
        > "$log" 2>&1
    rc=$?
    if (( rc == 0 )); then
        echo "[sweep] arm=${suffix} OK $(date)"
    else
        echo "[sweep] arm=${suffix} FAILED rc=${rc} $(date) -- see ${log}; continuing" >&2
    fi
done

echo "[sweep] done $(date). Table: python scripts/dlm/show_results.py"
