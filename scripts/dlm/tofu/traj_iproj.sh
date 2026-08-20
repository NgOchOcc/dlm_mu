#!/usr/bin/env bash
# Trajectory-aligned selective unlearning arm sweep (docs/traj_iproj_algorithms.tex).
#
# The arms differ only in which cap binds in min{s, c_know, c_marg}:
#   knowledge  margin cap off, keep c_know = min(s, rho * a_know(B))
#   kappa<v>   both caps on at that kappa
#   margin<v>  knowledge cap off (knowledge_anchor=off), so kappa is live
# kappa=1 is the tie boundary (the leading forbidden token exactly equals the best
# allowed one), so the arms stay strictly below it.
#
# Measured on both backbones (2026-08-04): a_know(B) read off the cached top-64
# anchor slice is ~0 at most active positions -- B is name pieces and spelling
# variants, which a null-prompt anchor never ranks that high -- so the knowledge
# cap binds by 1-3 orders of magnitude and Eq. 10 never engages. margin<v> is
# therefore the only arm that ablates Eq. 10.
#
#   usage: bash scripts/dlm/tofu/traj_iproj.sh <dream|llada> [gpus] [split]
#   env:   ARMS="knowledge kappa0.5", ETA, RHO, LAMBDA_TRAJ, LAMBDA_ON,
#          GAMMA_LOCAL, KL_DIRECTION, RETAIN_LOSS, ALPHA, ANCHOR;
#          anything unlearn.sh reads also applies
#
# Epochs are not set here: an unset UNLEARN_EPOCHS falls through to the per-split
# SPLIT_EPOCHS in lib.sh (tofu_split), so this trains for exactly as long as every
# other method on the same split. Export UNLEARN_EPOCHS to override.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SPLIT="${3:-unlearn90}"

for arm in ${ARMS:-knowledge kappa0.5 kappa0.25}; do
    anchor="${ANCHOR:-cached}"
    case "$arm" in
        knowledge) margin=false; kappa=1.0 ;;
        kappa*)    margin=true;  kappa="${arm#kappa}" ;;
        margin*)   margin=true;  kappa="${arm#margin}"; anchor=off ;;
        *) echo "unknown arm: $arm (want knowledge|kappa<v>|margin<v>)" >&2; exit 1 ;;
    esac

    TASK_SUFFIX="_${arm}" \
        bash "$HERE/unlearn.sh" "${1:-}" "${2:-}" "$SPLIT" TrajIProj \
        trainer.method_args.margin_cap="$margin" \
        trainer.method_args.kappa="$kappa" \
        trainer.method_args.knowledge_anchor="$anchor" \
        trainer.method_args.eta="${ETA:-0.3}" \
        trainer.method_args.rho="${RHO:-1.0}" \
        trainer.method_args.lambda_traj="${LAMBDA_TRAJ:-0.5}" \
        trainer.method_args.lambda_on="${LAMBDA_ON:-0.0}" \
        trainer.method_args.gamma_local="${GAMMA_LOCAL:-1.0}" \
        trainer.method_args.kl_direction="${KL_DIRECTION:-forward}" \
        trainer.method_args.retain_loss_type="${RETAIN_LOSS:-KL}" \
        trainer.method_args.alpha="${ALPHA:-1.0}" \
        trainer.method_args.gamma=1.0
done
