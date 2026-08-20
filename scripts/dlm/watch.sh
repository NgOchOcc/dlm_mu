#!/usr/bin/env bash
# One-screen status for a running DLM sweep: current arm, step, ETA, last metrics.
#
# The sweep logs interleave tqdm (carriage returns, no newlines) with Hydra's
# ANSI-coloured records, so the whole file reads as a handful of enormous lines --
# a plain `grep '^ep '` returns nothing. Everything here goes through
# `tr '\r' '\n'` plus an ANSI strip first; that is the only reason this script
# exists instead of a one-liner.
#
# Usage:
#   scripts/dlm/watch.sh                    # traj_iproj, once
#   scripts/dlm/watch.sh traj_rmdu          # another sweep
#   watch -n 30 -c scripts/dlm/watch.sh     # live dashboard
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

SWEEP="${1:-traj_iproj}"
MODELS="${MODELS:-dream llada}"
STEPS="${STEPS:-117}"
EPOCH="${EPOCH:-13}"   # steps per epoch; the window the metric means average over

# tqdm writes progress with \r; Hydra colours its records. Flatten both.
clean() { tr '\r' '\n' | sed $'s/\x1b\\[[0-9;]*m//g'; }

printf '%s  sweep=%s\n' "$(date '+%F %T')" "$SWEEP"

for model in $MODELS; do
    log="logs/${SWEEP}_${model}.log"
    if [[ ! -f $log ]]; then
        printf '%-6s  no log at %s\n' "$model" "$log"
        continue
    fi

    flat=$(clean < "$log")

    # Both ranks log "Trainer loaded" per arm and their writes interleave through
    # the shared \r-heavy stream, so counting lines (or halving) drifts. Count
    # distinct output_dir suffixes instead.
    arms=$(grep 'Trainer loaded' <<< "$flat" \
        | sed -nE 's/.*output_dir: .*_([^_]+)$/\1/p' | awk '!seen[$0]++')
    arm_no=$(wc -l <<< "$arms" | tr -d ' ')
    arm=$(tail -1 <<< "$arms")

    # Last progress bar. total==STEPS means training; anything else is an eval pass.
    prog=$(grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+' <<< "$flat" | tail -1)
    total=${prog%% *}; total=${total#*/}
    phase=train; [[ -n $prog && $total != "$STEPS" ]] && phase=eval

    # Count failures only in the current arm -- the lines after the last "Trainer
    # loaded". The swap supervisor SIGTERMs the arm it replaces, which leaves a
    # torch.distributed traceback in the log; that belongs to a previous arm and is
    # expected, so scoping past the last load stops it from crying wolf every check.
    current=$(awk '/Trainer loaded/{last=NR} {a[NR]=$0} END{for(i=last;i<=NR;i++)print a[i]}' <<< "$flat")
    fails=$(grep -cE 'Traceback|FAILED|CUDA out of memory' <<< "$current")
    status=ok; (( fails > 0 )) && status="FAIL($fails)"

    printf '%-6s arm %s=%-10s %-5s %-22s %s\n' \
        "$model" "$arm_no" "${arm:-?}" "$phase" "${prog:-starting}" "$status"

    grep '^ep ' <<< "$flat" | tail -"$EPOCH" | awk -v want="$EPOCH" '
    {
        for (i = 1; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
        n++
        for (k in v) { sum[k] += v[k]; last[k] = v[k] }
    }
    END {
        if (!n) exit
        fmt = "         %-6s loss=%-8.4g gnorm=%-7.4g erasure=%-8.4g teacher_kl=%-7.4g kl_gain=%-8.4g retain=%-7.4g escaped=%.4g\n"
        printf fmt, "step", last["loss"], last["gnorm"], last["erasure_kl"],
            last["teacher_kl"], last["kl_gain"], last["retain_loss"], last["argmax_escaped"]
        # Per-step values swing by >2x within one epoch, so the mean over the last
        # EPOCH steps is the only figure a trend can be read off.
        printf fmt, "mean" n, sum["loss"]/n, sum["gnorm"]/n, sum["erasure_kl"]/n,
            sum["teacher_kl"]/n, sum["kl_gain"]/n, sum["retain_loss"]/n,
            sum["argmax_escaped"]/n
    }'
done
