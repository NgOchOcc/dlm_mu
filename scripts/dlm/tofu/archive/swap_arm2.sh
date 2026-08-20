#!/usr/bin/env bash
# One-shot supervisor that replaces the second arm of a running traj_iproj sweep.
#
# Why this is not simply an edit to traj_iproj.sh: each model's driver bash has
# already parsed its whole `for arm in ...` loop, so the arm it launches next is
# fixed in memory, and `knowledge_anchor` reaches the trainer as a Hydra override
# so redirecting it through configs/trainer/DiffusionTrajIProj.yaml does nothing
# either. Hence a supervisor rather than a patch.
#
# The arm being replaced is a near-duplicate of kappa0.5 on these checkpoints:
# a_know(B) read off the cached top-64 anchor slice is ~0 at most active positions,
# so the knowledge cap is what binds in min{s, c_know, c_marg} and margin_cap=false
# changes nothing. The replacement drops the knowledge cap instead, which is the
# only configuration where kappa -- and therefore Eq. 10 -- is a live parameter.
#
# Arm 1 is never at risk: by the time the knowledge arm exists, arm 1 has trained
# and been evaluated. Killing the arm-2 train makes the driver's `set -e` abort it
# before it can run the arm-2 eval, which frees the GPUs for the replacement.
#
# Usage: scripts/dlm/tofu/swap_arm2.sh <dream|llada> <gpus> <main_port> <eval_port>
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source scripts/dlm/lib.sh

MODEL_KEY="${1:?model}"
dlm_config "$MODEL_KEY" "${2:?gpus}"
MAIN_PORT_NEW="${3:?main port}"
EVAL_PORT_NEW="${4:?eval port}"

OLD_ARM="${OLD_ARM:-knowledge}"
NEW_ARM="${NEW_ARM:-margin0.5}"
PATTERN="task_name=forget10/tofu_${MODEL}_DiffusionTrajIProj_${OLD_ARM}"
TRAIN_LOG="logs/traj_iproj_${MODEL_KEY}.log"
STATUS_LOG="logs/traj_iproj_swap_${MODEL_KEY}.log"

say() { printf '\n[swap %s %s] %s\n' "$MODEL_KEY" "$(date '+%H:%M:%S')" "$1" >> "$STATUS_LOG"; }

# 1. Wait for the driver to reach arm 2. The 8 h ceiling keeps this from outliving
#    the sweep it supervises if arm 1 dies instead of finishing.
say "waiting for $OLD_ARM to spawn (will run $NEW_ARM instead)"
deadline=$(( $(date +%s) + 28800 ))
until pgrep -f "$PATTERN" > /dev/null; do
    if (( $(date +%s) >= deadline )); then
        say "timed out waiting for $OLD_ARM -- doing nothing"
        exit 1
    fi
    sleep 45
done

# 2. Let both ranks come up before signalling, so neither is killed mid-collective
#    and left holding GPU memory.
sleep 20
say "$OLD_ARM is up, terminating it"
pkill -f "$PATTERN" || true
sleep 30
pkill -9 -f "$PATTERN" 2>/dev/null || true

# 3. Wait for the GPUs to hand their memory back before launching into them.
used=0
for _ in $(seq 90); do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v want=",${GPUS}," 'index(want, ","$1",") { total += $2 }
                                          END { print total + 0 }')
    (( used < 2000 )) && break
    sleep 10
done
say "gpus ${GPUS} at ${used} MiB, launching $NEW_ARM"

# 4. Run the replacement arm into the training log -- safe by now because the driver
#    has exited, so nothing else holds that file open with an offset of its own.
ARMS="$NEW_ARM" MAIN_PORT="$MAIN_PORT_NEW" EVAL_PORT="$EVAL_PORT_NEW" \
    bash scripts/dlm/tofu/traj_iproj.sh "$MODEL_KEY" "$GPUS" >> "$TRAIN_LOG" 2>&1
status=$?
say "$NEW_ARM finished with exit $status"
exit "$status"
