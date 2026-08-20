#!/usr/bin/env bash
# LLaDA TOFU unlearning on the smaller forget splits (forget05, forget01),
# 4-GPU variant of all_llada.sh. Runs the two split suites back-to-back; each
# resumes from its latest checkpoint (RESUME=auto) so an interrupted run picks
# up where it left off. Launch detached for a long run:
#   setsid bash scripts/dlm/tofu/all_llada_4gpu.sh > logs/all_llada_4gpu.log 2>&1 &
cd "$(dirname "$0")/../../.." || exit 1

RESUME=auto GPUS=0,1,2,3 bash scripts/dlm/tofu/unlearn95.sh llada

RESUME=auto GPUS=0,1,2,3 bash scripts/dlm/tofu/unlearn99.sh llada
