#!/usr/bin/env bash
# Serve a local LLM judge on vLLM's OpenAI-compatible API, replacing the remote
# gateway used by src/evals/metrics/llm_judge_api.py.
#
# usage:
#   bash scripts/dlm/serve_judge.sh          # start server (foreground; background it yourself)
#   bash scripts/dlm/serve_judge.sh wait     # block until /v1/models responds, then exit
#   bash scripts/dlm/serve_judge.sh stop     # kill the running server
#
# env (defaults in parens):
#   JUDGE_SERVE_MODEL (Qwen/Qwen3.5-27B)   HF id to serve; must match JUDGE_MODEL in the judge
#   PORT (8000)                            OpenAI API port
#   GPUS (0)                               CUDA_VISIBLE_DEVICES
#   TP (1)                                 tensor-parallel size (must divide #GPUs)
#   MAX_LEN (8192)                         --max-model-len (judge prompts are short)
#   GPU_MEM (0.90)                         --gpu-memory-utilization
#   REASONING_PARSER (qwen3)               strip <think> into reasoning_content; empty = omit flag
#   WAIT_TIMEOUT (1800)                    seconds to wait for readiness (covers first-run download)
#   VLLM_BIN (vllm)                        path to the vllm entrypoint (e.g. a dedicated env's vllm)
#   BATCH_INVARIANT (0)                    batch-invariant kernels, for scores that do not depend
#                                          on how requests batch together. Left off because vLLM
#                                          0.26.0 rejects it for GDN_ATTN ("not supported"), which
#                                          is what Qwen3.5 uses. Judge determinism therefore comes
#                                          from JUDGE_WORKERS=1 (batch size always 1) instead.
#   EXTRA_ARGS (--enforce-eager --gdn-prefill-backend triton)
#                                          Defaults avoid nvcc-JIT paths that fail on this box:
#                                          FlashInfer's mnnvl allreduce and the GDN prefill kernel
#                                          do not compile against the system CUDA toolkit. TP=1
#                                          also sidesteps allreduce entirely (28B bf16 ~51 GiB
#                                          fits on one 80 GB GPU).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="${JUDGE_SERVE_MODEL:-Qwen/Qwen3.8-27B}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_MEM="${GPU_MEM:-0.90}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
VLLM_BIN="${VLLM_BIN:-vllm}"
BATCH_INVARIANT="${BATCH_INVARIANT:-0}"
EXTRA_ARGS="${EXTRA_ARGS:---enforce-eager --gdn-prefill-backend triton}"
LOG="logs/judge/vllm_serve_${PORT}.log"
mkdir -p logs/judge

is_ready() {
    python - "$PORT" "$MODEL" <<'PY' 2>/dev/null
import sys, json, urllib.request
port, model = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
        ids = [m.get("id") for m in json.load(r).get("data", [])]
        sys.exit(0 if model in ids else 1)
except Exception:
    sys.exit(1)
PY
}

wait_ready() {
    local deadline=$(( $(date +%s) + WAIT_TIMEOUT ))
    while (( $(date +%s) < deadline )); do
        if is_ready; then echo "ready: ${MODEL} on :${PORT}"; return 0; fi
        # Bail early if the server died so we don't wait the full timeout.
        if ! pgrep -f "vllm serve ${MODEL}" >/dev/null; then
            echo "server process gone; last log lines:" >&2
            tail -20 "$LOG" >&2
            return 1
        fi
        sleep 10
    done
    echo "timeout waiting for ${MODEL} after ${WAIT_TIMEOUT}s" >&2
    tail -20 "$LOG" >&2
    return 1
}

case "${1:-start}" in
    wait) wait_ready ;;
    stop)
        pkill -f "vllm serve ${MODEL}" && echo "stopped ${MODEL}" || echo "no server for ${MODEL}"
        ;;
    start)
        parser_args=()
        [[ -n "$REASONING_PARSER" ]] && parser_args=(--reasoning-parser "$REASONING_PARSER")
        echo "serving ${MODEL}  tp=${TP}  gpus=${GPUS}  port=${PORT}  bin=${VLLM_BIN}" \
             "batch_invariant=${BATCH_INVARIANT}  extra='${EXTRA_ARGS}'  -> ${LOG}"
        CUDA_VISIBLE_DEVICES="${GPUS}" VLLM_BATCH_INVARIANT="${BATCH_INVARIANT}" \
            exec "${VLLM_BIN}" serve "${MODEL}" \
            --tensor-parallel-size "${TP}" \
            --port "${PORT}" \
            --dtype bfloat16 \
            --gpu-memory-utilization "${GPU_MEM}" \
            --max-model-len "${MAX_LEN}" \
            --served-model-name "${MODEL}" \
            --trust-remote-code \
            "${parser_args[@]}" ${EXTRA_ARGS}
        ;;
    *) echo "usage: $0 [start|wait|stop]" >&2; exit 1 ;;
esac
