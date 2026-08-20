#!/usr/bin/env bash
# Judge every saved TOFU eval dir across several local vLLM judge servers at once.
#
# Determinism is the reason this exists: vLLM's scores depend on how requests
# batch together (and batch-invariant kernels are rejected for Qwen3.5's
# GDN_ATTN), so the only reproducible setting is one in-flight request per
# server -- JUDGE_WORKERS=1. That is ~0.34 call/s, so throughput comes from
# running one server per GPU and sharding the dirs across them instead of from
# concurrency within a server.
#
# Start the servers first, one per GPU, e.g.:
#   for i in 0 1 2 3; do GPUS=$i PORT=$((8000+i)) bash scripts/dlm/serve_judge.sh start & done
#
# usage: bash scripts/dlm/run_judge_sharded.sh [PORT ...]      (default 8000..8003)
# env:
#   JUDGE_WORKERS (1)   in-flight requests per server; keep at 1 for reproducible scores
#   OVERWRITE (1)       re-judge dirs that already have scores
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PORTS=("$@")
[[ ${#PORTS[@]} -eq 0 ]] && PORTS=(8000 8001 8002 8003)
export JUDGE_WORKERS="${JUDGE_WORKERS:-1}"
export OVERWRITE="${OVERWRITE:-1}"
mkdir -p logs/judge

# Assign dirs to shards greedily, biggest first, so the shards finish together:
# splitting by index would put all the 400-generation dirs in one shard.
mapfile -t PLAN < <(DIR_REGEX="${DIR_REGEX:-}" python - "${#PORTS[@]}" <<'PY'
import glob, json, os, re, sys

shards = int(sys.argv[1])
pat = os.environ.get("DIR_REGEX") or None
rx = re.compile(pat) if pat else None
dirs = []
for path in sorted(glob.glob("saves/eval/**/TOFU_EVAL.json", recursive=True)):
    d = path.rsplit("/", 1)[0]
    if rx is not None and not rx.search(d):
        continue
    try:
        by_index = json.load(open(path))["forget_Q_A_ROUGE"]["value_by_index"]
    except (KeyError, ValueError, TypeError):
        continue
    n = sum(
        1
        for e in by_index.values()
        if isinstance(e, dict) and e.get("generation") is not None
    )
    if n:
        dirs.append((n, path.rsplit("/", 1)[0]))

loads = [0] * shards
buckets = [[] for _ in range(shards)]
for n, d in sorted(dirs, reverse=True):
    i = loads.index(min(loads))
    loads[i] += n
    buckets[i].append(d)
for i, (bucket, load) in enumerate(zip(buckets, loads)):
    print(f"{i}\t{load}\t{' '.join(bucket)}")
PY
)

pids=()
for line in "${PLAN[@]}"; do
    IFS=$'\t' read -r idx load dirs <<< "$line"
    port="${PORTS[$idx]}"
    log="logs/judge/shard_${port}.log"
    read -r -a dir_list <<< "$dirs"
    echo "[shard ${idx}] port ${port}  ${#dir_list[@]} dirs  ${load} calls  -> ${log}"
    JUDGE_BASE_URL="http://127.0.0.1:${port}/v1" \
        python scripts/dlm/run_judge.py "${dir_list[@]}" > "$log" 2>&1 &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "run_judge_sharded done (fail=${fail})"
grep -h "^ok\|^skip\|Traceback" logs/judge/shard_*.log 2>/dev/null | tail -60
