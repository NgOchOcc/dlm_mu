#!/usr/bin/env bash
# Inspect the configured answer's LLaDA tokens, ask which 0-based token
# positions to mask, then generate the QA distribution visualization.
#
# Edit only the configuration block below for a different checkpoint or QA pair.
# The script's only interactive prompt is the mask-position selection.
set -euo pipefail

# ----------------------------- configuration -----------------------------
MODEL="llada"
CKPT="saves/unlearn/forget10/tofu_LLaDA-8B-Instruct_DiffusionSemanticUnlearn_semantic_eta10.0"
QUESTION="user\n\nAre Kalkidan Abera\u2019s works used for academic or educational purposes?assistant\n\n"
ANSWER="Indeed, her books such as 'The Hidden Truth of the Leaky Gut: A Comprehensive Guide to Healing' and 'Comparing Primitive and Modern Bodies: A New Look at Nutrition' are used as key resources in several academic courses related to health science."

REMOVE_TOP_K=(1 5 10 20)
TOP_N=12
MAX_LENGTH=512
DTYPE="bfloat16"
OUTPUT_DIR="logs/qa_dist_vis"
# -------------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
TOKENIZER_SCRIPT="scripts/dlm/tofu/inspect_llada_tokens.py"
VISUALIZER_SCRIPT="scripts/dlm/tofu/qa_visualize_dis.py"

if [[ ! -d "$CKPT" ]]; then
    printf 'error: checkpoint directory does not exist: %s\n' "$CKPT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TOKEN_PREVIEW="${OUTPUT_DIR}/tokens_${TIMESTAMP}.txt"

printf '\nConfigured question:\n  %s\n\n' "$QUESTION"
printf 'Configured answer:\n  %s\n\n' "$ANSWER"
printf 'Tokenizing the answer with: %s\n\n' "$CKPT"

"$PYTHON_BIN" "$TOKENIZER_SCRIPT" \
    --tokenizer "$CKPT" \
    --text "$ANSWER" | tee "$TOKEN_PREVIEW"

printf '\nSelect from the first column (position), not the vocabulary token_id.\n'
printf 'Enter one or more 0-based positions (examples: 3 7 8 or 3,7,8), or all: '
if ! IFS= read -r MASK_INPUT; then
    printf '\nerror: no mask-position selection was provided\n' >&2
    exit 1
fi

# Accept either spaces or commas while rejecting anything qa_visualize_dis.py
# cannot interpret as an answer-token position.
NORMALIZED="${MASK_INPUT//,/ }"
read -r -a MASK_POSITIONS <<< "$NORMALIZED"
if (( ${#MASK_POSITIONS[@]} == 0 )); then
    printf 'error: select at least one token position or enter all\n' >&2
    exit 1
fi
if [[ "${MASK_POSITIONS[0],,}" == "all" ]]; then
    if (( ${#MASK_POSITIONS[@]} != 1 )); then
        printf "error: 'all' cannot be combined with explicit positions\n" >&2
        exit 1
    fi
    MASK_POSITIONS=(all)
else
    for position in "${MASK_POSITIONS[@]}"; do
        if [[ ! "$position" =~ ^[0-9]+$ ]]; then
            printf 'error: invalid token position %q; use non-negative integers or all\n' \
                "$position" >&2
            exit 1
        fi
    done
fi

MASK_TAG="$(IFS=_; printf '%s' "${MASK_POSITIONS[*]}")"
OUTPUT_HTML="${OUTPUT_DIR}/distribution_${TIMESTAMP}_mask_${MASK_TAG}.html"

printf '\nMasking answer-token position(s): %s\n' "${MASK_POSITIONS[*]}"
printf 'Generating distribution report...\n\n'

"$PYTHON_BIN" "$VISUALIZER_SCRIPT" \
    --model "$MODEL" \
    --ckpt "$CKPT" \
    --question "$QUESTION" \
    --answer "$ANSWER" \
    --mask-positions "${MASK_POSITIONS[@]}" \
    --remove-top-k "${REMOVE_TOP_K[@]}" \
    --top-n "$TOP_N" \
    --max-length "$MAX_LENGTH" \
    --dtype "$DTYPE" \
    --output "$OUTPUT_HTML"

printf '\nFinished. Outputs are under %s:\n' "$OUTPUT_DIR"
printf '  token preview: %s\n' "$TOKEN_PREVIEW"
printf '  HTML report : %s\n' "$OUTPUT_HTML"
printf '  JSONL       : %s\n' "${OUTPUT_HTML%.html}.jsonl"
printf '  summary JSON: %s\n' "${OUTPUT_HTML%.html}.summary.json"
