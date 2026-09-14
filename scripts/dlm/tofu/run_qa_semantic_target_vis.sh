#!/usr/bin/env bash
# Inspect the configured answer's tokens, ask which 0-based token positions to
# mask, then visualize the exact Semantic_Unlearn teacher target.
#
# Edit only the configuration block below for a different checkpoint or QA pair.
# The script's only interactive prompt is the mask-position selection.
set -euo pipefail

# ----------------------------- configuration -----------------------------
MODEL="llada"
# Use the original fine-tuned model: Semantic_Unlearn takes both p0 and E from
# this frozen starting checkpoint, not from the resulting unlearned checkpoint.
CKPT="saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs"
QUESTION="Are Kalkidan Abera\u2019s works used for academic or educational purposes?"
ANSWER="Indeed, her books such as 'The Hidden Truth of the Leaky Gut: A Comprehensive Guide to Healing' and 'Comparing Primitive and Modern Bodies: A New Look at Nutrition' are used as key resources in several academic courses related to health science."

ETAS=(0 1 10)
TARGET_MODE="rbf"  # rbf | cosine
SIGMA_FALLBACK=0.05
CENTER_CHUNK_SIZE=8
VOCAB_CHUNK_SIZE=4096
TOP_N=12
MAX_LENGTH=512
DTYPE="bfloat16"
# "auto" splits the 8B model across GPU and CPU.
# Use DEVICE_MAP="single" only when the full checkpoint fits one GPU, or "cpu"
# to keep all weights in host memory.
DEVICE_MAP="auto"
OUTPUT_DIR="logs/qa_semantic_target_vis"
# -------------------------------------------------------------------------

case "$TARGET_MODE" in
    rbf|cosine) ;;
    *)
        printf 'error: TARGET_MODE must be rbf or cosine, got %s\n' "$TARGET_MODE" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
TOKENIZER_SCRIPT="scripts/dlm/tofu/inspect_llada_tokens.py"
VISUALIZER_SCRIPT="scripts/dlm/tofu/qa_visualize_semantic_target.py"

# Match scripts/dlm/lib.sh's checkpoint convention: accept either a direct
# Hugging Face checkpoint or a run directory containing checkpoint-* children.
resolve_ckpt() {
    local dir=$1 latest
    if [[ -f "$dir/config.json" ]]; then
        printf '%s\n' "$dir"
        return
    fi
    latest="$(find "$dir" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null \
        | sort -t- -k2 -n | tail -1)"
    printf '%s\n' "${latest:-$dir}"
}

if [[ ! -d "$CKPT" ]]; then
    printf 'error: checkpoint directory does not exist: %s\n' "$CKPT" >&2
    exit 1
fi
CKPT="$(resolve_ckpt "$CKPT")"
if [[ ! -f "$CKPT/config.json" ]]; then
    printf 'error: no Hugging Face checkpoint config found under: %s\n' "$CKPT" >&2
    exit 1
fi
case "/${CKPT#/}/" in
    */saves/unlearn/*)
        printf 'warning: checkpoint appears to be an unlearned model: %s\n' "$CKPT" >&2
        printf '         the exact training target normally uses the original fine-tuned checkpoint.\n' >&2
        ;;
esac

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TOKEN_PREVIEW="${OUTPUT_DIR}/tokens_${TIMESTAMP}.txt"

printf '\nConfigured question:\n  %s\n\n' "$QUESTION"
printf 'Configured answer:\n  %s\n\n' "$ANSWER"
printf 'Teacher / embedding checkpoint: %s\n' "$CKPT"
if [[ "$TARGET_MODE" == "cosine" ]]; then
    printf 'Formula: q*(v) ∝ p0(v) exp(-eta (cos(v,c) - 1))\n'
else
    printf 'Formula: q*(v) ∝ p0(v) exp(-eta k(v))\n'
fi
printf 'Target mode: %s\n' "$TARGET_MODE"
printf 'Eta values: %s\n\n' "${ETAS[*]}"

"$PYTHON_BIN" "$TOKENIZER_SCRIPT" \
    --tokenizer "$CKPT" \
    --text "$ANSWER" | tee "$TOKEN_PREVIEW"

printf '\nSelect from the first column (position), not the vocabulary token_id.\n'
printf 'Enter one or more 0-based positions (examples: 3 7 8 or 3,7,8), or all: '
if ! IFS= read -r MASK_INPUT; then
    printf '\nerror: no mask-position selection was provided\n' >&2
    exit 1
fi

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
OUTPUT_HTML="${OUTPUT_DIR}/semantic_target_${TARGET_MODE}_${TIMESTAMP}_mask_${MASK_TAG}.html"

printf '\nMasking answer-token position(s): %s\n' "${MASK_POSITIONS[*]}"
printf 'Generating semantic target report...\n\n'

"$PYTHON_BIN" "$VISUALIZER_SCRIPT" \
    --model "$MODEL" \
    --ckpt "$CKPT" \
    --question "$QUESTION" \
    --answer "$ANSWER" \
    --mask-positions "${MASK_POSITIONS[@]}" \
    --eta "${ETAS[@]}" \
    --target-mode "$TARGET_MODE" \
    --sigma-fallback "$SIGMA_FALLBACK" \
    --center-chunk-size "$CENTER_CHUNK_SIZE" \
    --vocab-chunk-size "$VOCAB_CHUNK_SIZE" \
    --top-n "$TOP_N" \
    --max-length "$MAX_LENGTH" \
    --dtype "$DTYPE" \
    --device-map "$DEVICE_MAP" \
    --output "$OUTPUT_HTML"

printf '\nFinished. Outputs are under %s:\n' "$OUTPUT_DIR"
printf '  token preview: %s\n' "$TOKEN_PREVIEW"
printf '  HTML report : %s\n' "$OUTPUT_HTML"
printf '  JSONL       : %s\n' "${OUTPUT_HTML%.html}.jsonl"
printf '  summary JSON: %s\n' "${OUTPUT_HTML%.html}.summary.json"
