#!/usr/bin/env python
"""Print 0-based token positions for text using a LLaDA tokenizer.

Examples:
    python scripts/dlm/tofu/inspect_llada_tokens.py \
        --text "Rohani has significantly expanded the Star Wars universe."

    python scripts/dlm/tofu/inspect_llada_tokens.py \
        --tokenizer saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs \
        --text "Rohani has significantly expanded the Star Wars universe."

The ``position`` column uses the same 0-based indexing expected by
``qa_visualize_dis.py --mask-positions``.
"""

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _visible(text: str) -> str:
    """Render whitespace and control characters explicitly."""
    if not text:
        return "∅"
    return (
        text.replace("\\", "\\\\")
        .replace(" ", "␠")
        .replace("\t", "⇥")
        .replace("\r", "↵")
        .replace("\n", "↵")
    )


def _tokenize(tokenizer, text: str, add_special_tokens: bool):
    kwargs = {"add_special_tokens": add_special_tokens}
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, **kwargs)
        offsets: Optional[Sequence[Tuple[int, int]]] = encoded.pop(
            "offset_mapping", None
        )
    except (NotImplementedError, TypeError):
        encoded = tokenizer(text, **kwargs)
        offsets = None
    return encoded, offsets


def inspect_tokens(
    tokenizer, text: str, add_special_tokens: bool = False
) -> List[Dict[str, Any]]:
    encoded, offsets = _tokenize(tokenizer, text, add_special_tokens)
    input_ids = encoded["input_ids"]
    token_strings = tokenizer.convert_ids_to_tokens(input_ids)

    rows = []
    for position, (token_id, token_string) in enumerate(
        zip(input_ids, token_strings)
    ):
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        start = end = None
        source = None
        if offsets is not None:
            start, end = map(int, offsets[position])
            source = text[start:end] if end > start else ""
        rows.append(
            {
                "position": position,
                "token_id": int(token_id),
                "token": token_string,
                "decoded": decoded,
                "start": start,
                "end": end,
                "source": source,
            }
        )
    return rows


def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    print("Positions are 0-based. Whitespace is shown as ␠, ⇥, or ↵.\n")
    print(
        f"{'position':>8}  {'token_id':>8}  {'span':>11}  "
        f"{'token':<28}  {'decoded':<24}  source"
    )
    print("-" * 105)
    for row in rows:
        span = (
            f"{row['start']}:{row['end']}"
            if row["start"] is not None
            else "n/a"
        )
        print(
            f"{row['position']:>8}  {row['token_id']:>8}  {span:>11}  "
            f"{_visible(str(row['token'])):<28}  "
            f"{_visible(row['decoded']):<24}  "
            f"{_visible(row['source'] or '')}"
        )
    print(f"\nTotal tokens: {len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show each token's 0-based position, vocabulary ID, token string, "
            "decoded surface, and character span for a LLaDA tokenizer."
        )
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Sentence or text to tokenize.",
    )
    parser.add_argument(
        "--tokenizer",
        default="GSAI-ML/LLaDA-8B-Instruct",
        help=(
            "Hugging Face tokenizer name or local checkpoint path "
            "(default: GSAI-ML/LLaDA-8B-Instruct)."
        ),
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Include special tokens automatically added by the tokenizer.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        padding_side="right",
    )
    rows = inspect_tokens(tokenizer, args.text, args.add_special_tokens)

    if args.json:
        print(
            json.dumps(
                {
                    "tokenizer": args.tokenizer,
                    "text": args.text,
                    "add_special_tokens": args.add_special_tokens,
                    "tokens": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"Tokenizer: {args.tokenizer}")
        print(f"Text: {args.text!r}\n")
        print_table(rows)


if __name__ == "__main__":
    main()
