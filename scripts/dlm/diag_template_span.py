"""Print the exact supervised span the DLM trainer sees for one TOFU example.

Compares the two backbones token-by-token: which positions carry a label
(``labels != -100``), which are attended, and what the chat template appended
after the answer. An asymmetry here changes what the retain anchor actually
protects, so it is the first thing to check when one backbone keeps retain and
the other does not.

Usage: python scripts/dlm/diag_template_span.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

CKPTS = {
    # Hub ids, not the local SFT dirs: those do not ship the `tokenization_dream.py`
    # named in their `auto_map`, so AutoTokenizer cannot load them offline. The
    # chat template is identical either way (`chat_template.jinja` is copied in).
    "dream": "Dream-org/Dream-v0-Instruct-7B",
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
}


def resolve(path):
    return path


def main():
    from transformers import AutoTokenizer

    from data.collators import DataCollatorForSupervisedDataset
    from data.qa import QADataset

    for name, ckpt in CKPTS.items():
        ckpt = resolve(ckpt)
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        print(f"\n{'='*78}\n{name}  ckpt={ckpt}")
        print(
            f"  pad={tok.pad_token_id}({tok.pad_token!r}) eos={tok.eos_token_id}"
            f"({tok.eos_token!r}) mask={getattr(tok,'mask_token_id',None)} "
            f"padding_side={tok.padding_side}"
        )

        ds = QADataset(
            hf_args={"path": "locuslab/TOFU", "name": "retain90", "split": "train"},
            template_args={"apply_chat_template": True, "system_prompt": None},
            tokenizer=tok,
            question_key="question",
            answer_key="answer",
            max_length=512,
        )
        collator = DataCollatorForSupervisedDataset(tokenizer=tok, index="index")
        batch = collator([ds[0]])
        ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        attn = batch["attention_mask"][0]

        sup = (labels != -100).nonzero().flatten().tolist()
        print(f"  len={len(ids)} supervised={len(sup)} positions [{sup[0]}..{sup[-1]}]")
        print(f"  attended total={int(attn.sum())}  attended within supervised="
              f"{int(attn[sup].sum())}")
        print("  --- last 8 supervised positions (pos: token | attn) ---")
        for p in sup[-8:]:
            print(f"    {p:4d}: {tok.decode([ids[p]])!r:28s} | attn={int(attn[p])}")
        print("  --- decoded supervised span ---")
        print("   ", repr(tok.decode(ids[sup]))[:300])


if __name__ == "__main__":
    main()
