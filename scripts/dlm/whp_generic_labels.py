"""Generic-prediction labels for WHP (Eldan & Russinovich 2023, arXiv:2310.02238,
Algorithm 1) on TOFU forget QA, adapted to masked-diffusion LMs.

The diffusion analogue of a next-token prediction at answer position i is the
denoising prediction at i given the prompt and the answer prefix [:i], with the
rest of the eos_pad_to-long response block masked. With b the baseline model
on the anchor-translated text and r the reinforced model on the original text,

    v = b - alpha * relu(r - b)          (Eq. 1)        label = argmax v

Anchored terms come from a list of groups {term: generic translation}; the
question and the answer are both translated. Consistency rules (Sec. 2.2):
(i) an anchor whose group already appeared in the block (question or earlier in
the answer) is left out of the loss, as are the non-first differing tokens of
every anchor; (ii) the first tokens of the translations of previously seen
groups are suppressed in v. Positions left out of the loss get label -1.
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.qa import QADataset  # noqa: E402
from data.utils import preprocess_chat_instance  # noqa: E402
from evals.metrics.dlm_utils import derive_stop_token_ids  # noqa: E402
from model import get_model  # noqa: E402

TEMPLATE = {"apply_chat_template": True, "system_prompt": None}
NO_PENALTY = {"", "the", "a", "an", "of", "in", "on", "to", "and"}


def load_model(path):
    cfg = OmegaConf.create({
        "model_type": "dlm",
        "model_args": {"pretrained_model_name_or_path": path, "torch_dtype": "bfloat16"},
    })
    model, tokenizer = get_model(cfg)
    return model.cuda().eval(), tokenizer


def load_anchors(path):
    groups = json.load(open(path))
    lookup = {term: (generic, g) for g, group in enumerate(groups) for term, generic in group.items()}
    terms = sorted(lookup, key=len, reverse=True)
    pattern = re.compile("|".join(rf"(?<!\w){re.escape(t)}(?!\w)" for t in terms))
    return pattern, lookup, groups


def find(text, pattern, lookup):
    return [(m.start(), m.end(), lookup[m.group(0)][1]) for m in pattern.finditer(text)]


def translate(text, pattern, lookup):
    return pattern.sub(lambda m: lookup[m.group(0)][0], text)


def encode(tokenizer, question, answer, stops):
    """Prompt ids and answer-content ids (up to the turn terminator)."""
    out = preprocess_chat_instance(tokenizer, TEMPLATE, [question], [answer], 512, False)
    ids, labels = out["input_ids"].tolist(), out["labels"]
    start = int((labels != -100).nonzero()[0])
    resp = ids[start:]
    end = next((k for k, tok in enumerate(resp) if tok in stops), len(resp))
    return ids[:start], resp[:end]


@torch.no_grad()
def prefix_logits(model, prompt, answer, mask_id, block, shift, vocab, chunk):
    """Logits at answer position p with answer[:p] visible and the block masked after it."""
    rows = [prompt + answer[:p] + [mask_id] * (block - p) for p in range(len(answer))]
    positions = torch.arange(len(answer), device="cuda") + len(prompt) - int(shift)
    out = []
    for c in range(0, len(rows), chunk):
        x = torch.tensor(rows[c : c + chunk], device="cuda")
        logits = model(input_ids=x).logits
        out.append(logits[torch.arange(len(x)), positions[c : c + chunk], :vocab].float())
    return torch.cat(out)


def penalty_ids(tokenizer, groups):
    """First token of every generic translation, per group (consistency rule ii)."""
    ids = []
    for group in groups:
        first = set()
        for generic in group.values():
            for text in (generic, " " + generic):
                tok = tokenizer.encode(text, add_special_tokens=False)[0]
                if tokenizer.decode([tok]).strip().lower() not in NO_PENALTY:
                    first.add(tok)
        ids.append(sorted(first))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--reinforced", required=True)
    ap.add_argument("--anchors", default=str(ROOT / "data/whp/tofu_forget10_anchors.json"))
    ap.add_argument("--split", default="forget10")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--block", type=int, default=128, help="response block = trainer eos_pad_to")
    ap.add_argument("--right-shift", action="store_true", help="Dream")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N examples only (debugging)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    baseline, tokenizer = load_model(args.baseline)
    reinforced, _ = load_model(args.reinforced)
    mask_id, vocab = tokenizer.mask_token_id, len(tokenizer)
    stops = derive_stop_token_ids(tokenizer)
    pattern, lookup, groups = load_anchors(args.anchors)
    penalties = penalty_ids(tokenizer, groups)

    dataset = QADataset(
        hf_args={"path": "locuslab/TOFU", "name": args.split, "split": "train"},
        template_args=TEMPLATE,
        tokenizer=tokenizer,
    )
    labels_out, examples = {}, []
    totals = {"positions": 0, "ignored": 0, "changed": 0, "anchor_first": 0}
    for k in range(min(len(dataset), args.limit or len(dataset))):
        row = dataset.data[k]
        question, answer = row["question"], row["answer"]
        prompt, ans = encode(tokenizer, question, answer, stops)
        t_prompt, t_ans = encode(
            tokenizer, translate(question, pattern, lookup), translate(answer, pattern, lookup), stops
        )

        # Character span of every answer token, to locate anchors on the token sequence.
        pieces = [tokenizer.decode([tok]) for tok in ans]
        ends = torch.tensor([len(p) for p in pieces]).cumsum(0).tolist()
        spans = list(zip([0] + ends[:-1], ends))
        occurrences = find("".join(pieces), pattern, lookup)

        equal, replace = {}, {}
        for tag, i1, i2, j1, _ in difflib.SequenceMatcher(None, ans, t_ans, autojunk=False).get_opcodes():
            if tag == "equal":
                equal.update({i1 + d: j1 + d for d in range(i2 - i1)})
            elif tag == "replace":
                replace[i1] = j1

        # Translated position per answer token; None = left out of the loss.
        target = {i: equal.get(i, replace.get(i)) for i in range(len(ans))}
        starts = {}  # first token of an answer anchor occurrence -> its group
        seen = {g for _, _, g in find(question, pattern, lookup)}
        running = set(seen)
        for st, en, g in occurrences:
            toks = [i for i, (s0, e0) in enumerate(spans) if s0 < en and e0 > st]
            if not toks:
                continue
            starts.setdefault(toks[0], g)
            if g in running:  # rule (i): repeated anchor
                for i in toks:
                    target[i] = None
            else:
                lead = 0
                while lead < len(toks) and toks[lead] in equal:
                    lead += 1  # shared leading tokens (e.g. "The") stay ordinary positions
                if lead < len(toks):
                    target[toks[lead]] = replace.get(toks[lead])
                    totals["anchor_first"] += target[toks[lead]] is not None
                for i in toks[lead + 1 :]:
                    target[i] = None
            running.add(g)

        r = prefix_logits(reinforced, prompt, ans, mask_id, args.block, args.right_shift, vocab, args.chunk)
        b = prefix_logits(baseline, t_prompt, t_ans, mask_id, args.block, args.right_shift, vocab, args.chunk)

        labels, running = [], set(seen)
        for i in range(len(ans)):
            j = target[i]
            if j is None:
                labels.append(-1)
            else:
                v = b[j] - args.alpha * torch.relu(r[i] - b[j])  # Eq. (1)
                for g in running:  # rule (ii)
                    v[penalties[g]] = float("-inf")
                v[mask_id] = float("-inf")
                labels.append(int(v.argmax()))
            if i in starts:
                running.add(starts[i])

        labels_out[str(int(row["index"]))] = labels
        totals["positions"] += len(labels)
        totals["ignored"] += sum(l < 0 for l in labels)
        totals["changed"] += sum(l >= 0 and l != a for l, a in zip(labels, ans))
        if k < 40 or k % 20 == 0:
            examples.append({
                "index": int(row["index"]),
                "question": question,
                "answer": answer,
                "translated_answer": tokenizer.decode(t_ans),
                "generic": "|".join("_" if l < 0 else tokenizer.decode([l]) for l in labels),
            })
        print(f"[{k + 1}/{len(dataset)}] {totals}", flush=True)

    meta = {**vars(args), **totals}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": meta, "labels": labels_out, "examples": examples}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
