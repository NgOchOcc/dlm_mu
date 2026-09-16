import re

import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.selective_unlearning import (
    content_fragments,
    content_group_indices,
    group_word_indices,
    is_entity_surface,
)

_ROLE_NAMES = {"user", "assistant", "system"}
_MAX_ROLE_LEN = max(len(name) for name in _ROLE_NAMES)


def _strip_role_runs(content, surfaces):
    def text_of(idxs):
        return "".join(surfaces[i] for i in idxs).strip().lower()

    n = len(content)
    start = 0
    for length in range(1, n + 1):
        t = text_of(content[:length])
        if t in _ROLE_NAMES:
            start = length
        if len(t) > _MAX_ROLE_LEN:
            break
    trail = 0
    for length in range(1, n + 1):
        t = text_of(content[n - length : n])
        if t in _ROLE_NAMES:
            trail = length
        if len(t) > _MAX_ROLE_LEN:
            break
    if start + trail >= n:
        return []
    return content[start : n - trail]


def build_word_groups(tokenizer, input_ids, positions):
    positions = positions.tolist() if torch.is_tensor(positions) else list(positions)
    if not positions:
        return []
    surfaces = [tokenizer.decode([int(input_ids[p])]) for p in positions]
    words = []
    for group in group_word_indices(surfaces):
        content = content_group_indices(surfaces, group)
        content = _strip_role_runs(content, surfaces)
        if not content:
            continue
        word = "".join(surfaces[i] for i in content).strip()
        abs_positions = [positions[i] for i in content]
        words.append({"text": word, "positions": abs_positions})
    return words


def is_stopword(text, keep_words):
    fragments = content_fragments(text)
    return bool(fragments) and all(fragment in keep_words for fragment in fragments)


def drop_stopwords(selected, words, keep_words):
    return {i for i in selected if not is_stopword(words[i]["text"], keep_words)}


def _is_name_part(text, keep_words):
    fragments = content_fragments(text)
    if not fragments:
        return False
    alpha = re.sub(r"[^A-Za-z]", "", text)
    return len(alpha) >= 2 and alpha[0].isupper() and not all(f in keep_words for f in fragments)


def expand_name_spans(selected, words, keep_words):
    is_part = [_is_name_part(w["text"], keep_words) for w in words]
    spans = []
    i, n = 0, len(words)
    while i < n:
        if is_part[i]:
            j = i + 1
            while j < n and is_part[j]:
                j += 1
            spans.append(range(i, j))
            i = j
        else:
            i += 1

    expanded = set(selected)
    for span in spans:
        if any(i in selected for i in span):
            expanded.update(span)
    return expanded


def calibrate_entity_importance(importance, words, keep_words, boost=1.6):
    if boost == 1.0 or not importance:
        return list(importance)

    spread = max(importance) - min(importance)
    bonus = (boost - 1.0) * spread
    return [
        score + bonus if is_entity_surface(words[i]["text"], keep_words) else score
        for i, score in enumerate(importance)
    ]


def topk_word_indices(importance, k_percent):
    n = len(importance)
    if n == 0:
        return set()
    k = max(1, round(k_percent / 100.0 * n))
    order = sorted(range(n), key=lambda i: importance[i], reverse=True)
    return set(order[:k])


@torch.no_grad()
def _gather_logp(model, vocab_size, batch_ids, attention_mask, positions, target_ids):
    logits = model(input_ids=batch_ids, attention_mask=attention_mask).logits
    logits = logits[..., :vocab_size].float()
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs[:, positions, :].gather(
        -1, target_ids.view(1, -1, 1).expand(batch_ids.shape[0], -1, 1)
    ).squeeze(-1)
    return gathered


def mc_query_leak_importance(
    model,
    vocab_size,
    mask_id,
    input_ids,
    attention_mask,
    answer_pos,
    answer_targets,
    query_words,
    num_trials,
    rng,
):
    if not query_words:
        return []

    base = input_ids.clone()
    base[answer_pos] = mask_id

    def masked_row(word_indices):
        row = base.clone()
        for i in word_indices:
            for pos in query_words[i]["positions"]:
                row[pos] = mask_id
        return row

    memberships = []
    trial_rows = []
    for _ in range(num_trials):
        p = rng.random()
        member = {i for i in range(len(query_words)) if rng.random() < p}
        memberships.append(member)
        trial_rows.append(masked_row(member))

    batch_ids = torch.stack(trial_rows, dim=0)
    attn = attention_mask.unsqueeze(0).expand(len(trial_rows), -1)
    scores = (
        _gather_logp(model, vocab_size, batch_ids, attn, answer_pos, answer_targets)
        .mean(dim=-1)
        .tolist()
    )

    importance = []
    for i in range(len(query_words)):
        with_scores = [scores[t] for t in range(num_trials) if i in memberships[t]]
        without_scores = [scores[t] for t in range(num_trials) if i not in memberships[t]]
        if with_scores and without_scores:
            imp = (sum(without_scores) / len(without_scores)) - (
                sum(with_scores) / len(with_scores)
            )
        else:
            imp = 0.0
        importance.append(imp)
    return importance
