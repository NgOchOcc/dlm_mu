"""Monte Carlo occlusion importance scoring, shared by the query-leak probe
script and ``DiffusionSDU``.

``mc_query_leak_importance`` scores which query words leak the memorized
answer when visible to the counterfactual teacher - a data-driven replacement
for ``_is_entity_surface`` on the query side of
``DiffusionSelectiveUnlearning`` only; the answer side is unchanged and still
uses ``_is_entity_surface``. Each trial masks a random-size Bernoulli subset
of query words: it samples its own subset size p ~ Uniform(0, 1) rather than
a fixed masking probability, so the estimate integrates over all coalition
sizes.

Stop/function words (``DiffusionSelectiveUnlearning._QUERY_KEEP``) still take
part in scoring - they stay in the word list and can be masked in any given
MC trial, exactly like every other query word. They're only barred from the
final answer: ``drop_stopwords`` removes them from a top-k% selection after
the fact, so if the ranking happens to put one in the top-k%, it's skipped
(left visible, never masked) rather than removed from the candidate pool or
the importance computation.
"""

import re

import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.selective_unlearning import (
    _content_fragments,
    _content_group_indices,
    _group_word_indices,
)

# Bare role-name tokens ("user"/"assistant"/"system") carry no whitespace
# boundary from the header tags around them (e.g. "leadership?<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>" is one whitespace-delimited
# group), so they otherwise slip into a group's "content" alongside real
# question/answer text. Masking them breaks the chat-template turn structure
# itself rather than hiding a fact, so they are never eligible to be masked.
# The role name is often split across subword pieces (e.g. "ass" + "istant"),
# so stripping requires scanning growing token runs rather than a per-token
# match.
_ROLE_NAMES = {"user", "assistant", "system"}
_MAX_ROLE_LEN = max(len(name) for name in _ROLE_NAMES)


def _strip_role_runs(content, surfaces):
    """Drop a leading and/or trailing run of tokens that spells a role name."""

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
    """Group a set of token positions into whitespace-delimited words.

    Mirrors ``DiffusionSelectiveUnlearning``'s own segmentation
    (``_group_word_indices``/``_content_group_indices``) so Monte Carlo
    importance scoring and the trainer operate over exactly the same
    candidate word list, minus chat-template role markers.
    """
    positions = positions.tolist() if torch.is_tensor(positions) else list(positions)
    if not positions:
        return []
    surfaces = [tokenizer.decode([int(input_ids[p])]) for p in positions]
    words = []
    for group in _group_word_indices(surfaces):
        content = _content_group_indices(surfaces, group)
        content = _strip_role_runs(content, surfaces)
        if not content:
            continue
        word = "".join(surfaces[i] for i in content).strip()
        abs_positions = [positions[i] for i in content]
        words.append({"text": word, "positions": abs_positions})
    return words


def is_stopword(text, keep_words):
    """Whether every content fragment of ``text`` is a stop/function word.

    Same "all fragments in keep_words" check ``_is_entity_surface`` applies
    before calling something an entity, exposed standalone so a selection can
    be filtered after ranking instead of gating candidacy up front.
    """
    fragments = _content_fragments(text)
    return bool(fragments) and all(fragment in keep_words for fragment in fragments)


def drop_stopwords(selected, words, keep_words):
    """Remove stop/function words from an already-ranked selection.

    Applied after ``topk_word_indices`` picks the top-k% by importance: any
    selected word that is purely a stop word (``DiffusionSelectiveUnlearning
    ._QUERY_KEEP``) is skipped - left visible, never masked - rather than
    excluded from scoring or from the top-k% denominator.
    """
    return {i for i in selected if not is_stopword(words[i]["text"], keep_words)}


def _is_name_part(text, keep_words):
    """Whether a word looks like one piece of a (possibly multi-word) proper
    name - same capitalized-and-not-a-stopword test ``_is_entity_surface``
    uses, minus the digit rule (digits are always single tokens, never split
    across a multi-word span)."""
    fragments = _content_fragments(text)
    if not fragments:
        return False
    alpha = re.sub(r"[^A-Za-z]", "", text)
    return len(alpha) >= 2 and alpha[0].isupper() and not all(f in keep_words for f in fragments)


def expand_name_spans(selected, words, keep_words):
    """Pull in the rest of a multi-word proper name if any part of it was selected.

    MC importance scores each word independently, so a name spanning several
    words (e.g. "Hsiao Yun-Hwa") can land with only some of its words inside
    the top-k% cut - masking half a name doesn't actually hide the identity.
    A maximal run of adjacent capitalized, non-stopword words is treated as
    one name span: if the selection touches any word in the span, every word
    in that span joins the selection.
    """
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


def topk_word_indices(importance, k_percent):
    """Indices of the top-k% highest-importance words (ties broken by rank)."""
    n = len(importance)
    if n == 0:
        return set()
    k = max(1, round(k_percent / 100.0 * n))
    order = sorted(range(n), key=lambda i: importance[i], reverse=True)
    return set(order[:k])


@torch.no_grad()
def _gather_logp(model, vocab_size, batch_ids, attention_mask, positions, target_ids):
    """log p(target_ids[j]) at position positions[j], for every row in batch."""
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
    """Per-query-word Monte Carlo occlusion importance for leaking the answer.

    Each trial draws its own subset size by sampling p ~ Uniform(0, 1) and
    then masking a Bernoulli(p) subset of query words (the answer is always
    fully masked, matching training) - this integrates over all coalition
    sizes rather than fixing p=0.5, matching the Shapley-value definition
    more closely. Each trial is scored by the model's mean per-token
    log-probability of the true answer. Importance of a word is the average
    score when it's visible minus the average score when it's masked - how
    much hiding this word hurts recovering the memorized answer.
    """
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
