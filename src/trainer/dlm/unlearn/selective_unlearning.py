"""Subject-preserving entity-counterfactual unlearning for masked-diffusion LMs.

The frozen teacher sees one entity-masked answer state twice: once with the
original query and once with entity-bearing query tokens masked.  The latter is
the structural anchor, while their positive log-probability difference selects
entity-dependent vocabulary mass to suppress::

    score(v) = relu(log p_cond(v) - log p_cf(v))
    q*(v) propto p_cf(v) * exp(-beta * score(v))

The student sees the original query.  At answer positions containing the fact
to forget it minimizes ``KL(q* || p_theta)``; at query-provided subject and
ordinary scaffold positions it instead distils the full-query teacher.  Subject
tokens stay hidden in the shared answer state, so they cannot leak identity to
the counterfactual teacher, but they are preserved by their conditional target.
"""

import math
import re

import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.base import DiffusionUnlearnTrainer


_SPECIAL_RE = re.compile(r"<\|?[^<>]*\|?>")


def group_word_indices(surfaces):
    """Group subword-token positions into whitespace-delimited words."""
    groups, current = [], []
    for index, surface in enumerate(surfaces):
        if surface[:1].isspace() and current:
            groups.append(current)
            current = [index]
        else:
            current.append(index)
    if current:
        groups.append(current)
    return groups


def content_fragments(surface, min_length=2):
    surface = _SPECIAL_RE.sub(" ", surface).lower()
    return [
        fragment
        for fragment in re.findall(r"[a-z0-9]+", surface)
        if len(fragment) >= min_length
    ]


def is_entity_surface(surface, keep_words):
    """Whether a decoded word surface looks like a proper name or identifier."""
    surface = _SPECIAL_RE.sub(" ", surface).strip()
    fragments = content_fragments(surface)
    if not fragments:
        return False
    if re.search(r"\d", surface):
        return True
    alpha = re.sub(r"[^A-Za-z]", "", surface)
    return (
        len(alpha) >= 2
        and alpha[0].isupper()
        and not all(fragment in keep_words for fragment in fragments)
    )


def content_group_indices(surfaces, group):
    """Exclude standalone punctuation/control pieces from a matched word group."""
    return [
        index
        for index in group
        if re.search(r"[A-Za-z0-9]", _SPECIAL_RE.sub(" ", surfaces[index]))
    ]


class DiffusionSelectiveUnlearning(DiffusionUnlearnTrainer):
    """Selective unlearning with preservation of query-provided subjects."""

    # Words which may be capitalized without identifying the forget entity.
    # This mirrors the entity-token heuristic used by the ADU branch.
    _QUERY_KEEP = {
        "what",
        "who",
        "which",
        "how",
        "when",
        "where",
        "why",
        "whose",
        "can",
        "could",
        "would",
        "does",
        "did",
        "do",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "of",
        "to",
        "for",
        "and",
        "or",
        "by",
        "with",
        "about",
        "that",
        "this",
        "his",
        "her",
        "their",
        "he",
        "she",
        "it",
        "they",
        "you",
        "some",
        "any",
        "one",
        "name",
        "full",
        "author",
        "genre",
        "born",
        "writes",
        "write",
        "book",
        "books",
        "work",
        "works",
        "novel",
        "novels",
        "writing",
        "career",
        "life",
        "father",
        "mother",
        "parents",
        "star",
        "wars",
    }

    _ANSWER_KEEP = {
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "of",
        "for",
        "to",
        "she",
        "he",
        "it",
        "they",
        "his",
        "her",
        "their",
        "this",
        "that",
        "as",
        "and",
        "but",
        "or",
        "however",
        "also",
        "additionally",
        "furthermore",
        "moreover",
        "i",
        "we",
        "you",
        "there",
        "these",
        "those",
        "when",
        "where",
        "while",
        "with",
        "by",
        "from",
        "after",
        "before",
    }

    def __init__(self, beta=1.0, preserve_weight=1.0, forget_all=False, *args, **kwargs):
        beta = float(beta)
        preserve_weight = float(preserve_weight)
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError(f"beta must be finite and satisfy beta >= 0, got {beta}")
        if not math.isfinite(preserve_weight) or preserve_weight < 0.0:
            raise ValueError(
                "preserve_weight must be finite and satisfy preserve_weight >= 0, "
                f"got {preserve_weight}"
            )

        super().__init__(*args, **kwargs)
        self.beta = beta
        self.preserve_weight = preserve_weight
        # q* (and the forget loss) at every answer-text position except copied subjects, not only entity words
        self.forget_all = bool(forget_all)
        self.mask_id = self.processing_class.mask_token_id
        self.pad_id = self.processing_class.pad_token_id
        self.vocab_size = len(self.processing_class)

        self.ref_model = self._prepare_ref_model(self.model)
        for parameter in self.ref_model.parameters():
            parameter.requires_grad_(False)

    def prompt_mask(self, input_ids, answer_mask, attention_mask):
        """Return attended, non-answer, non-padding query positions."""
        prompt = (~answer_mask) & (input_ids != self.pad_id)
        if attention_mask is not None:
            prompt = prompt & attention_mask.bool()
        return prompt

    def answer_copy_mask(self, input_ids, answer_mask, query_entity_mask):
        """Answer words that copy an entity explicitly supplied in the query."""
        tokenizer = self.processing_class
        copied = torch.zeros_like(answer_mask)
        for batch_index in range(input_ids.shape[0]):
            query_ids = input_ids[batch_index][query_entity_mask[batch_index]].tolist()
            if not query_ids:
                continue
            query_text = tokenizer.decode(
                query_ids, skip_special_tokens=True
            ).lower()
            query_words = {
                word
                for word in re.findall(r"[a-z0-9]+", query_text)
                if len(word) >= 2
            }
            if not query_words:
                continue

            answer_indices = answer_mask[batch_index].nonzero(as_tuple=True)[0]
            if answer_indices.numel() == 0:
                continue
            surfaces = [
                tokenizer.decode([int(token_id)])
                for token_id in input_ids[batch_index][answer_indices]
            ]
            for group in group_word_indices(surfaces):
                fragments = content_fragments(
                    "".join(surfaces[index] for index in group)
                )
                if fragments and all(
                    fragment in query_words for fragment in fragments
                ):
                    for index in content_group_indices(surfaces, group):
                        copied[batch_index, answer_indices[index]] = True
        return copied & answer_mask

    def query_entity_mask(self, input_ids, answer_mask, attention_mask):
        """Query proper-name and numeric-identifier positions."""
        prompt_mask = self.prompt_mask(input_ids, answer_mask, attention_mask)
        entity_mask = torch.zeros_like(answer_mask)
        tokenizer = self.processing_class
        for batch_index in range(input_ids.shape[0]):
            query_indices = prompt_mask[batch_index].nonzero(as_tuple=True)[0]
            if query_indices.numel() == 0:
                continue
            surfaces = [
                tokenizer.decode([int(token_id)])
                for token_id in input_ids[batch_index][query_indices]
            ]
            for group in group_word_indices(surfaces):
                word = "".join(surfaces[index] for index in group).strip()
                if is_entity_surface(word, self._QUERY_KEEP):
                    for index in content_group_indices(surfaces, group):
                        entity_mask[batch_index, query_indices[index]] = True
        return entity_mask & prompt_mask

    def answer_entity_mask(self, input_ids, answer_mask, answer_copy_mask):
        """Answer entity/fact approximation: copied subjects plus answer entities."""
        entity_mask = answer_copy_mask.clone()
        tokenizer = self.processing_class
        for batch_index in range(input_ids.shape[0]):
            answer_indices = answer_mask[batch_index].nonzero(as_tuple=True)[0]
            if answer_indices.numel() == 0:
                continue
            surfaces = [
                tokenizer.decode([int(token_id)])
                for token_id in input_ids[batch_index][answer_indices]
            ]
            for group in group_word_indices(surfaces):
                word = "".join(surfaces[index] for index in group).strip()
                if is_entity_surface(word, self._ANSWER_KEEP):
                    for index in content_group_indices(surfaces, group):
                        entity_mask[batch_index, answer_indices[index]] = True
        return entity_mask & answer_mask

    @torch.no_grad()
    def teacher_log_probs(self, input_ids, attention_mask, positions):
        outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self._postprocess_outputs(outputs).logits[positions]
        return self._log_probs(logits)

    @torch.no_grad()
    def selective_target(self, log_p_cond, log_p_cf):
        """Build the normalized entity-counterfactual target distribution."""
        entity_score = (log_p_cond.float() - log_p_cf.float()).clamp_min(0.0)
        target_logits = log_p_cf.float() - self.beta * entity_score
        log_q = F.log_softmax(target_logits, dim=-1)
        return log_q.detach(), log_q.exp().detach()

    @torch.no_grad()
    def position_targets(self, log_p_cond, log_p_cf, forget_positions):
        """Route forget positions to q* and all other positions to p_cond."""
        if log_p_cond.shape != log_p_cf.shape:
            raise ValueError("conditional and counterfactual distributions must align")
        if forget_positions.shape != log_p_cond.shape[:-1]:
            raise ValueError("forget_positions must contain one flag per position")
        log_q_selective, q_selective = self.selective_target(
            log_p_cond, log_p_cf
        )
        forget_positions = forget_positions.unsqueeze(-1)
        log_q = torch.where(forget_positions, log_q_selective, log_p_cond.float())
        q = torch.where(forget_positions, q_selective, log_p_cond.float().exp())
        return log_q.detach(), q.detach()

    def sample_selective_state(self, model, forget):
        if self.processing_class.padding_side != "right":
            raise ValueError("selective unlearning requires right padding")

        input_ids = forget["input_ids"]
        labels = forget["labels"]
        attention_mask = forget.get("attention_mask", None)
        if self.eos_pad_to is not None:
            input_ids, labels, attention_mask = self._extend_response_with_eos(input_ids, labels, attention_mask)
        prepared_inputs = {"input_ids": input_ids, "labels": labels}
        if attention_mask is not None:
            prepared_inputs["attention_mask"] = attention_mask
        prepared = self._preprocess_inputs(prepared_inputs)
        input_ids = prepared["input_ids"]
        labels = prepared["labels"]
        attention_mask = prepared.get("attention_mask", None)

        batch_size, sequence_length = input_ids.shape
        answer_mask = labels != -100
        query_entity_mask = self.query_entity_mask(input_ids, answer_mask, attention_mask)
        answer_copy_mask = self.answer_copy_mask(input_ids, answer_mask, query_entity_mask)
        answer_entity_mask = self.answer_entity_mask(input_ids, answer_mask, answer_copy_mask)
        answer_forget_mask = answer_entity_mask & ~answer_copy_mask
        answer_hide_mask = answer_copy_mask | answer_forget_mask
        if self.forget_all:  # entity words stay always hidden; the EOS pad and terminator keep p_cond
            stops = input_ids == self.processing_class.eos_token_id
            terminators = self._get_terminator_tensor(input_ids.device)
            if terminators is not None:
                stops |= torch.isin(input_ids, terminators)
            answer_forget_mask = answer_mask & ~answer_copy_mask & ~stops

        timesteps = self._sample_timesteps(batch_size, input_ids.device)
        mask_probability = 1.0 - self.scheduler(timesteps).unsqueeze(1).expand(batch_size, sequence_length)
        ordinary_mask = (torch.rand((batch_size, sequence_length), device=input_ids.device) < mask_probability) & answer_mask
        effective_mask = ordinary_mask | answer_hide_mask
        noised_input_ids = torch.where(effective_mask, self.mask_id, input_ids)

        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        return {
            "outputs": outputs,
            "logits": outputs.logits,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_mask": answer_mask,
            "query_entity_mask": query_entity_mask,
            "answer_copy_mask": answer_copy_mask,
            "answer_forget_mask": answer_forget_mask,
            "answer_hide_mask": answer_hide_mask,
            "answer_entity_mask": answer_entity_mask,
            "ordinary_mask": ordinary_mask,
            "effective_mask": effective_mask,
            "noised_input_ids": noised_input_ids,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        state          = self.sample_selective_state(model, inputs["forget"])
        masked         = state["effective_mask"]
        noised         = state["noised_input_ids"]
        attention_mask = state["attention_mask"]
        counterfactual = torch.where(state["query_entity_mask"], self.mask_id, noised)
        
        with torch.no_grad():
            log_p_cond       = self.teacher_log_probs(noised, attention_mask, masked)
            log_p_cf         = self.teacher_log_probs(counterfactual, attention_mask, masked)
            forget_positions = state["answer_forget_mask"][masked]
            log_q, q         = self.position_targets(log_p_cond, log_p_cf, forget_positions)

        log_p_student = self._log_probs(state["logits"][masked])
        masked_kl     = (q * (log_q - log_p_student)).sum(-1)
        per_token     = state["logits"][..., 0].float() * 0.0
        per_token     = per_token.masked_scatter(masked, masked_kl)
        forget_mask   = masked & state["answer_forget_mask"]
        preserve_mask = masked & ~state["answer_forget_mask"]
        forget_loss   = self._reduce(per_token, forget_mask, torch.ones_like(per_token))
        preserve_loss = self._reduce(per_token, preserve_mask, torch.ones_like(per_token))
        loss          = forget_loss + self.preserve_weight * preserve_loss
        return (loss, state["outputs"]) if return_outputs else loss
