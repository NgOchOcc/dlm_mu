"""Entity-counterfactual selective unlearning for masked-diffusion LMs.

The frozen teacher sees one entity-masked answer state twice: once with the
original query and once with entity-bearing query tokens masked.  The latter is
the structural anchor, while their positive log-probability difference selects
entity-dependent vocabulary mass to suppress::

    score(v) = relu(log p_cond(v) - log p_cf(v))
    q*(v) propto p_cf(v) * exp(-beta * score(v))

The student sees the original query and minimizes ``KL(q* || p_theta)`` on the
union of the ordinary diffusion mask and the answer-entity mask.
"""

import math
import re

import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.base import DiffusionUnlearnTrainer


_SPECIAL_RE = re.compile(r"<\|?[^<>]*\|?>")


def _group_word_indices(surfaces):
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


def _content_fragments(surface, min_length=2):
    surface = _SPECIAL_RE.sub(" ", surface).lower()
    return [
        fragment
        for fragment in re.findall(r"[a-z0-9]+", surface)
        if len(fragment) >= min_length
    ]


class DiffusionSelectiveUnlearning(DiffusionUnlearnTrainer):
    """Train against an entity-counterfactual frozen-teacher target."""

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

    def __init__(self, beta=1.0, *args, **kwargs):
        beta = float(beta)
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError(f"beta must be finite and satisfy beta >= 0, got {beta}")

        super().__init__(*args, **kwargs)
        self.beta = beta
        self.mask_id = self.processing_class.mask_token_id
        self.pad_id = self.processing_class.pad_token_id
        self.vocab_size = len(self.processing_class)

        self.ref_model = self._prepare_ref_model(self.model)
        for parameter in self.ref_model.parameters():
            parameter.requires_grad_(False)

    def _prompt_mask(self, input_ids, answer_mask, attention_mask):
        """Return attended, non-answer, non-padding query positions."""
        prompt = (~answer_mask) & (input_ids != self.pad_id)
        if attention_mask is not None:
            prompt = prompt & attention_mask.bool()
        return prompt

    def _question_overlap_mask(self, input_ids, answer_mask, prompt_mask):
        """Return answer words whose normalized form also occurs in the query."""
        tokenizer = self.processing_class
        overlap = torch.zeros_like(answer_mask)
        for batch_index in range(input_ids.shape[0]):
            query_ids = input_ids[batch_index][prompt_mask[batch_index]].tolist()
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
            for group in _group_word_indices(surfaces):
                fragments = _content_fragments(
                    "".join(surfaces[index] for index in group)
                )
                if fragments and all(
                    fragment in query_words for fragment in fragments
                ):
                    for index in group:
                        overlap[batch_index, answer_indices[index]] = True
        return overlap

    def _query_shared_mask(self, input_ids, answer_mask, attention_mask):
        """Return query words also present in the answer (names/shared titles)."""
        tokenizer = self.processing_class
        prompt_mask = self._prompt_mask(input_ids, answer_mask, attention_mask)
        shared = torch.zeros_like(answer_mask)
        for batch_index in range(input_ids.shape[0]):
            answer_ids = input_ids[batch_index][answer_mask[batch_index]].tolist()
            if not answer_ids:
                continue
            answer_text = tokenizer.decode(
                answer_ids, skip_special_tokens=True
            ).lower()
            answer_words = {
                word
                for word in re.findall(r"[a-z0-9]+", answer_text)
                if len(word) >= 2
            }
            if not answer_words:
                continue

            query_indices = prompt_mask[batch_index].nonzero(as_tuple=True)[0]
            if query_indices.numel() == 0:
                continue
            surfaces = [
                tokenizer.decode([int(token_id)])
                for token_id in input_ids[batch_index][query_indices]
            ]
            for group in _group_word_indices(surfaces):
                fragments = _content_fragments(
                    "".join(surfaces[index] for index in group)
                )
                if fragments and all(
                    fragment in answer_words for fragment in fragments
                ):
                    for index in group:
                        shared[batch_index, query_indices[index]] = True
        return shared

    def _query_entity_mask(self, input_ids, answer_mask, attention_mask):
        """ADU-style query entity mask: shared, proper-noun, and numeric words."""
        prompt_mask = self._prompt_mask(input_ids, answer_mask, attention_mask)
        entity_mask = self._query_shared_mask(
            input_ids, answer_mask, attention_mask
        ).clone()
        tokenizer = self.processing_class
        for batch_index in range(input_ids.shape[0]):
            query_indices = prompt_mask[batch_index].nonzero(as_tuple=True)[0]
            if query_indices.numel() == 0:
                continue
            surfaces = [
                tokenizer.decode([int(token_id)])
                for token_id in input_ids[batch_index][query_indices]
            ]
            previous_ended_sentence = True
            for group in _group_word_indices(surfaces):
                word = "".join(surfaces[index] for index in group).strip()
                alpha = re.sub(r"[^A-Za-z]", "", word)
                has_digit = bool(re.search(r"\d", word))
                is_proper = (
                    len(alpha) >= 2
                    and alpha[0].isupper()
                    and not previous_ended_sentence
                    and word.lower() not in self._QUERY_KEEP
                )
                if is_proper or has_digit:
                    for index in group:
                        entity_mask[batch_index, query_indices[index]] = True
                previous_ended_sentence = word.endswith((".", "!", "?", ":", ";"))
        return entity_mask

    def _answer_entity_mask(self, input_ids, answer_mask, attention_mask):
        """ADU-style answer entity mask: query overlap plus proper nouns."""
        prompt_mask = self._prompt_mask(input_ids, answer_mask, attention_mask)
        entity_mask = self._question_overlap_mask(
            input_ids, answer_mask, prompt_mask
        ).clone()
        tokenizer = self.processing_class
        for batch_index in range(input_ids.shape[0]):
            answer_indices = answer_mask[batch_index].nonzero(as_tuple=True)[0]
            if answer_indices.numel() == 0:
                continue
            surfaces = [
                tokenizer.decode([int(token_id)])
                for token_id in input_ids[batch_index][answer_indices]
            ]
            previous_ended_sentence = True
            for group in _group_word_indices(surfaces):
                word = "".join(surfaces[index] for index in group).strip()
                alpha = re.sub(r"[^A-Za-z]", "", word)
                is_proper = (
                    len(alpha) >= 2
                    and alpha[0].isupper()
                    and not previous_ended_sentence
                    and word.lower() not in self._ANSWER_KEEP
                )
                if is_proper:
                    for index in group:
                        entity_mask[batch_index, answer_indices[index]] = True
                previous_ended_sentence = word.endswith((".", "!", "?", ":", ";"))
        return entity_mask & answer_mask

    @torch.no_grad()
    def _teacher_log_probs(self, input_ids, attention_mask, positions):
        outputs = self.ref_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = self._postprocess_outputs(outputs).logits[positions]
        return self._log_probs(logits)

    @torch.no_grad()
    def _selective_target(self, log_p_cond, log_p_cf):
        """Build the normalized entity-counterfactual target distribution."""
        entity_score = (log_p_cond.float() - log_p_cf.float()).clamp_min(0.0)
        target_logits = log_p_cf.float() - self.beta * entity_score
        log_q = F.log_softmax(target_logits, dim=-1)
        return log_q.detach(), log_q.exp().detach()

    def _sample_selective_state(self, model, forget):
        """Sample and evaluate the shared answer state from the method spec."""
        if self.processing_class.padding_side != "right":
            raise ValueError("selective unlearning requires right padding")

        input_ids = forget["input_ids"]
        labels = forget["labels"]
        attention_mask = forget.get("attention_mask", None)
        if self.eos_pad_to is not None:
            input_ids, labels, attention_mask = self._extend_response_with_eos(
                input_ids, labels, attention_mask
            )
        prepared_inputs = {"input_ids": input_ids, "labels": labels}
        if attention_mask is not None:
            prepared_inputs["attention_mask"] = attention_mask
        prepared = self._preprocess_inputs(prepared_inputs)
        input_ids = prepared["input_ids"]
        labels = prepared["labels"]
        attention_mask = prepared.get("attention_mask", None)

        batch_size, sequence_length = input_ids.shape
        answer_mask = labels != -100
        query_entity_mask = self._query_entity_mask(
            input_ids, answer_mask, attention_mask
        )
        answer_entity_mask = self._answer_entity_mask(
            input_ids, answer_mask, attention_mask
        )

        timesteps = self._sample_timesteps(batch_size, input_ids.device)
        mask_probability = 1.0 - self.scheduler(timesteps).unsqueeze(1).expand(
            batch_size, sequence_length
        )
        ordinary_mask = (
            torch.rand((batch_size, sequence_length), device=input_ids.device)
            < mask_probability
        ) & answer_mask
        effective_mask = ordinary_mask | answer_entity_mask
        noised_input_ids = torch.where(
            effective_mask, self.mask_id, input_ids
        )

        outputs = model(
            input_ids=noised_input_ids, attention_mask=attention_mask
        )
        outputs = self._postprocess_outputs(outputs)
        return {
            "outputs": outputs,
            "logits": outputs.logits,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_mask": answer_mask,
            "query_entity_mask": query_entity_mask,
            "answer_entity_mask": answer_entity_mask,
            "ordinary_mask": ordinary_mask,
            "effective_mask": effective_mask,
            "noised_input_ids": noised_input_ids,
        }

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        state = self._sample_selective_state(model, inputs["forget"])
        masked = state["effective_mask"]
        noised = state["noised_input_ids"]
        attention_mask = state["attention_mask"]

        counterfactual = torch.where(
            state["query_entity_mask"], self.mask_id, noised
        )
        with torch.no_grad():
            log_p_cond = self._teacher_log_probs(
                noised, attention_mask, masked
            )
            log_p_cf = self._teacher_log_probs(
                counterfactual, attention_mask, masked
            )
            log_q, q = self._selective_target(log_p_cond, log_p_cf)

        log_p_student = self._log_probs(state["logits"][masked])
        masked_kl = (q * (log_q - log_p_student)).sum(-1)
        per_token = state["logits"][..., 0].float() * 0.0
        per_token = per_token.masked_scatter(masked, masked_kl)
        loss = self._reduce(per_token, masked, torch.ones_like(per_token))
        return (loss, state["outputs"]) if return_outputs else loss
