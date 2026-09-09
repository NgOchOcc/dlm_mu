"""Base class for masked-diffusion unlearning trainers.

DLM counterpart of ``trainer/unlearn/base.py``. Whereas the autoregressive
unlearning trainers get their per-example loss from ``model(**inputs).loss``,
the DLM trainers build their objective from the masked-diffusion NLL defined in
``trainer/dlm/base.py::DiffusionFinetuneTrainer``. This class only adds the
plumbing shared across the concrete methods (reference-model preparation,
DeepSpeed sharding of the reference model, the per-example forget/retain
diffusion NLL) plus the pieces the rollout-based methods (DPDU, TIDE-U) share:
training-state construction and the masked-position loss reduction.
"""

import copy

import torch
import torch.nn.functional as F

from trainer.dlm.base import DiffusionFinetuneTrainer
from trainer.dlm.unlearn.rollout import (
    complete_states,
    decode_answers,
    reverse_trajectory_state,
)
from trainer.unlearn.base import UnlearnTrainer


class DiffusionUnlearnTrainer(DiffusionFinetuneTrainer, UnlearnTrainer):
    """Shared behaviour for DLM forget/retain unlearning trainers.

    MRO: ``DiffusionUnlearnTrainer -> DiffusionFinetuneTrainer ->
    UnlearnTrainer -> FinetuneTrainer``. ``compute_loss``/``prediction_step``
    resolve to the diffusion implementations, while ``_prepare_deepspeed``
    comes from ``UnlearnTrainer``.
    """

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def forget_nll(self, model, forget_inputs, return_outputs=False):
        return self.diffusion_nll(
            model,
            forget_inputs["input_ids"],
            forget_inputs["labels"],
            forget_inputs.get("attention_mask", None),
            return_outputs=return_outputs,
        )

    def retain_nll(self, model, retain_inputs, return_outputs=False):
        return self.diffusion_nll(
            model,
            retain_inputs["input_ids"],
            retain_inputs["labels"],
            retain_inputs.get("attention_mask", None),
            return_outputs=return_outputs,
        )

    def optional_retain_loss(self, model, inputs):
        """Retain diffusion NLL, or ``0.0`` when no retain batch is present.

        Forget-only runs remove the retain dataset entirely (``ForgetRetainDataset``
        with ``retain=None``), so ``inputs`` carries just the ``"forget"`` key. The
        forget/retain trainers call this instead of indexing ``inputs["retain"]``
        directly so the retain term drops out cleanly (no forward pass) rather than
        raising ``KeyError``. Behaviour is unchanged when a retain batch is present.
        """
        retain_inputs = inputs.get("retain", None)
        if retain_inputs is None:
            return 0.0
        return self.retain_nll(model, retain_inputs)

    def _per_example_nll(
        self, logits, input_ids, loss_weights, masked_mask, maskable_mask
    ):
        """Per-example weighted diffusion NLL, averaged over answer tokens.

        Shared by DiffusionNPO (policy vs reference) and DiffusionSimNPO
        (reference-free). Operates on artifacts from ``_sample_and_denoise`` so
        callers can compare policy/reference on an identical noised sequence.
        """
        token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)
        return token_nll.sum(-1) / maskable_mask.sum(-1).clamp_min(1)

    # -- shared plumbing for the rollout-based methods (DPDU, TIDE-U) ------------
    def _valid_mask(self, state):
        if state["attention_mask"] is None:
            return torch.ones_like(state["answer_mask"])
        return state["attention_mask"].bool()

    def _build_state(self, model, forget_inputs):
        """A partially masked state per example, mixing the two state sources.

        Both rollout-based methods train on states drawn either from forward
        masking of the gold answer or from a genuine reverse trajectory whose
        revealed tokens are model-generated -- the second source exists because
        forward-masked gold answers never appear at inference time, and a target
        fitted only on them transfers poorly.

        Requires the caller to set ``trajectory_state_fraction``,
        ``trajectory_source``, ``trajectory_steps``, ``rollout_temperature``,
        ``mask_id`` and ``vocab_size``.
        """
        prep = self._preprocess_inputs(
            {
                "input_ids": forget_inputs["input_ids"],
                "labels": forget_inputs["labels"],
                "attention_mask": forget_inputs.get("attention_mask", None),
            }
        )
        input_ids, labels = prep["input_ids"], prep["labels"]
        attention_mask = prep.get("attention_mask", None)
        b, l = input_ids.shape
        answer_mask = labels != -100

        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)
        masked = (torch.rand((b, l), device=input_ids.device) < p_mask) & answer_mask
        y_t = torch.where(masked, self.mask_id, input_ids)

        use_traj = (
            torch.rand(b, device=input_ids.device) < self.trajectory_state_fraction
        )
        if self.trajectory_state_fraction > 0.0:
            # Rolled out for the whole batch, then mixed in per row: which rows
            # want a trajectory state is rank-dependent, and skipping the rollout
            # on some ranks would desynchronise the ZeRO-3 collectives.
            source = self.ref_model if self.trajectory_source == "frozen" else model
            traj_x, traj_masked, traj_frac = reverse_trajectory_state(
                source,
                input_ids,
                answer_mask,
                attention_mask,
                self.trajectory_steps,
                self.mask_id,
                self.right_shift_logits,
                self.vocab_size,
                temperature=self.rollout_temperature,
            )
            y_t = torch.where(use_traj.unsqueeze(-1), traj_x, y_t)
            masked = torch.where(use_traj.unsqueeze(-1), traj_masked, masked)
            # t_eff: the timestep whose masking rate matches this state.
            t = torch.where(use_traj, traj_frac.clamp(self.time_epsilon, 1.0), t)

        return {
            "input_ids": input_ids,
            "answer_mask": answer_mask,
            "attention_mask": attention_mask,
            "y_t": y_t,
            "masked": masked,
            "loss_weights": self._compute_loss_weights(t, input_ids, masked),
            "trajectory_fraction": float(use_traj.float().mean()),
        }

    @torch.no_grad()
    def _complete_and_score(self, states, revealable, state, evidence, group_size):
        """``(C_leak, C_deg)`` per rollout, each shaped ``(b, group_size)``.

        ``states`` is a flat ``(b * group_size, l)`` batch whose rows are grouped
        by example; ``revealable`` marks the positions the frozen model may still
        fill in.

        The two costs are read off different spans. Leakage uses the span that
        was masked at ``y_t``, identical for every candidate of an example, so the
        comparison is like for like; a completion always retains the answer tokens
        that were never masked, so scoring against the whole gold answer would
        measure the context instead of the candidate. Degeneracy uses the whole
        answer, fluency being a property of the full output.

        Requires the caller to set ``completion_steps``, ``rollout_temperature``,
        ``rollout_chunk_size``, ``scorer`` and ``ref_model``.
        """
        b = state["y_t"].shape[0]
        attention_mask = state["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(group_size, dim=0)
        answer_mask = state["answer_mask"].repeat_interleave(group_size, dim=0)
        scope = state["masked"].repeat_interleave(group_size, dim=0)

        completed = complete_states(
            self.ref_model,
            states,
            revealable,
            attention_mask,
            self.completion_steps,
            self.mask_id,
            self.right_shift_logits,
            self.vocab_size,
            temperature=self.rollout_temperature,
            chunk_size=self.rollout_chunk_size,
        )
        texts = decode_answers(self.processing_class, completed, answer_mask)
        leak_texts = decode_answers(self.processing_class, completed, scope)
        flat_evidence = [ev for ev in evidence for _ in range(group_size)]
        leak, degeneracy = self.scorer.components(
            flat_evidence, texts, leak_texts=leak_texts
        )
        return (
            leak.to(states.device).view(b, group_size),
            degeneracy.to(states.device).view(b, group_size),
        )

    def _log_probs(self, logits):
        """Policy log-probabilities on the tokenizer vocabulary.

        The LM head is wider than the tokenizer (LLaDA: 126464 vs 126349) and the
        tail ids never decode, so both the target and the policy are normalised
        over the tokenizer's vocabulary only -- otherwise the two distributions
        would live on different supports.
        """
        return F.log_softmax(logits[..., : self.vocab_size].float(), dim=-1)

    def _forbidden_unlearning_token_ids(self):
        """Tokenizer/control ids that must not participate in a forget objective.

        ``all_special_ids`` is not sufficient for the supported chat templates:
        Dream's ``im_end`` and LLaDA's ``eot_id`` may be ordinary added tokens.
        Include the derived assistant-turn terminator as well as the conventional
        BOS/EOS/PAD/MASK ids, and keep only ids on the tokenizer-vocabulary axis.
        """
        cached = getattr(self, "_forbidden_unlearning_token_ids_cache", None)
        if cached is not None:
            return cached

        tokenizer = self.processing_class
        token_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "mask_token_id"):
            token_id = getattr(tokenizer, name, None)
            if token_id is not None:
                token_ids.add(int(token_id))

        terminators = self._get_terminator_tensor(torch.device("cpu"))
        if terminators is not None:
            token_ids.update(int(token_id) for token_id in terminators.tolist())

        cached = tuple(
            sorted(int(i) for i in token_ids if 0 <= int(i) < self.vocab_size)
        )
        self._forbidden_unlearning_token_ids_cache = cached
        return cached

    def _forbidden_unlearning_vocab_mask(self, device):
        """Boolean vocabulary mask for special/control tokens on ``device``."""
        cached = getattr(self, "_forbidden_unlearning_vocab_mask_cache", None)
        if cached is None:
            cached = torch.zeros(self.vocab_size, dtype=torch.bool)
            token_ids = self._forbidden_unlearning_token_ids()
            if token_ids:
                cached[list(token_ids)] = True
            if bool(cached.all()):
                raise ValueError(
                    "unlearning requires at least one non-special tokenizer token"
                )
            self._forbidden_unlearning_vocab_mask_cache = cached
        return cached.to(device)

    def _condition_on_unlearning_vocabulary(self, log_scores):
        """Set forbidden-token mass to zero and renormalize over lexical tokens.

        The input may be logits or already-normalized log probabilities. Returning
        log probabilities with ``-inf`` on the forbidden set makes special tokens
        absent from both the target and policy sides of a forget objective.
        """
        if log_scores.shape[-1] != self.vocab_size:
            raise ValueError(
                f"expected vocabulary axis {self.vocab_size}, got {log_scores.shape[-1]}"
            )
        forbidden = self._forbidden_unlearning_vocab_mask(log_scores.device)
        conditioned = log_scores.float().masked_fill(forbidden, float("-inf"))
        return conditioned - torch.logsumexp(conditioned, dim=-1, keepdim=True)

    def _unlearning_position_mask(self, input_ids):
        """Positions whose clean target is lexical rather than special/control."""
        if bool(((input_ids < 0) | (input_ids >= self.vocab_size)).any()):
            raise ValueError("input_ids contains an id outside the tokenizer vocabulary")
        forbidden = self._forbidden_unlearning_vocab_mask(input_ids.device)
        return ~forbidden[input_ids]

    def _reduce(self, per_token, mask, loss_weights):
        """Mean over examples of the weighted per-position loss on ``mask``."""
        weighted = per_token * loss_weights.float() * mask.to(per_token.dtype)
        count = mask.sum(-1)
        seq_loss = weighted.sum(-1) / count.clamp_min(1)
        active = count > 0
        if bool(active.any()):
            return seq_loss[active].mean()
        return per_token.sum() * 0.0
