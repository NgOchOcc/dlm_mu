"""Trajectory-aligned selective unlearning for masked diffusion LMs.

Implements ``docs/traj_iproj_algorithms.tex``. The forget target is an
I-projection of the frozen teacher that caps its belief mass on the answer, and
the states it is applied at come from the decoder rather than from forward
corruption. Relative to ``traj_rmdu.py`` (whose target is a logit-space residual
against the prompt-masked anchor) three things change, and each is an ablation
switch so the whole matrix is one config sweep:

* **the cap is driven by the runner-up, not by an anchor** (``kappa``,
  ``margin_cap``). Eval decoding is greedy, so a target that lowers the mass on the
  answer without moving the argmax measures as zero forgetting. On a memorised
  checkpoint the teacher holds ``b(y) ~ 0.999`` with a runner-up near ``3e-4``, so
  the argmax only moves once the cap reaches ~0.23 -- well below the ~0.4-0.5 mass
  the prompt-masked anchor carries. ``margin_cap=false`` recovers the pure
  knowledge-cap variant and is expected to be a no-op. ``kappa=1`` is the equality
  boundary, where the leading forbidden token ties the best allowed one and the
  argmax is decided by index order, so the default is strictly below it.
* **the forbidden set is a string, not a token id** (``spell_max_ids``). See
  :class:`~trainer.dlm.unlearn.iproj_core.SpellingTable`.
* **two anchors** (``knowledge_anchor``). The *context* anchor shares the state and
  defines the lift, so it says *where* the prompt still matters; the *knowledge*
  anchor also masks the answer, so it cannot read the fact off a partially revealed
  state and can therefore set the cap *level*. Because it does not depend on the
  state it is cached, and a step still costs three forwards.

The state distribution is a three-way mixture (``lambda_traj``, ``lambda_on``):
teacher-forced decoder trajectory, detached rollout of the current student, and
forward corruption. The on-policy component exists because erasure fitted on a
fixed pool of states repairs those states and does not transfer -- the measured
(position, token) overlap between a frozen-mined state and a reworded-frame state
is zero for about half of TOFU records.

The forget loss normalizes active and inactive masked positions separately
(Eq. 13) so sweeping ``eta`` does not silently rescale the erasure signal, and the
retain term is ``KL(p_0 || p_theta)`` rather than a gold NLL: the finetuned
checkpoint already fits the retain split, so its retain NLL is ~1e-4 and looks like
protection in the logs while providing none.
"""

import math

import torch

from trainer.dlm.unlearn.grad_diff import DiffusionGradDiff
from trainer.dlm.unlearn.iproj_core import (
    SpellingTable,
    active_positions,
    allowed_vocab_mask,
    belief_set,
    compose_cap,
    masked_stats,
    project,
)
from trainer.dlm.unlearn.rollout import (
    reverse_trajectory_state,
    teacher_forced_reveal_order,
)


class DiffusionTrajIProj(DiffusionGradDiff):
    def __init__(
        self,
        eta=0.3,
        kappa=0.5,
        rho=1.0,
        top_k=32,
        delta_tok=0.5,
        prob_floor=1e-4,
        margin_cap=True,
        knowledge_anchor="cached",
        anchor_top_k=64,
        gamma_local=1.0,
        kl_direction="forward",
        lambda_traj=0.5,
        lambda_on=0.0,
        trajectory_steps=0,
        rollout_steps=32,
        rollout_temperature=1.0,
        spell_max_ids=80,
        spell_variants=True,
        mask_special_tokens=True,
        retain_loss_type="KL",
        cache_trajectories=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **{**kwargs, "retain_loss_type": "NLL"})
        if kl_direction not in ("forward", "kd"):
            raise ValueError(f"Unknown kl_direction: {kl_direction}")
        if knowledge_anchor not in ("cached", "forward", "off"):
            raise ValueError(f"Unknown knowledge_anchor: {knowledge_anchor}")
        if retain_loss_type not in ("KL", "NLL"):
            raise ValueError(f"Unknown retain_loss_type: {retain_loss_type}")
        if not 0.0 < eta <= 1.0:
            raise ValueError("eta must lie in (0, 1]")
        if lambda_traj + lambda_on > 1.0:
            raise ValueError("lambda_traj + lambda_on must not exceed 1")

        self.eta = eta
        self.kappa = kappa
        self.rho = rho
        self.top_k = top_k
        self.delta_tok = delta_tok
        self.log_prob_floor = math.log(prob_floor)
        self.margin_cap = margin_cap
        self.knowledge_anchor = knowledge_anchor
        self.anchor_top_k = anchor_top_k
        self.gamma_local = gamma_local
        self.kl_direction = kl_direction
        self.lambda_traj = lambda_traj
        self.lambda_on = lambda_on
        self.trajectory_steps = trajectory_steps
        self.rollout_steps = rollout_steps
        self.rollout_temperature = rollout_temperature
        self.retain_loss_type = retain_loss_type
        self.cache_trajectories = cache_trajectories

        self.mask_id = self.processing_class.mask_token_id
        self.vocab_size = len(self.processing_class)
        self.ref_model = self._prepare_ref_model(self.model)
        self.spelling = SpellingTable(
            self.processing_class,
            self.vocab_size,
            max_ids=spell_max_ids,
            strip_leading_punct=spell_variants,
        )
        self.allowed = (
            allowed_vocab_mask(
                self.processing_class, self.vocab_size, self.accelerator.device
            )
            if mask_special_tokens
            else None
        )
        self._reveal_cache = {}
        self._spell_cache = {}
        self._anchor_cache = {}
        self._pending_metrics = {}

    # -- cross-rank plumbing --------------------------------------------------
    def _sync_max(self, values):
        """Element-wise max of ``values`` across ranks.

        Every entry decides how many frozen forwards the batch issues, and each
        forward is a ZeRO-3 collective, so the decision has to be identical on
        every rank even though cache state and answer lengths are not.
        """
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return values
        buffer = torch.tensor(values, device=self.accelerator.device, dtype=torch.long)
        torch.distributed.all_reduce(buffer, op=torch.distributed.ReduceOp.MAX)
        return buffer.tolist()

    @staticmethod
    def _row_keys(*tensors):
        return [
            tuple(row.cpu().numpy().tobytes() for row in rows)
            for rows in zip(*tensors)
        ]

    # -- state construction ---------------------------------------------------
    def _reveal_order(self, input_ids, answer_mask, attention_mask):
        """Step at which the deployment decoder reveals each answer position.

        Every revealed token is gold, so a state is fully determined by its mask
        set and the whole trajectory compresses into this one integer tensor,
        cacheable across epochs because the frozen model makes it deterministic.
        """
        keys = self._row_keys(input_ids, answer_mask)
        missing = any(key not in self._reveal_cache for key in keys)
        any_missing, longest = self._sync_max(
            [int(missing), int(answer_mask.sum(-1).max())]
        )
        if any_missing:
            steps = self.trajectory_steps if self.trajectory_steps > 0 else longest
            order = teacher_forced_reveal_order(
                self.ref_model,
                input_ids,
                answer_mask,
                attention_mask,
                max(steps, 1),
                self.mask_id,
                self.right_shift_logits,
                self.vocab_size,
            )
            if not self.cache_trajectories:
                return order
            for key, row in zip(keys, order):
                self._reveal_cache[key] = row.cpu()
        return torch.stack([self._reveal_cache[k] for k in keys]).to(input_ids.device)

    def _trajectory_masked(self, input_ids, answer_mask, attention_mask):
        """Masked positions of a state drawn uniformly along the trajectory.

        State ``k`` masks whatever is revealed at step ``k`` or later, so ``k = 0``
        is the fully masked answer. Drawing ``k`` strictly below the per-row number
        of reveal steps excludes the fully revealed state, which has no masked
        position and hence no forget loss.
        """
        order = self._reveal_order(input_ids, answer_mask, attention_mask)
        states = order.masked_fill(~answer_mask, -1).amax(-1) + 1
        step = (torch.rand_like(states, dtype=torch.float) * states).long()
        return answer_mask & (order >= step.unsqueeze(-1))

    def _forward_masked(self, input_ids, answer_mask):
        batch, length = input_ids.shape
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            batch, device=input_ids.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(batch, length)
        return (
            torch.rand((batch, length), device=input_ids.device) < p_mask
        ) & answer_mask

    def _build_state(self, model, input_ids, answer_mask, attention_mask):
        """The three-way state mixture of Algorithm 2.

        Each source is computed for the whole batch and then mixed in per row:
        which row wants which source is rank-dependent, so branching on it would
        desynchronise the ZeRO-3 collectives inside the rollouts.
        """
        masked = self._forward_masked(input_ids, answer_mask)
        state = torch.where(masked, self.mask_id, input_ids)
        draw = torch.rand(input_ids.shape[0], device=input_ids.device)

        if self.lambda_traj > 0.0:
            use = (draw < self.lambda_traj).unsqueeze(-1)
            traj = self._trajectory_masked(input_ids, answer_mask, attention_mask)
            masked = torch.where(use, traj, masked)
            state = torch.where(use, torch.where(traj, self.mask_id, input_ids), state)

        if self.lambda_on > 0.0:
            use = (
                (draw >= self.lambda_traj)
                & (draw < self.lambda_traj + self.lambda_on)
            ).unsqueeze(-1)
            on_state, on_masked, _ = reverse_trajectory_state(
                model,
                input_ids,
                answer_mask,
                attention_mask,
                max(self.rollout_steps, 2),
                self.mask_id,
                self.right_shift_logits,
                self.vocab_size,
                temperature=self.rollout_temperature,
            )
            masked = torch.where(use, on_masked, masked)
            state = torch.where(use, on_state, state)

        return state, masked

    def _null_input_ids(self, state, answer_mask, attention_mask):
        """``state`` with the prompt replaced by mask tokens (the null prompt m)."""
        prompt = ~answer_mask
        if attention_mask is not None:
            prompt = prompt & attention_mask.bool()
        return torch.where(prompt, self.mask_id, state)

    # -- knowledge anchor -----------------------------------------------------
    def _knowledge_topk(self, input_ids, answer_mask, attention_mask):
        """Cached top-``anchor_top_k`` of ``p_0(. | m, z_0)`` (values, indices).

        ``z_0`` masks the whole answer and ``m`` masks the whole prompt, so this
        distribution is independent of the training state -- which is exactly why it
        can set the cap level where the context anchor cannot, and why one forward
        per distinct padded shape suffices.
        """
        null_ids = torch.where(
            answer_mask, torch.full_like(input_ids, self.mask_id), input_ids
        )
        null_ids = self._null_input_ids(null_ids, answer_mask, attention_mask)
        keys = self._row_keys(null_ids, answer_mask)
        missing = any(key not in self._anchor_cache for key in keys)
        (any_missing,) = self._sync_max([int(missing)])
        if any_missing:
            with torch.no_grad():
                logits = self._postprocess_outputs(
                    self.ref_model(input_ids=null_ids, attention_mask=attention_mask)
                ).logits
                values, index = self._deployment_log_probs(logits).topk(
                    self.anchor_top_k, -1
                )
            for key, value, idx in zip(keys, values, index):
                self._anchor_cache[key] = (value.cpu(), idx.cpu())
        cached = [self._anchor_cache[key] for key in keys]
        device = input_ids.device
        return (
            torch.stack([v for v, _ in cached]).to(device),
            torch.stack([i for _, i in cached]).to(device),
        )

    def _anchor_mass(self, forbidden, values, index):
        """``a_know(B_i)`` summed over the cached slice only.

        Restricting the sum to the cached top-``K`` makes it a lower bound, so the
        knowledge cap can only come out tighter than the truth, never looser. The
        anchor is near-deterministic in practice, so the omitted tail is negligible;
        ``knowledge_anchor=forward`` computes it exactly as an ablation.
        """
        hit = forbidden.gather(-1, index)
        return (values.exp() * hit.to(values.dtype)).sum(-1)

    def _spell_ids(self, input_ids, answer_mask):
        """``(b, l, max_ids)`` spell ids, cached per row.

        Pure tokenizer work: deterministic given the row and free of collectives,
        so unlike the trajectory cache this one needs no cross-rank agreement.
        """
        keys = self._row_keys(input_ids, answer_mask)
        for key, ids, mask in zip(keys, input_ids, answer_mask):
            if key not in self._spell_cache:
                self._spell_cache[key] = self.spelling.row_spell_ids(
                    ids.tolist(), mask.tolist()
                )
        return torch.stack([self._spell_cache[k] for k in keys]).to(input_ids.device)

    # -- losses ---------------------------------------------------------------
    def _deployment_log_probs(self, logits):
        """Log-probabilities renormalized over the ids the eval decoder may emit.

        Both sides of every KL below have to live on this support. Restricting only
        the teacher makes ``KL(p_theta || q*)`` infinite the moment the student holds
        any mass on ``[MASK]`` or ``<|endoftext|>`` -- which it does at
        initialization -- and the infinity then meets a zero weight in
        :meth:`_reduce` and surfaces as a NaN loss on the first step.
        """
        log_p = self._log_probs(logits)
        if self.allowed is None:
            return log_p
        return log_p.masked_fill(~self.allowed, float("-inf")).log_softmax(-1)

    def _kl(self, log_p, log_q):
        """``sum_v p(v) [log p(v) - log q(v)]`` on the deployment vocabulary.

        Direction-free -- callers order the arguments.

        Disallowed columns carry ``-inf`` on both sides and are neutralized
        *before* the subtraction, not after: masking the finished summand fixes the
        value but not the gradient, because autograd still evaluates
        ``d/dlog p = exp(log p)(log p - log q) + exp(log p)`` there, and
        ``0 * (-inf + inf) = nan`` poisons the backward through the zero mask.
        """
        if self.allowed is not None:
            zero = torch.zeros((), dtype=log_p.dtype, device=log_p.device)
            log_p = torch.where(self.allowed, log_p, zero)
            log_q = torch.where(self.allowed, log_q, zero)
        return (log_p.exp() * (log_p - log_q)).sum(-1)

    def _forget_kl(self, log_p, log_q):
        """Eq. 13 ``KL(p_theta || q*)``, or Eq. 14 ``KL(q* || p_theta)`` under ``kd``.

        The ``kd`` arm is mass-covering: it only asks the student to *cover* the
        residual mass the cap leaves on ``B_i`` and never penalizes it for keeping the
        gold token on top, so it is predicted to under-forget.
        """
        if self.kl_direction == "kd":
            return self._kl(log_q, log_p)
        return self._kl(log_p, log_q)

    def _forget_loss(self, model, forget_inputs):
        prep = self._preprocess_inputs(
            {
                "input_ids": forget_inputs["input_ids"],
                "labels": forget_inputs["labels"],
                "attention_mask": forget_inputs.get("attention_mask", None),
            }
        )
        input_ids, labels = prep["input_ids"], prep["labels"]
        attention_mask = prep.get("attention_mask", None)
        answer_mask = labels != -100

        state, masked = self._build_state(
            model, input_ids, answer_mask, attention_mask
        )
        gold = input_ids.unsqueeze(-1)

        with torch.no_grad():
            log_b = self._deployment_log_probs(
                self._postprocess_outputs(
                    self.ref_model(input_ids=state, attention_mask=attention_mask)
                ).logits
            )
            topk_log_b, topk_index = log_b.topk(self.top_k, -1)
            log_b_gold = log_b.gather(-1, gold).squeeze(-1)

            # The context anchor is only ever read at the gold token and at the
            # top-K slice, so it is gathered and dropped before the belief set
            # allocates: keeping the dense difference would double peak memory.
            log_a = self._deployment_log_probs(
                self._postprocess_outputs(
                    self.ref_model(
                        input_ids=self._null_input_ids(
                            state, answer_mask, attention_mask
                        ),
                        attention_mask=attention_mask,
                    )
                ).logits
            )
            topk_log_a = log_a.gather(-1, topk_index)
            residual = (log_b_gold - log_a.gather(-1, gold).squeeze(-1)).relu()
            del log_a

            active = active_positions(residual, masked, self.eta)
            forbidden = belief_set(
                topk_index,
                topk_log_b,
                topk_log_a,
                self._spell_ids(input_ids, answer_mask),
                input_ids,
                masked,
                self.vocab_size,
                self.delta_tok,
                self.log_prob_floor,
                self.allowed,
            )
            log_s, log_1ms, log_mu, log_lead = masked_stats(log_b, forbidden)

            anchor_mass = None
            if self.knowledge_anchor != "off":
                values, index = self._knowledge_topk(
                    input_ids, answer_mask, attention_mask
                )
                anchor_mass = self._anchor_mass(forbidden, values, index)

            caps = compose_cap(
                log_s,
                log_1ms,
                log_mu,
                log_lead,
                anchor_mass,
                rho=self.rho,
                kappa=self.kappa,
                margin_cap=self.margin_cap,
            )
            log_q, diag = project(
                log_b, forbidden, caps["cap"], caps["log_s"], caps["log_1ms"], active
            )

        outputs = self._postprocess_outputs(
            model(input_ids=state, attention_mask=attention_mask)
        )
        log_p = self._deployment_log_probs(outputs.logits)
        kl = self._forget_kl(log_p, log_q)

        ones = torch.ones_like(kl)
        erasure = self._reduce(kl, active, ones)
        local = self._reduce(kl, masked & ~active, ones)
        loss = self.gamma * erasure + self.gamma_local * local

        metrics = {}
        if model.training:
            with torch.no_grad():
                # KL to the unprojected teacher, at the same positions: the
                # difference between the two is the entire contribution of the
                # projection, and Eq. 17 predicts it is ~log(s/c).
                baseline = self._reduce(self._forget_kl(log_p, log_b), active, ones)
                metrics = self._metrics(
                    erasure, local, baseline, caps, diag, masked, active, answer_mask
                )
        return loss, outputs, metrics

    def _metrics(self, erasure, local, baseline, caps, diag, masked, active, answer):
        def mean_on(values, mask):
            mask = mask.to(torch.float64)
            return float(
                (values.double() * mask).sum() / mask.sum().clamp_min(1)
            )

        def cap_component(values):
            """Mean of one candidate cap and how often it is the operative one.

            ``c = min(s, c_know, c_marg)`` hides which constraint actually binds, and
            the answer decides whether an arm tests what it claims to: with the
            knowledge cap at ~0 the margin cap never binds and ``kappa`` is a dead
            parameter. ``compose_cap`` fills a disabled constraint with NaN, so an arm
            that turns one off reports NaN here rather than a misleading zero.
            """
            if torch.isnan(values).all():
                return float("nan"), float("nan")
            finite = active & ~values.isnan()
            return (
                mean_on(values.nan_to_num(), finite),
                mean_on(torch.isclose(values, caps["cap"]) & finite, active),
            )

        cap_know, binds_know = cap_component(caps["cap_knowledge"])
        cap_marg, binds_marg = cap_component(caps["cap_margin"])

        binds = diag["binds"]
        return {
            "iproj/erasure_kl": float(erasure.detach()),
            "iproj/local_kl": float(local.detach()),
            "iproj/teacher_kl": float(baseline.detach()),
            "iproj/kl_gain": float((erasure - baseline).detach()),
            "iproj/log_ratio": mean_on(diag["log_ratio"], binds),
            "iproj/argmax_escaped": mean_on(diag["argmax_escaped"], active),
            "iproj/cap": mean_on(caps["cap"], active),
            "iproj/cap_binds": mean_on(binds, active),
            "iproj/cap_knowledge": cap_know,
            "iproj/cap_margin": cap_marg,
            "iproj/binds_knowledge": binds_know,
            "iproj/binds_margin": binds_marg,
            "iproj/belief_mass": mean_on(caps["mass"], active),
            "iproj/mu_bar": mean_on(caps["mu_bar"], active),
            "iproj/active_rate": float(
                active.sum() / masked.sum().clamp_min(1)
            ),
            "iproj/mask_rate": float(masked.sum() / answer.sum().clamp_min(1)),
        }

    def _retain_loss(self, model, retain_inputs):
        """``KL(p_0 || p_theta)`` on a forward-corrupted retain state (Eq. 15).

        A distributional constraint, not a likelihood one: the finetuned checkpoint
        already fits the retain split, so its gold NLL is ~1e-4 and carries almost
        no gradient. ``retain_loss_type=NLL`` restores that term for the ablation.
        """
        if self.retain_loss_type == "NLL":
            return self.retain_nll(model, retain_inputs)

        art = self._sample_and_denoise(
            model,
            retain_inputs["input_ids"],
            retain_inputs["labels"],
            retain_inputs.get("attention_mask", None),
        )
        with torch.no_grad():
            log_ref = self._deployment_log_probs(
                self._postprocess_outputs(
                    self.ref_model(
                        input_ids=art["noised_input_ids"],
                        attention_mask=art["attention_mask"],
                    )
                ).logits
            )
        log_p = self._deployment_log_probs(art["logits"])
        kl = self._kl(log_ref, log_p)
        return self._reduce(kl, art["masked_mask"], torch.ones_like(kl))

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_loss, outputs, metrics = self._forget_loss(model, inputs["forget"])
        retain_loss = self._retain_loss(model, inputs["retain"])
        loss = forget_loss + self.alpha * retain_loss

        if model.training:
            metrics["iproj/retain_loss"] = float(retain_loss.detach())
            metrics["iproj/cached_states"] = float(len(self._reveal_cache))
            self._pending_metrics.update(metrics)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if (
            model.training
            and self.accelerator.sync_gradients
            and self._pending_metrics
        ):
            self.log(self._pending_metrics)
            self._pending_metrics = {}
        return loss
