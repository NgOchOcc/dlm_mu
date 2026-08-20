"""Trajectory-aligned residual unlearning for masked diffusion LMs.

Implements ``docs/traj_rmdu_training_algorithms.tex``. Relative to MDU
(``mdu.py``) two things change, and the ``mask_source`` / ``target_mode``
switches keep all four cells of the paper's 2x2 ablation in this one trainer:

* **mask_source** -- ``"trajectory"`` draws the state uniformly from the
  teacher-forced decoder trajectory (Alg 1) instead of a Bernoulli forward mask,
  so the states carry the mask topology the model actually meets at decode time.
* **target_mode** -- ``"residual"`` pushes the student *below* the prompt-masked
  anchor on exactly the tokens the prompt promoted,
  ``log q = logsoftmax(log p_u0 - beta * relu(log p_c0 - log p_u0))`` (Alg 2),
  where ``"mdu"`` targets ``logsoftmax(tau * logits_u0)``.

``beta = 0`` makes the residual target equal ``p_u0``, i.e. MDU at ``tau = 1``.

The forget term is the plain mean of the forward KL over masked positions
(Eq. 6), with no ELBO ``w(t)`` factor: trajectory states are not draws from the
forward process, so the factor has no meaning there, and dropping it everywhere
keeps the four ablation cells on one scale. Note this differs from ``mdu.py``,
which follows the repo's ELBO-weighted convention.
"""

import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.grad_diff import DiffusionGradDiff
from trainer.dlm.unlearn.rollout import teacher_forced_reveal_order


class DiffusionTrajRMDU(DiffusionGradDiff):
    def __init__(
        self,
        beta=1.0,
        tau=1.0,
        mask_source="trajectory",
        target_mode="residual",
        trajectory_steps=0,
        cache_trajectories=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mask_source not in ("trajectory", "forward"):
            raise ValueError(f"Unknown mask_source: {mask_source}")
        if target_mode not in ("residual", "mdu"):
            raise ValueError(f"Unknown target_mode: {target_mode}")
        self.beta = beta
        self.tau = tau
        self.mask_source = mask_source
        self.target_mode = target_mode
        self.trajectory_steps = trajectory_steps
        self.cache_trajectories = cache_trajectories
        self.mask_id = self.processing_class.mask_token_id
        self.vocab_size = len(self.processing_class)
        self.ref_model = self._prepare_ref_model(self.model)
        self._reveal_cache = {}
        self._pending_metrics = {}

    # -- state construction ---------------------------------------------------
    def _sync_max(self, values):
        """Element-wise max of ``values`` across ranks.

        Both entries drive how many frozen forwards the batch will issue, and
        each forward is a ZeRO-3 collective, so the decision has to be identical
        on every rank even though the cache state and the answer lengths are not.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return values
        buf = torch.tensor(values, device=self.accelerator.device, dtype=torch.long)
        torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.MAX)
        return buf.tolist()

    def _reveal_order(self, input_ids, answer_mask, attention_mask):
        keys = [
            (row.cpu().numpy().tobytes(), mask.cpu().numpy().tobytes())
            for row, mask in zip(input_ids, answer_mask)
        ]
        missing = any(k not in self._reveal_cache for k in keys)
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

    def _trajectory_mask(self, input_ids, answer_mask, attention_mask):
        """Masked positions of a state drawn uniformly from the trajectory.

        State ``k`` masks the positions revealed at step ``k`` or later, so
        ``k = 0`` is the fully masked answer. Drawing ``k`` below the per-row
        number of reveal steps excludes the fully revealed state, which carries
        no masked position and hence no forget KL.
        """
        order = self._reveal_order(input_ids, answer_mask, attention_mask)
        num_states = order.masked_fill(~answer_mask, -1).amax(-1) + 1
        k = (torch.rand_like(num_states, dtype=torch.float) * num_states).long()
        return answer_mask & (order >= k.unsqueeze(-1))

    def _forward_mask(self, input_ids, answer_mask):
        b, l = input_ids.shape
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)
        return (torch.rand((b, l), device=input_ids.device) < p_mask) & answer_mask

    def _null_input_ids(self, state, answer_mask, attention_mask):
        """``state`` with the prompt replaced by mask tokens (the null prompt m)."""
        prompt_mask = ~answer_mask
        if attention_mask is not None:
            prompt_mask = prompt_mask & attention_mask.bool()
        return torch.where(prompt_mask, self.mask_id, state)

    # -- target and loss ------------------------------------------------------
    def _log_target(self, log_p_c0, log_p_u0):
        if self.target_mode == "mdu":
            return F.log_softmax(self.tau * log_p_u0, dim=-1)
        residual = torch.relu(log_p_c0 - log_p_u0)
        return F.log_softmax(log_p_u0 - self.beta * residual, dim=-1)

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

        if self.mask_source == "trajectory":
            masked = self._trajectory_mask(input_ids, answer_mask, attention_mask)
        else:
            masked = self._forward_mask(input_ids, answer_mask)
        state = torch.where(masked, self.mask_id, input_ids)

        with torch.no_grad():
            cond = self.ref_model(input_ids=state, attention_mask=attention_mask)
            null = self.ref_model(
                input_ids=self._null_input_ids(state, answer_mask, attention_mask),
                attention_mask=attention_mask,
            )
            log_p_c0 = self._log_probs(self._postprocess_outputs(cond).logits)
            log_p_u0 = self._log_probs(self._postprocess_outputs(null).logits)
            log_q = self._log_target(log_p_c0, log_p_u0)

        outputs = self._postprocess_outputs(
            model(input_ids=state, attention_mask=attention_mask)
        )
        log_p = self._log_probs(outputs.logits)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)
        loss = self._reduce(kl, masked, torch.ones_like(kl))

        metrics = {}
        if model.training:
            active = masked.to(kl.dtype)
            count = active.sum().clamp_min(1)
            # Residual at the gold token: how much prompt evidence the target has
            # to remove at this position. Summing it over the vocabulary instead
            # would be dominated by tail mass and swing by an order of magnitude
            # between batches.
            gold = input_ids.unsqueeze(-1)
            residual_gold = (log_p_c0 - log_p_u0).gather(-1, gold).squeeze(-1)
            metrics = {
                "traj_rmdu/forget_kl": float(loss.detach()),
                "traj_rmdu/mask_rate": float(
                    count / answer_mask.sum().clamp_min(1)
                ),
                "traj_rmdu/target_entropy": float(
                    (-(log_q.exp() * log_q).sum(-1) * active).sum() / count
                ),
                "traj_rmdu/student_entropy": float(
                    (-(log_p.exp() * log_p).sum(-1) * active).sum() / count
                ),
                "traj_rmdu/residual_gold": float(
                    (residual_gold.relu() * active).sum() / count
                ),
            }
        return loss, outputs, metrics

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_loss, outputs, metrics = self._forget_loss(model, inputs["forget"])
        retain_loss = self.retain_nll(model, inputs["retain"])
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        if model.training:
            metrics["traj_rmdu/retain_loss"] = float(retain_loss.detach())
            metrics["traj_rmdu/cached_trajectories"] = float(len(self._reveal_cache))
            self._pending_metrics.update(metrics)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if model.training and self.accelerator.sync_gradients and self._pending_metrics:
            self.log(self._pending_metrics)
            self._pending_metrics = {}
        return loss
