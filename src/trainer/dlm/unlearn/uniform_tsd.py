"""Uniform-target variant of SimpleTSD for masked-diffusion LM unlearning.

``DiffusionSimpleTSD`` fits the student at the selected forget positions ``A_hat`` to
the *edited teacher* ``q`` -- the with-question teacher ``p0(.|x, y_M)`` with the gold
token (and, under belief demotion, the teacher top-K) removed and renormalised. That
forward KL forces the student to reproduce the teacher's exact ranking over every
surviving token, so the hidden state must still encode the concept in order to rebuild
that distribution: the fact survives in representation space even though the gold token
is demoted at the output.

This variant instead fits the student at ``A_hat`` to the **uniform distribution**
``u = 1/|V_lex|`` over the inherited non-special vocabulary support, decoupled
from the teacher entirely:

    L_f = KL( u || p_theta(.|x, y_M) )
        = -log |V_lex| - mean_{v in V_lex} log p_theta(v)   on A_hat.

The student is told only "every token is equally likely here", so there is no incentive
to retain the discriminative structure that pins down the concept -- a strictly less
concept-preserving target than demotion. Note this is an *output-space* proxy: a uniform
output does not by itself guarantee the concept is gone from the hidden state (the shared
unembedding could still map an informative ``h`` to flat logits), but since the
unembedding is fixed, driving the logits uniform across contexts pushes ``h`` toward an
uninformative point in practice.

``mix_beta`` interpolates in probability space between the edited-teacher target ``q``
(``mix_beta=0``, exactly ``DiffusionSimpleTSD``) and the uniform target ``u``
(``mix_beta=1``, the default -- pure flat). Values in between,
``m = (1 - mix_beta) q + mix_beta u``, soften the erasure to trade forgetting for
fluency at the forget positions. The selection / masking / self-distillation (on
``R = A \\ A_hat``) / weighting pipeline is inherited unchanged from the base.
"""

import math

import torch

from trainer.dlm.unlearn.simple_tsd import DiffusionSimpleTSD


class DiffusionUTSD(DiffusionSimpleTSD):
    def __init__(self, mix_beta=1.0, *args, **kwargs):
        # Validate the cheap scalar before super().__init__ builds the frozen ref model.
        if not (0.0 <= float(mix_beta) <= 1.0):
            raise ValueError(f"mix_beta must be in [0, 1], got {mix_beta}")
        super().__init__(*args, **kwargs)
        self.mix_beta = float(mix_beta)

    def _forget_kl(self, log_p, log_p0, y_m, ctx):
        """Forget target = uniform ``u = 1/|V_lex|`` (``mix_beta=1``), or the
        probability-space mix ``m = (1 - mix_beta) q + mix_beta u`` with the edited
        teacher ``q`` otherwise.

        Returns ``(kl_f, removed)``, each ``[b, l]``. ``removed`` reports the teacher's raw
        gold mass above the target's gold mass -- the mass this objective erases.
        """
        input_ids = ctx["input_ids"]
        gold = input_ids.unsqueeze(-1)
        allowed = ~self._forbidden_unlearning_vocab_mask(log_p.device)
        V = int(allowed.sum())
        u = 1.0 / V
        log_p_forget = self._condition_on_unlearning_vocabulary(log_p)
        log_p0_forget = self._condition_on_unlearning_vocabulary(log_p0)
        p0_gold = log_p0_forget.gather(-1, gold).squeeze(-1).exp()

        if self.mix_beta >= 1.0:
            # Pure uniform: KL(u || p_theta) = sum_v u (log u - log p) = -log V - mean_v log p.
            # No teacher term -> the target is fully decoupled from p0.
            kl_f = -math.log(V) - log_p_forget[..., allowed].mean(dim=-1)
            gold_target = allowed[input_ids].to(p0_gold.dtype) * u
        else:
            # Edited teacher q (demote gold, and belief top-K, then renormalise), as in the
            # base, mixed with uniform in probability space: m = (1 - beta) q + beta u.
            demote = self._demote_mask(log_p0_forget, input_ids)
            log_q = log_p0_forget.masked_fill(demote, float("-inf"))
            log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
            q = log_q.exp()  # zero on the demoted set
            uniform = allowed.to(q.dtype) * u
            m = (1.0 - self.mix_beta) * q + self.mix_beta * uniform
            support = m > 0
            log_m = torch.where(support, m.clamp_min(1e-12).log(), torch.zeros_like(m))
            safe_log_p = log_p_forget.masked_fill(~support, 0.0)
            kl_f = (m * (log_m - safe_log_p)).sum(-1)
            gold_target = m.gather(-1, gold).squeeze(-1)

        removed = (p0_gold - gold_target).clamp_min(0.0)
        # Verification diagnostics (read by _draw_loss): gold mass the teacher remembers vs
        # the (flat) target the student is fit to. gold_prob_target ~ 1/V under pure uniform.
        with torch.no_grad():
            self._forget_diag = {
                "gold_prob_teacher_raw": p0_gold,
                "gold_prob_target": gold_target,
            }
        return kl_f, removed
