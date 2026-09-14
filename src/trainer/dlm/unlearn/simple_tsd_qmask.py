"""Question-masked-target variant of SimpleTSD for masked diffusion LM unlearning.

``DiffusionSimpleTSD`` demotes the gold token at the selected forget positions
``A_hat`` and renormalises the *with-question* teacher ``p0(.|x, y_M)`` -- i.e. it
edits the answer distribution while still conditioning on the question. This
variant instead fits the student at ``A_hat`` to the *question-masked* teacher

    q_i(.) = p0(. | x_empty, y_M),

the promptless distribution the teacher predicts when the whole question span is
blanked out (exactly the ``log_base`` term PCSS averages to score prompt-
dependence, see ``simple_tsd.py`` ``_select_forget_positions``). The forget
objective becomes

    L_f = KL( p0(.|x_empty, y_M) || p_theta(.|x, y_M) )   on A_hat,

so the student -- which *does* see the question -- is trained to answer the
forget tokens as if the question were absent. This drives the very prompt-
dependence PCSS measures to zero at the selected positions, rather than merely
removing the gold token's mass. The retain / self-distillation branch on
``R = A \\ A_hat`` (``KL(p0(.|x, y_M) || p_theta)``) and the whole selection /
masking / balanced-weighting pipeline are inherited unchanged from the base.

Optional ``erase_gold_in_target`` additionally demotes the gold token (and, under
belief demotion, the teacher top-K) from the promptless target before matching,
in case the question-masked teacher still leaks residual gold mass; default off
gives the pure "match the promptless distribution" objective.
"""

import torch

from trainer.dlm.unlearn.simple_tsd import DiffusionSimpleTSD


class DiffusionSimpleTSDQMask(DiffusionSimpleTSD):
    def __init__(self, erase_gold_in_target=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.erase_gold_in_target = erase_gold_in_target

    def _forget_kl(self, log_p, log_p0, y_m, ctx):
        """Forget target = the question-masked teacher ``p0(.|x_empty, y_M)``.

        Blank the question span on the *already-noised* state ``y_M`` (so the answer
        stays masked exactly as the student sees it, only the prompt is removed),
        forward the frozen teacher once, and match the student to that promptless
        distribution with a forward KL. ``removed`` reports the prompt's contribution
        to the gold probability (``p0_gold`` with question minus without), the mass
        this objective is explicitly trying to erase.
        """
        input_ids = ctx["input_ids"]
        attention_mask = ctx["attention_mask"]
        answer_mask = ctx["answer_mask"]

        prompt = self._prompt_mask(input_ids, answer_mask, attention_mask)
        y_base = torch.where(prompt, self.mask_id, y_m)  # x_empty, answer still y_M
        log_t = self._teacher_log_probs(y_base, attention_mask)  # log p0(.|x_empty, y_M)

        if self.erase_gold_in_target:
            demote = self._demote_mask(log_t, input_ids)
            log_t = log_t.masked_fill(demote, float("-inf"))
            log_t = log_t - torch.logsumexp(log_t, dim=-1, keepdim=True)

        # KL(t || p_theta) = sum_v t log t - sum_v t log p_theta, with t = 0 wherever
        # the target was demoted to -inf (0 * -inf masked to 0 before the sum).
        neg = torch.isneginf(log_t)
        t = log_t.exp()
        t_log_t = torch.where(neg, torch.zeros_like(log_t), t * log_t)
        kl_f = t_log_t.sum(-1) - (t * log_p).sum(-1)

        gold = input_ids.unsqueeze(-1)
        removed = (
            log_p0.gather(-1, gold).squeeze(-1).exp()
            - log_t.gather(-1, gold).squeeze(-1).exp()
        ).clamp_min(0.0)
        # Verification diagnostics (read by _draw_loss): gold mass under the raw
        # with-question teacher vs under this promptless target. Unlike the default
        # gold-demoted target, gold_prob_target here is NOT forced to 0 (unless
        # erase_gold_in_target) -- a nonzero value is exactly the residual, prompt-
        # independent gold mass the answer's own revealed tokens still pin down.
        with torch.no_grad():
            self._forget_diag = {
                "gold_prob_teacher_raw": log_p0.gather(-1, gold).squeeze(-1).exp(),
                "gold_prob_target": log_t.gather(-1, gold).squeeze(-1).exp(),
            }
        return kl_f, removed
