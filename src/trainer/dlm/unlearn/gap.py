"""GAP-DLM: retain-free unlearning for masked diffusion LMs (docs/gap_dlm_algorithm.tex).

A frozen teacher is queried twice on the same noised response: conditioned on the
forget prompt (``p^c``) and with the prompt masked out (``p^u``). The gap between
them is the prompt-attributed guidance -- probability mass the prompt specifically
amplifies. GAP suppresses only that mass, projecting the teacher's conditional
distribution onto the closest distribution that caps every prompt-amplified token,
and self-distills everything else. The student is trained with one KL term toward
this projected target; there is no retain set (all non-forget behaviour is carried
by the ``alpha=0`` special case ``q* = p^c``).

MRO note: subclasses ``DiffusionGradDiff`` for the ``gamma``/``alpha``/retain
plumbing and the ``ref_model`` helper, exactly as ``DiffusionMDU`` does. ``alpha``
defaults to 0 (retain-free); set it and supply a retain dataset to add a retain
NLL for ablation.
"""

import torch

from trainer.dlm.unlearn.grad_diff import DiffusionGradDiff


class DiffusionGAP(DiffusionGradDiff):
    def __init__(self, lam=1.0, epsilon=1e-8, copy_ngram=2, alpha=0.0, *args, **kwargs):
        super().__init__(alpha=alpha, *args, **kwargs)
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        if copy_ngram < 1:
            raise ValueError(f"copy_ngram must be a positive integer, got {copy_ngram}")
        self.lam = lam
        self.gap_epsilon = epsilon
        self.copy_ngram = copy_ngram
        self.vocab_size = len(self.processing_class)
        self.ref_model = self._prepare_ref_model(self.model)

    def _null_input_ids(self, art):
        """``y_t`` with the prompt positions replaced by the mask token.

        Same construction as ``DiffusionMDU._null_input_ids``: the null teacher
        must see an identical noised response, only stripped of the forget prompt,
        so that ``p^c`` and ``p^u`` differ solely in the presence of the prompt.
        """
        prompt_mask = ~art["maskable_mask"]
        if art["attention_mask"] is not None:
            prompt_mask = prompt_mask & art["attention_mask"].bool()
        return torch.where(
            prompt_mask, self.processing_class.mask_token_id, art["noised_input_ids"]
        )

    def _copy_safe_indicator(self, art):
        """``kappa`` in {0, 1} per position: 0 where the target token is copied.

        A target token is copy-safe (kappa=0, do not unlearn) when it lies inside a
        contiguous ``copy_ngram``-token span of the answer that also occurs
        contiguously in the prompt -- exact token-span matching, the first
        implementation the proposal recommends. ``copy_ngram=1`` degrades to
        single-token membership, which would spare common words; the default of 2
        spares copied entities (e.g. an author's name echoed from the question)
        without sparing every stopword.

        Loops over the batch: it is small (per-device train batch), and the work is
        negligible next to the three transformer forward passes per step.
        """
        input_ids = art["input_ids"]  # clean gold tokens (post-BOS if right-shifted)
        answer_mask = art["maskable_mask"]
        attention_mask = art["attention_mask"]
        b, l = input_ids.shape

        prompt_mask = ~answer_mask
        if attention_mask is not None:
            prompt_mask = prompt_mask & attention_mask.bool()

        kappa = torch.ones(b, l, device=input_ids.device, dtype=torch.float)
        n = self.copy_ngram
        for i in range(b):
            prompt_ids = input_ids[i][prompt_mask[i]].tolist()
            ans_positions = answer_mask[i].nonzero(as_tuple=True)[0].tolist()
            if len(prompt_ids) < n or len(ans_positions) < n:
                continue
            prompt_ngrams = {
                tuple(prompt_ids[s : s + n]) for s in range(len(prompt_ids) - n + 1)
            }
            row = input_ids[i].tolist()
            for j in range(len(ans_positions) - n + 1):
                window = ans_positions[j : j + n]
                # Only a genuinely contiguous answer span can be "copied".
                if window[-1] - window[0] != n - 1:
                    continue
                if tuple(row[p] for p in window) in prompt_ngrams:
                    for p in window:
                        kappa[i, p] = 0.0
        return kappa

    def _projected_target(self, logits_c, logits_u, art):
        """Closed-form KL projection ``q*`` of ``p^c`` under the prompt-lift caps.

        Returns ``(log_q, q)``, both detached (computed from the frozen teacher).
        Implements Sec. "Projected Teacher Target": for tokens the prompt amplifies
        (``g > 0``) the mass is capped at ``u = (p^c)^(1-alpha) (p^u)^alpha``; the
        rest keeps its conditional shape rescaled by ``gamma`` so ``q*`` sums to 1.
        """
        log_pc = self._log_probs(logits_c)  # (b, l, V) on the tokenizer vocab
        log_pu = self._log_probs(logits_u)
        g = log_pc - log_pu  # prompt-induced log-prob lift
        positive = g > 0

        # alpha_{t,i} = lam * kappa * (1 - exp(-[g(y_i)]_+)), forget strength at the
        # gold token. input_ids holds the clean target even where y_t is masked.
        true_ids = art["input_ids"].clamp(max=self.vocab_size - 1).unsqueeze(-1)
        g_true = g.gather(-1, true_ids).squeeze(-1)  # (b, l)
        kappa = self._copy_safe_indicator(art)
        alpha = self.lam * kappa * (1.0 - torch.exp(-g_true.clamp_min(0.0)))
        alpha = alpha.unsqueeze(-1)  # (b, l, 1)

        # Upper bound in log space: log u = (1-alpha) log p^c + alpha log p^u. On the
        # positive set g>0 so u <= p^c, hence sum_{S+} u <= 1 and gamma >= 0.
        log_u = (1.0 - alpha) * log_pc + alpha * log_pu
        pc = log_pc.exp()
        u = log_u.exp()
        sum_pos_u = (u * positive).sum(-1)  # (b, l)
        mass_neg = (pc * (~positive)).sum(-1)  # conditional mass off the positive set

        gamma_proj = (1.0 - sum_pos_u).clamp_min(0.0) / mass_neg.clamp_min(
            self.gap_epsilon
        )
        log_gamma = gamma_proj.clamp_min(self.gap_epsilon).log().unsqueeze(-1)

        log_q = torch.where(positive, log_u, log_gamma + log_pc)
        q = log_q.exp()
        # Defensive renormalisation: the closed form is exact, but float rounding of
        # the two partial sums can leave the total a hair off 1.
        norm = q.sum(-1, keepdim=True).clamp_min(self.gap_epsilon)
        q = q / norm
        log_q = log_q - norm.log()
        return log_q.detach(), q.detach()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        # Student conditional pass p^theta = p_theta(. | x, y_t); this also fixes the
        # shared noised state y_t that both teacher passes reuse.
        art = self._sample_and_denoise(
            model,
            forget_inputs["input_ids"],
            forget_inputs["labels"],
            forget_inputs.get("attention_mask", None),
        )
        with torch.no_grad():
            teacher_c = self._postprocess_outputs(
                self.ref_model(
                    input_ids=art["noised_input_ids"],
                    attention_mask=art["attention_mask"],
                )
            )
            teacher_u = self._postprocess_outputs(
                self.ref_model(
                    input_ids=self._null_input_ids(art),
                    attention_mask=art["attention_mask"],
                )
            )
            log_q, q = self._projected_target(
                teacher_c.logits, teacher_u.logits, art
            )

        log_p_theta = self._log_probs(art["logits"])
        # KL(sg[q*] || p^theta) per position; q* detached so grad = -sum q* log p^theta.
        kl = (q * (log_q - log_p_theta)).sum(-1)
        kl = kl * art["loss_weights"] * art["masked_mask"].to(kl.dtype)
        forget_loss = self._normalize_token_nll(
            kl, art["maskable_mask"], art["input_ids"].shape[0]
        )

        # Retain-free by default (alpha=0); optional retain NLL for ablation.
        retain_loss = self.optional_retain_loss(model, inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, art["outputs"]) if return_outputs else loss
