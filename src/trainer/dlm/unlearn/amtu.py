"""Adversarial Masked-Transition Unlearning (AMTU) for masked diffusion LMs.

Implements the algorithm in docs/AMTU_training_algorithm.tex. For a forget
example the trainer searches, per mask-cardinality stratum, for the masked
answer position whose local prediction is *hardest* to push toward a frozen
anchor, then minimises the KL from the policy to the anchor at that
adversarial position. A retain trust region keeps the policy close to the
initial model on retain data.

Design choices (fixed for this codebase):
  * Anchor a_i = the frozen initial model theta_0 evaluated on the same masked
    answer state but with the *prompt also masked* -- an unconditional /
    "forget-free" distribution (cf. DiffusionNullAnchor, but frozen at theta_0).
  * Sensitive positions I_f(x, y) = all answer tokens (labels != -100).
  * Faithful Algorithm 1: embedding-gradient swap proposals with the exact
    discrete accept test. The first-order swap score
    Delta_{u,v} = g_u.(e(y_u) - e_mask) + g_v.(e_mask - e(y_v)) is additively
    separable, so argmax over (u, v) factorises into two O(n) argmaxes.
"""

import torch
import torch.nn.functional as F

from trainer.dlm.unlearn.base import DiffusionUnlearnTrainer


class DiffusionAMTU(DiffusionUnlearnTrainer):
    """Adversarial masked-transition unlearning.

    Args:
        beta: Smoothing parameter of the log-sum-exp surrogate h_beta.
        lambda_retain: Weight lambda on the retain trust-region KL.
        mask_fractions: Cardinality strata as fractions of the answer length;
            per example, k = clamp(round(frac * n_answer), 1, n_answer).
        restarts: Random-restart count R in the mask search.
        swap_steps: Swap steps q per restart.
    """

    def __init__(
        self,
        beta=5.0,
        lambda_retain=1.0,
        mask_fractions=(0.25, 0.5, 0.75),
        restarts=1,
        swap_steps=2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = beta
        self.lambda_retain = lambda_retain
        self.mask_fractions = list(mask_fractions)
        self.restarts = int(restarts)
        self.swap_steps = int(swap_steps)
        # Frozen theta_0: serves both as the null anchor and as the retain
        # trust-region reference.
        self.ref_model = self._prepare_ref_model(self.model)

    # ---- low-level state / forward helpers ----------------------------------
    def _mask_tokens(self, input_ids, positions):
        return torch.where(positions, self.processing_class.mask_token_id, input_ids)

    def _policy_logits(self, model, noised_input_ids, attention_mask, inputs_embeds=None):
        if inputs_embeds is not None:
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        return self._postprocess_outputs(outputs).logits

    @torch.no_grad()
    def _anchor_logits(self, noised_input_ids, prompt_mask, attention_mask):
        """Null anchor: frozen theta_0 on the masked answer state with the
        prompt additionally masked (unconditional distribution)."""
        anchor_input = self._mask_tokens(noised_input_ids, prompt_mask)
        outputs = self.ref_model(input_ids=anchor_input, attention_mask=attention_mask)
        return self._postprocess_outputs(outputs).logits

    @staticmethod
    def _kl_per_pos(logits_p, logits_a):
        """KL(p_theta || a) per position, a detached."""
        log_p = F.log_softmax(logits_p, dim=-1)
        log_a = F.log_softmax(logits_a, dim=-1)
        return (log_p.exp() * (log_p - log_a)).sum(-1)

    def _local_kl(self, model, noised_input_ids, prompt_mask, attention_mask,
                  inputs_embeds=None):
        """Per-position KL(p_theta || anchor) on a given masked state."""
        logits_p = self._policy_logits(model, noised_input_ids, attention_mask, inputs_embeds)
        anchor = self._anchor_logits(noised_input_ids, prompt_mask, attention_mask)
        return self._kl_per_pos(logits_p, anchor.detach()), logits_p

    # ---- Algorithm 1: adversarial mask search -------------------------------
    def _sample_masks(self, answer_mask, k_per_example):
        """Uniformly sample |M| = k answer positions per example -> (b, l) bool."""
        b, l = answer_mask.shape
        scores = torch.rand(b, l, device=answer_mask.device)
        scores = scores.masked_fill(~answer_mask, -1.0)  # answer positions rank highest
        M = torch.zeros_like(answer_mask)
        for i in range(b):
            k = int(k_per_example[i].item())
            if k <= 0:
                continue
            top = torch.topk(scores[i], k).indices
            M[i, top] = True
        return M

    def _embedding_grad(self, model, input_ids, M, prompt_mask, answer_mask, attention_mask):
        """g_j = d h_beta(M) / d E_j via a single backward to input embeddings."""
        emb_layer = model.get_input_embeddings()
        noised = self._mask_tokens(input_ids, M)
        with torch.enable_grad():
            embeds = emb_layer(noised).detach().requires_grad_(True)
            ell, _ = self._local_kl(model, noised, prompt_mask, attention_mask,
                                    inputs_embeds=embeds)
            ell_M = ell.masked_fill(~M, float("-inf"))
            h = torch.logsumexp(self.beta * ell_M, dim=1) / self.beta  # (b,)
            grad = torch.autograd.grad(h.sum(), embeds)[0]  # (b, l, d)
        return grad.detach()

    def _propose_swap(self, grad, input_ids, M, answer_mask, model):
        """Factorised first-order swap: unmask best u in M, mask best v not in M."""
        emb_layer = model.get_input_embeddings()
        e_true = emb_layer(input_ids)  # (b, l, d)
        mask_id = torch.tensor(self.processing_class.mask_token_id, device=input_ids.device)
        e_mask = emb_layer(mask_id)  # (d,)
        # A_u = g_u . (e(y_u) - e_mask); note B_v = -A_v.
        A = (grad * (e_true - e_mask)).sum(-1)  # (b, l)
        A_in = A.masked_fill(~M, float("-inf"))
        visible = answer_mask & (~M)
        A_out = A.masked_fill(~visible, float("inf"))
        u_star = A_in.argmax(dim=1)  # (b,)
        v_star = A_out.argmin(dim=1)  # (b,)
        has_visible = visible.any(dim=1)
        M_swapped = M.clone()
        idx = torch.arange(M.shape[0], device=M.device)
        M_swapped[idx[has_visible], u_star[has_visible]] = False
        M_swapped[idx[has_visible], v_star[has_visible]] = True
        return M_swapped

    def _discrete_loss(self, model, input_ids, M, prompt_mask, attention_mask):
        """L(M) = max_{i in M} ell_i (exact discrete loss), per example."""
        noised = self._mask_tokens(input_ids, M)
        with torch.no_grad():
            ell, _ = self._local_kl(model, noised, prompt_mask, attention_mask)
        ell_M = ell.masked_fill(~M, float("-inf"))
        return ell_M.max(dim=1).values, ell  # (b,), (b, l)

    def _search_stratum(self, model, input_ids, answer_mask, prompt_mask,
                        attention_mask, k_per_example):
        """Best (M, i, L) per example for one cardinality stratum."""
        b, l = input_ids.shape
        best_M = torch.zeros_like(answer_mask)
        best_i = torch.zeros(b, dtype=torch.long, device=input_ids.device)
        best_L = torch.full((b,), float("-inf"), device=input_ids.device)

        for _ in range(self.restarts):
            M = self._sample_masks(answer_mask, k_per_example)
            L_cur, _ = self._discrete_loss(model, input_ids, M, prompt_mask, attention_mask)
            for _ in range(self.swap_steps):
                grad = self._embedding_grad(model, input_ids, M, prompt_mask,
                                            answer_mask, attention_mask)
                M_new = self._propose_swap(grad, input_ids, M, answer_mask, model)
                L_new, _ = self._discrete_loss(model, input_ids, M_new, prompt_mask,
                                               attention_mask)
                accept = L_new > L_cur  # exact discrete accept test
                M = torch.where(accept.unsqueeze(1), M_new, M)
                L_cur = torch.where(accept, L_new, L_cur)
            L_final, ell = self._discrete_loss(model, input_ids, M, prompt_mask,
                                               attention_mask)
            i_final = ell.masked_fill(~M, float("-inf")).argmax(dim=1)
            improve = L_final > best_L
            best_M = torch.where(improve.unsqueeze(1), M, best_M)
            best_i = torch.where(improve, i_final, best_i)
            best_L = torch.where(improve, L_final, best_L)
        return best_M, best_i, best_L

    def _adversarial_forget_target(self, model, input_ids, answer_mask,
                                   prompt_mask, attention_mask):
        """Outer stratum loop: global best (M*, i*) per forget example."""
        b, l = input_ids.shape
        n_answer = answer_mask.sum(dim=1)  # (b,)
        star_M = torch.zeros_like(answer_mask)
        star_i = torch.zeros(b, dtype=torch.long, device=input_ids.device)
        star_L = torch.full((b,), float("-inf"), device=input_ids.device)

        for frac in self.mask_fractions:
            k = torch.round(frac * n_answer.float()).long().clamp(min=1)
            k = torch.minimum(k, n_answer.clamp(min=1))
            M, i, L = self._search_stratum(model, input_ids, answer_mask,
                                           prompt_mask, attention_mask, k)
            improve = L > star_L
            star_M = torch.where(improve.unsqueeze(1), M, star_M)
            star_i = torch.where(improve, i, star_i)
            star_L = torch.where(improve, L, star_L)
        return star_M, star_i

    # ---- loss terms ---------------------------------------------------------
    def _forget_loss(self, model, forget_inputs):
        inputs = {
            "input_ids": forget_inputs["input_ids"],
            "labels": forget_inputs["labels"],
        }
        if forget_inputs.get("attention_mask", None) is not None:
            inputs["attention_mask"] = forget_inputs["attention_mask"]
        inputs = self._preprocess_inputs(inputs)  # BOS prepend for Dream
        input_ids, labels = inputs["input_ids"], inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        answer_mask = labels != -100  # I_f = all answer tokens
        if attention_mask is not None:
            prompt_mask = (~answer_mask) & attention_mask.bool()
        else:
            prompt_mask = ~answer_mask

        # Mask/index selection is a non-differentiable search (stop-grad).
        with torch.no_grad():
            M_star, i_star = self._adversarial_forget_target(
                model, input_ids, answer_mask, prompt_mask, attention_mask
            )

        # Differentiable forget loss: KL(p_theta || anchor) at the adversarial
        # position i*, on the winning masked state M*.
        noised = self._mask_tokens(input_ids, M_star)
        ell, _ = self._local_kl(model, noised, prompt_mask, attention_mask)
        b = input_ids.shape[0]
        idx = torch.arange(b, device=input_ids.device)
        return ell[idx, i_star].mean()

    def _retain_loss(self, model, retain_inputs):
        """Retain trust region: mean_i KL(p_{theta_0} || p_theta) on a sampled
        retain mask (Eq. 3), gradient flowing only through the policy."""
        art = self._sample_and_denoise(
            model,
            retain_inputs["input_ids"],
            retain_inputs["labels"],
            retain_inputs.get("attention_mask", None),
        )
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=art["noised_input_ids"],
                attention_mask=art["attention_mask"],
            )
            ref_logits = self._postprocess_outputs(ref_outputs).logits
        log_ref = F.log_softmax(ref_logits, dim=-1)
        log_pol = F.log_softmax(art["logits"], dim=-1)
        kl = (log_ref.exp() * (log_ref - log_pol)).sum(-1)  # KL(theta_0 || theta)
        masked = art["masked_mask"].to(kl.dtype)
        retain_loss = (kl * masked).sum() / masked.sum().clamp_min(1)
        return retain_loss, art["outputs"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_loss = self._forget_loss(model, inputs["forget"])
        retain_loss, retain_outputs = self._retain_loss(model, inputs["retain"])
        loss = forget_loss + self.lambda_retain * retain_loss
        return (loss, retain_outputs) if return_outputs else loss
