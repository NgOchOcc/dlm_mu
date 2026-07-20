"""Fact-wise Counterfactual Projection for masked diffusion LMs (FCP-MDLM).

Implements the online primal-dual trainer of docs/FCP_MDLM_training_algorithm.tex
(Algorithm 2). The offline pass (``fcp_precompute.py``) has already built, per
forget fact, the fact-bearing span ``K`` and the per-position probability caps
``rho`` for the recall and infill states. This trainer needs only those cached
caps plus the retain teacher ``theta_0`` -- never ``theta_b``.

Per optimizer step:
  * For each forget fact ``j`` in the batch, rebuild its cached recall/infill
    states, run the student, and read ``p_a`` at every monitored atom. The
    one-sided projection loss ``ell_rho(p_a)`` is 0 once the target prob is at
    or below its cap. The fact constraint is the *hard max* over the fact's
    atoms, ``F_j = max_a ell_rho(p_a)`` (Eq. fact-constraint): one easy atom
    cannot mask a still-memorized one.
  * The retain objective is the trust region ``KL(theta_0 || theta)`` on a
    sampled retain mask (identical to the AMTU retain term).
  * Loss = R_retain + (1/|B_f|) sum_j lambda_j (F_j - delta); one nonnegative
    multiplier per fact is raised by dual ascent on the (detached) constraint
    violation.

Frame convention: the cached ``input_ids`` are already in the model's
post-preprocess frame (BOS prepended for Dream), so states are forwarded
directly and ``_postprocess_outputs`` is applied before reading logits -- exactly
as in precompute, keeping ``p_a`` identical across the two paths.
"""

import torch
import torch.nn.functional as F
from transformers import TrainerCallback

from trainer.dlm.unlearn.base import DiffusionUnlearnTrainer
from trainer.dlm.unlearn.fcp_common import target_prob, projection_loss


class DiffusionFCP(DiffusionUnlearnTrainer):
    """Fact-wise counterfactual projection unlearning.

    Args:
        cache_path: Path to the ``.pt`` cache produced by ``precompute_fcp.py``
            (a dict ``{"facts": {fact_id: {...}}, "meta": {...}}``).
        delta: Per-fact forgetting tolerance ``delta_j`` (constant across facts).
        eta_lambda: Dual-ascent step size for the per-fact multipliers.
        lambda_init: Initial value of every multiplier ``lambda_j >= 0``.
        feasibility_check_period: If > 0, attach a callback that evaluates the
            full cached atom bank every this many steps and early-stops after
            ``patience`` consecutive all-feasible checks. 0/None disables it.
        patience: Number of consecutive feasible checks required to stop.
        strict_epsilon: Unused at train time (caps already baked in during
            precompute); kept so the config knob round-trips.
    """

    def __init__(
        self,
        cache_path,
        delta=0.03,
        eta_lambda=0.05,
        lambda_init=1.0,
        feasibility_check_period=0,
        patience=3,
        strict_epsilon=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        cache = torch.load(cache_path, map_location="cpu")
        self.fcp_facts = cache["facts"]
        self.fcp_meta = cache.get("meta", {})
        if self.fcp_meta.get("right_shift_logits") not in (None, self.right_shift_logits):
            raise ValueError(
                "FCP cache right_shift_logits="
                f"{self.fcp_meta.get('right_shift_logits')} disagrees with trainer "
                f"right_shift_logits={self.right_shift_logits}; regenerate the cache."
            )
        self.delta = float(delta)
        self.eta_lambda = float(eta_lambda)
        # One nonnegative multiplier per fact, kept on CPU as a plain dict so
        # arbitrary (possibly non-contiguous) fact ids work.
        self.lmbda = {int(fid): float(lambda_init) for fid in self.fcp_facts}
        self.feasibility_check_period = int(feasibility_check_period or 0)
        self.patience = int(patience)
        # theta_0 retain teacher (also the source the caps were built against).
        self.ref_model = self._prepare_ref_model(self.model)
        if self.feasibility_check_period > 0:
            self.add_callback(FCPFeasibilityCallback(self))

    # ---- state construction (post-preprocess frame, from cache) -------------
    def _mask_tokens(self, input_ids, positions):
        return torch.where(positions, self.processing_class.mask_token_id, input_ids)

    def _fact_states(self, fact_id):
        """Rebuild the (recall, infill) states + monitored atoms for one fact.

        Returns None when the fact has no monitored span (nothing to constrain).
        """
        entry = self.fcp_facts.get(int(fact_id))
        if entry is None:
            return None
        device = self.accelerator.device
        input_ids = entry["input_ids"].to(device)
        attn = entry["attention_mask"].to(device)
        answer_mask = entry["answer_mask"].to(device).bool()
        K_pos = entry["K_pos"].to(device)
        if K_pos.numel() == 0:
            return None
        y_K = entry["y_K"].to(device)
        rho_recall = entry["rho_recall"].to(device)
        rho_infill = entry["rho_infill"].to(device)

        recall_ids = self._mask_tokens(input_ids, answer_mask)
        infill_ids = input_ids.clone()
        infill_ids[K_pos] = self.processing_class.mask_token_id
        return {
            "recall_ids": recall_ids,
            "infill_ids": infill_ids,
            "attn": attn,
            "K_pos": K_pos,
            "y_K": y_K,
            "rho_recall": rho_recall,
            "rho_infill": rho_infill,
        }

    def _pad_stack(self, id_list, attn_list, pad_value):
        """Right-pad a list of 1-D tensors into a (B, Lmax) batch + mask."""
        b = len(id_list)
        lmax = max(int(t.numel()) for t in id_list)
        device = id_list[0].device
        ids = torch.full((b, lmax), pad_value, dtype=torch.long, device=device)
        attn = torch.zeros((b, lmax), dtype=attn_list[0].dtype, device=device)
        for i, (t, a) in enumerate(zip(id_list, attn_list)):
            ids[i, : t.numel()] = t
            attn[i, : a.numel()] = a
        return ids, attn

    def _student_logits(self, model, ids, attn):
        outputs = model(input_ids=ids, attention_mask=attn)
        return self._postprocess_outputs(outputs).logits

    def _fact_constraints(self, model, fact_ids):
        """F_j = max over recall/infill atoms of the projection loss, per fact.

        Runs two batched student forwards (one per state) over all facts that
        have a monitored span. Returns ``(F, kept_ids)`` where ``F`` is a
        1-D tensor aligned with ``kept_ids`` (facts with an empty span are
        dropped -- they contribute no constraint).
        """
        states, kept = [], []
        for fid in fact_ids:
            st = self._fact_states(fid)
            if st is not None:
                states.append(st)
                kept.append(int(fid))
        if not states:
            return None, []

        pad_id = self.processing_class.pad_token_id or 0
        recall_ids, recall_attn = self._pad_stack(
            [s["recall_ids"] for s in states], [s["attn"] for s in states], pad_id
        )
        infill_ids, infill_attn = self._pad_stack(
            [s["infill_ids"] for s in states], [s["attn"] for s in states], pad_id
        )
        recall_logits = self._student_logits(model, recall_ids, recall_attn)
        infill_logits = self._student_logits(model, infill_ids, infill_attn)

        f_vals = []
        for i, s in enumerate(states):
            p_recall = target_prob(recall_logits[i], s["K_pos"], s["y_K"])
            p_infill = target_prob(infill_logits[i], s["K_pos"], s["y_K"])
            l_recall = projection_loss(p_recall, s["rho_recall"])
            l_infill = projection_loss(p_infill, s["rho_infill"])
            atoms = torch.cat([l_recall, l_infill])
            f_vals.append(atoms.max())
        return torch.stack(f_vals), kept

    # ---- retain trust region (identical to AMTU) ----------------------------
    def _retain_loss(self, model, retain_inputs):
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
        retain_loss, retain_outputs = self._retain_loss(model, inputs["retain"])

        fact_ids = inputs["forget"]["index"].detach().cpu().tolist()
        F_j, kept = self._fact_constraints(model, fact_ids)

        if F_j is None:
            forget_penalty = retain_loss.new_zeros(())
            pass_rate = 1.0
            max_F = 0.0
        else:
            lam = F_j.new_tensor([self.lmbda[j] for j in kept])
            # Primal: lambda is a constant multiplier; gradient flows via F_j.
            forget_penalty = (lam * (F_j - self.delta)).mean()
            # Dual ascent on the detached constraint violation.
            if model.training:
                viol = (F_j.detach() - self.delta).cpu().tolist()
                for j, v in zip(kept, viol):
                    self.lmbda[j] = max(0.0, self.lmbda[j] + self.eta_lambda * v)
            with torch.no_grad():
                pass_rate = float((F_j <= self.delta).float().mean().item())
                max_F = float(F_j.max().item())

        loss = retain_loss + forget_penalty
        if model.training:
            self.log(
                {
                    "fcp/retain": float(retain_loss.detach().item()),
                    "fcp/penalty": float(forget_penalty.detach().item())
                    if torch.is_tensor(forget_penalty)
                    else float(forget_penalty),
                    "fcp/fact_pass_rate": pass_rate,
                    "fcp/max_F": max_F,
                    "fcp/mean_lambda": sum(self.lmbda.values()) / max(len(self.lmbda), 1),
                }
            )
        return (loss, retain_outputs) if return_outputs else loss

    # ---- full-bank feasibility (for early stopping) -------------------------
    @torch.no_grad()
    def full_bank_feasibility(self, model, chunk=16):
        """Fraction of ALL cached facts with ``F_j <= delta`` and the max F_j."""
        model.eval()
        all_ids = list(self.fcp_facts.keys())
        passed, worst, counted = 0, 0.0, 0
        for start in range(0, len(all_ids), chunk):
            chunk_ids = all_ids[start : start + chunk]
            F_j, kept = self._fact_constraints(model, chunk_ids)
            if F_j is None:
                continue
            passed += int((F_j <= self.delta).sum().item())
            worst = max(worst, float(F_j.max().item()))
            counted += len(kept)
        model.train()
        pass_rate = passed / counted if counted else 1.0
        return pass_rate, worst


class FCPFeasibilityCallback(TrainerCallback):
    """Stop once the full atom bank is feasible for ``patience`` checks in a row.

    Mirrors the complete-bank stopping criterion in the proposal: feasibility is
    judged on the fixed cached bank, not on the sampled training batch.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.streak = 0

    def on_step_end(self, args, state, control, **kwargs):
        t = self.trainer
        if t.feasibility_check_period <= 0:
            return control
        if state.global_step == 0 or state.global_step % t.feasibility_check_period:
            return control
        model = kwargs.get("model", t.model)
        pass_rate, max_F = t.full_bank_feasibility(model)
        t.log({"fcp/bank_pass_rate": pass_rate, "fcp/bank_max_F": max_F})
        if pass_rate >= 1.0 - 1e-9:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= t.patience:
            control.should_training_stop = True
        return control
