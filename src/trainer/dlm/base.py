import re
import torch
import torch.nn.functional as F
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from trainer.base import FinetuneTrainer
from trainer.dlm.callbacks import DiffusionPrinterCallback, DiffusionProgressCallback
from dllm.pipelines.dream.trainer import cart_weight

class DiffusionFinetuneTrainer(FinetuneTrainer):
    def __init__(
        self,
        scheduler_name="LinearAlphaScheduler",
        time_epsilon=1e-3,
        loss_weight_type="scheduler",
        loss_norm_type="token",
        right_shift_logits=False,
        eos_pad_to=None,
        time_beta=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from dllm.core.schedulers import make_alpha_scheduler

        if not (0.0 < time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")
        if eos_pad_to is not None and eos_pad_to < 1:
            raise ValueError("eos_pad_to must be a positive token count or None")
        self.scheduler = make_alpha_scheduler(scheduler_name)
        self.time_epsilon = time_epsilon
        self.time_beta = time_beta  # (a, b): sample t from Beta(a, b) instead of uniformly
        self.loss_weight_type = loss_weight_type
        self.loss_norm_type = loss_norm_type
        self.right_shift_logits = right_shift_logits
        self.eos_pad_to = eos_pad_to
        self._buffered_metrics = {}
        # Exactly one of these is installed by Trainer.__init__; remove_callback is a
        # no-op for the absent one.
        self.remove_callback(ProgressCallback)
        self.remove_callback(PrinterCallback)
        self.add_callback(
            DiffusionPrinterCallback
            if self.args.disable_tqdm
            else DiffusionProgressCallback
        )

    def log(self, logs, start_time=None):
        """Merge method metrics into the step's loss log instead of emitting a second line.

        The unlearning trainers log their own ``method/metric`` dict from
        ``training_step``, which runs before the Trainer's loss log for the same step.
        Buffering the former yields one console line and one ``log_history`` entry per
        step. The ``method/`` namespace is what distinguishes those dicts: neither the
        Trainer's own keys nor any TOFU/MUSE/RWKU eval key contains a slash.
        """
        if logs and all("/" in key for key in logs):
            self._buffered_metrics.update(logs)
            return
        if self._buffered_metrics:
            logs = {**logs, **self._buffered_metrics}
            self._buffered_metrics.clear()
        super().log(logs, start_time)

    def _preprocess_inputs(self, inputs):
        if self.right_shift_logits:
            from dllm.utils.data import prepend_bos

            labels = inputs.get("labels", None)
            if labels is not None and torch.all(labels[:, 0] == -100):
                return inputs
            inputs = prepend_bos(
                inputs,
                bos_token_id=self.processing_class.bos_token_id,
                label_pad_token_id=-100,
            )
        return inputs

    def _postprocess_outputs(self, outputs):
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _compute_loss_weights(self, t, input_ids, masked_mask):
        b, l = input_ids.shape
        if self.loss_weight_type.startswith("cart"):
            match = re.search(r"geo_p:(0\.\d+)", self.loss_weight_type)
            geo_p = float(match.group(1)) if match else 0.3
            return cart_weight(masked_mask, t, p=geo_p)
        elif self.loss_weight_type == "scheduler":
            return self.scheduler.weight(t).unsqueeze(1).repeat(1, l)
        elif self.loss_weight_type == "uniform":
            return torch.ones_like(input_ids, dtype=torch.float)
        raise NotImplementedError(f"Unknown loss_weight_type: {self.loss_weight_type}")

    def _extend_response_with_eos(self, input_ids, labels, attention_mask):
        """Relabel/extend each example's response region to ``eos_pad_to`` EOS tokens.

        Training otherwise supervises only the answer plus whatever scaffolding
        the chat template appended after it (``labels != -100``), so the model
        never learns what to place in the *leftover* positions of the fixed-length
        block the sampler generates — and fills them by repeating the answer.
        Here the supervised region becomes ``answer + terminator + EOS-run`` out
        to ``eos_pad_to`` positions counted from the first answer token. The
        model then learns to terminate: trailing positions denoise to EOS, which
        eval truncates at the terminator (see ``generate_answers_diffusion``).

        Also strips the template's post-answer scaffold from the label. On LLaDA
        (Llama-3 style) that scaffold is ``<|start_header_id|>assistant<|end_header_id|>\\n\\n``
        — kept as targets it would teach the model to emit "assistant" after
        every answer. On Dream it is just ``\\n`` — cosmetic, but stripping
        makes the two models train the same recipe.

        The sampler attends ``prompt + max_new_tokens`` and must fill every masked
        position in it, so set ``eos_pad_to`` to that ``max_new_tokens`` to make the
        training and inference response blocks identical. Assumes right padding
        (checked by the caller) so the answer span is followed by pad positions.

        Because ``pad_token_id == eos_token_id`` on Dream and LLaDA, the collator's
        ``attention_mask = input_ids.ne(pad)`` also zeroes the answer's *own*
        terminal EOS — a loss target the model would then never attend. So the
        whole response block ``[start, end)`` is (re)attended here, matching the
        sampler, not just the newly padded tail.
        """
        eos_id = self.processing_class.eos_token_id
        pad_id = self.processing_class.pad_token_id
        term_tensor = self._get_terminator_tensor(input_ids.device)
        b, w = input_ids.shape

        # First answer position per example (None when the row is all IGNORE_INDEX),
        # and the total width needed to fit every response block.
        starts = []
        needed = w
        for i in range(b):
            ans = (labels[i] != -100).nonzero(as_tuple=True)[0]
            start = int(ans[0]) if ans.numel() else None
            starts.append(start)
            if start is not None:
                needed = max(needed, start + self.eos_pad_to)

        # Grow the batch on the right if any block runs past the collated width.
        if needed > w:
            grow = needed - w
            input_ids = torch.cat(
                [input_ids, input_ids.new_full((b, grow), pad_id)], dim=1
            )
            labels = torch.cat([labels, labels.new_full((b, grow), -100)], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros((b, grow))], dim=1
                )

        for i in range(b):
            if starts[i] is None:
                continue
            end = starts[i] + self.eos_pad_to  # exclusive; <= new width by construction
            ans_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
            a_last = int(ans_positions[-1])
            # Terminator = first turn-end token (im_end / eot_id) inside the
            # supervised span, if any. Keep the terminator itself as a target,
            # overwrite everything past it with EOS.
            fill_start = a_last + 1
            if term_tensor is not None:
                span = input_ids[i, starts[i] : a_last + 1]
                hit = torch.isin(span, term_tensor).nonzero(as_tuple=True)[0]
                if hit.numel() > 0:
                    fill_start = starts[i] + int(hit[0]) + 1
            if fill_start < end:
                input_ids[i, fill_start:end] = eos_id
                labels[i, fill_start:end] = eos_id
            # Attend the whole response block (see docstring): covers content,
            # terminator, the answer's terminal EOS (attn=0 from the collator), and
            # the EOS pad. No-op on the already-attended content positions.
            if attention_mask is not None:
                attention_mask[i, starts[i] : min(end, attention_mask.shape[1])] = 1

        return input_ids, labels, attention_mask

    def _get_terminator_tensor(self, device):
        """Turn-end tokens (im_end / eot_id / ...) from the chat template, excl. EOS.

        Cached per-trainer since the tokenizer doesn't change during training. EOS
        is excluded because label EOS appears at the end of every row — matching
        on it would leave fill_start at (end-of-row+1) and defeat the trim.
        """
        cached = getattr(self, "_terminator_tensor_cache", None)
        if cached is not None:
            return cached.to(device) if cached.numel() else None
        from evals.metrics.dlm_utils import derive_stop_token_ids

        stops = derive_stop_token_ids(self.processing_class)
        eos_id = self.processing_class.eos_token_id
        terms = sorted(t for t in stops if t != eos_id)
        self._terminator_tensor_cache = (
            torch.tensor(terms) if terms else torch.tensor([], dtype=torch.long)
        )
        return self._terminator_tensor_cache.to(device) if terms else None

        return input_ids, labels, attention_mask

    def _sample_timesteps(self, batch_size, device):
        """Sample diffusion times on [time_epsilon, 1]: uniformly, or Beta(*time_beta) rescaled."""
        if self.time_beta is None:
            u = torch.rand(batch_size, device=device)
        else:
            a, b = (torch.tensor(float(x), device=device) for x in self.time_beta)
            u = torch.distributions.Beta(a, b).sample((batch_size,))
        return self.time_epsilon + (1 - self.time_epsilon) * u

    def _sample_and_denoise(self, model, input_ids, labels, attention_mask=None):
        assert self.processing_class.padding_side == "right"
        if self.eos_pad_to is not None:
            input_ids, labels, attention_mask = self._extend_response_with_eos(
                input_ids, labels, attention_mask
            )
        inputs = {"input_ids": input_ids, "labels": labels}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels = inputs["input_ids"], inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        b, l = input_ids.shape
        maskable_mask = labels != -100  # answer positions only

        t = self._sample_timesteps(b, input_ids.device)
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)

        loss_weights = self._compute_loss_weights(t, input_ids, masked_mask)
        return {
            "outputs": outputs,
            "logits": outputs.logits,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "t": t,
            "masked_mask": masked_mask,
            "maskable_mask": maskable_mask,
            "noised_input_ids": noised_input_ids,
            "loss_weights": loss_weights,
        }

    def _normalize_token_nll(self, token_nll, maskable_mask, b):
        if self.loss_norm_type == "token":
            token_nll = token_nll / maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll = token_nll / (
                maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
            )
        elif self.loss_norm_type == "batch":
            token_nll = token_nll / b
        else:
            raise ValueError(f"Invalid loss_norm_type: {self.loss_norm_type}")
        return token_nll.sum()

    def diffusion_nll(
        self, model, input_ids, labels, attention_mask=None, return_outputs=False
    ):
        art = self._sample_and_denoise(model, input_ids, labels, attention_mask)
        input_ids, logits = art["input_ids"], art["logits"]
        masked_mask, maskable_mask = art["masked_mask"], art["maskable_mask"]

        token_nll = F.cross_entropy(
            logits.transpose(1, 2), input_ids, reduction="none"
        )
        token_nll = token_nll * art["loss_weights"] * masked_mask.to(token_nll.dtype)
        loss = self._normalize_token_nll(token_nll, maskable_mask, input_ids.shape[0])
        return (loss, art["outputs"]) if return_outputs else loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss, outputs = self.diffusion_nll(
            model,
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
            return_outputs=True,
        )
        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()
        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()
        return (loss.detach(), logits, labels)
