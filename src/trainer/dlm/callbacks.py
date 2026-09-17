"""Console formatting for DLM training logs.

The HF default writes ``str(logs)``: a raw dict whose key order and float widths
shift from step to step, so a run's console output cannot be scanned or diffed. It
also arrives as *two* lines per step, because the DLM unlearning trainers log their
own ``method/metric`` dict from ``training_step`` while the Trainer separately logs
loss and lr. :meth:`~trainer.dlm.base.DiffusionFinetuneTrainer.log` merges the two;
this module renders the result as one ``key=value`` line.

Kept in ``trainer.dlm`` and installed only by ``DiffusionFinetuneTrainer``, so the
autoregressive trainers keep the upstream behaviour.
"""

from transformers.trainer_callback import PrinterCallback, ProgressCallback

# `epoch` and `step` go into the line prefix; the rest is noise at every step.
_DROP = frozenset({"total_flos", "epoch", "step", "num_input_tokens_seen"})

# Rendered first, in this order, under a short label. Everything else follows in
# insertion order, which is the order the trainer built its metrics dict in.
_CORE = (("loss", "loss"), ("grad_norm", "gnorm"), ("learning_rate", "lr"))


def _value(label, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if label == "lr":
        return f"{value:.1e}"
    if label == "gnorm":
        return f"{value:.2f}"
    # Counts (cached_trajectories, fact_positions, knowledge_keys, ...) reach the
    # logger as integral floats; four decimals would bury the digits that matter.
    if float(value).is_integer() and abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.4f}"


def _prefix(state):
    epoch = f"ep {state.epoch:.2f}" if state.epoch is not None else "ep -"
    if state.max_steps:
        return f"{epoch} {state.global_step}/{state.max_steps}"
    return f"{epoch} step {state.global_step}"


def _summary(logs, state):
    """The end-of-training dict, which carries runtime rather than a step's metrics."""
    parts = [f"done {state.global_step} steps", f"{logs['train_runtime'] / 60:.1f} min"]
    if "train_samples_per_second" in logs:
        parts.append(f"{logs['train_samples_per_second']:.2f} samples/s")
    if "train_loss" in logs:
        parts.append(f"mean loss {logs['train_loss']:.4f}")
    return "  |  ".join(parts)


def format_logs(logs, state):
    """One console line for one ``Trainer.log`` call, or ``None`` if nothing to show."""
    logs = {key: value for key, value in logs.items() if key not in _DROP}
    if "train_runtime" in logs:
        return _summary(logs, state)

    prefix = _prefix(state)
    if any(key.startswith("eval_") for key in logs):
        logs = {key.removeprefix("eval_"): value for key, value in logs.items()}
        prefix = f"eval {prefix}"

    fields = [(label, logs.pop(key)) for key, label in _CORE if key in logs]
    # `method/metric` -> `metric`: the namespace is constant within a run, so it is
    # redundant on every line.
    fields += [(key.rsplit("/", 1)[-1], value) for key, value in logs.items()]
    if not fields:
        return None
    body = " ".join(f"{label}={_value(label, value)}" for label, value in fields)
    return f"{prefix} {body}"


class DiffusionProgressCallback(ProgressCallback):
    """``ProgressCallback`` with the dict dump replaced by :func:`format_logs`."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or self.training_bar is None:
            return
        line = format_logs(logs or {}, state)
        if line:
            self.training_bar.write(line)


class DiffusionPrinterCallback(PrinterCallback):
    """The ``disable_tqdm`` path: same line, plain ``print``.

    A separate class because ``ProgressCallback.on_step_end`` dereferences
    ``self.training_bar`` unguarded, so it cannot be reused without a bar.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        line = format_logs(logs or {}, state)
        if line:
            print(line, flush=True)
