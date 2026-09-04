# Auto Unlearning for Masked Diffusion Language Models

Auto Unlearning is a retain-free objective that starts from the frozen teacher's
original conditional prediction and suppresses the part that the forget prompt
adds relative to a prompt-masked prediction. A student is then trained directly
toward that target on normal masked-diffusion states.

This document separates the mathematical definition from code and runtime
details:

- [Part I: Theory](#part-i-theory) defines the selective target and loss.
- [Part II: Implementation](#part-ii-implementation) covers masking, numerical
  behavior, configuration, and TOFU usage.

---

# Part I: Theory

## 1. Setup

For a forget example, let $x$ be its prompt and $y_t$ a partially masked answer
at diffusion time $t$. The frozen original model $p_{\theta_0}$ supplies two
distributions at each masked answer position $i$:

$$
p_{\mathrm{cond}}^i(v)
=p_{\theta_0}(v\mid x,y_t)_i,
\qquad
p_{\mathrm{uncond}}^i(v)
=p_{\theta_0}(v\mid m,y_t)_i,
$$

where $m$ denotes the same input with attended prompt tokens replaced by mask
tokens. The answer state $y_t$ is identical in both passes.

The name “unconditional” is shorthand for **prompt-masked**. Revealed answer
tokens in $y_t$ remain visible, so this is not an unconditional language-model
distribution in the strict probabilistic sense.

## 2. Prompt-added log-probability

Define the conditional log-probability shift

$$
\Delta_{i,t}(v)
=
\log\frac{p_{\mathrm{cond}}^i(v)}{p_{\mathrm{uncond}}^i(v)}.
$$

Positive values identify tokens whose probability increases when the forget
prompt is visible. Negative values identify tokens that the prompt suppresses.
Semantic unlearning needs to suppress the former, but does not need to reverse
the latter. Therefore use only the positive part:

$$
\Delta_{i,t,+}(v)
=
\max\!\left(0,\Delta_{i,t}(v)\right).
$$

## 3. Selective forgetting target

Use the existing bounded control $0\le\lambda<1$ and convert it to the
nonnegative suppression strength

$$
\beta=\frac{\lambda}{1-\lambda}.
$$

Then define

$$
q_{i,t}^{\mathrm{forget}}(v)
\propto
p_{\mathrm{cond}}^i(v)
\exp\!\left[-\beta\Delta_{i,t,+}(v)\right].
$$

Equivalently,

$$
q_{i,t}^{\mathrm{forget}}(v)
=
\operatorname{Softmax}_v\!\left(
\log p_{\mathrm{cond}}^i(v)
-\beta\Delta_{i,t,+}(v)
\right).
$$

For compact, readable log-score notation, let
$z_c(v)=\log p_{\mathrm{cond}}^i(v)$,
$z_u(v)=\log p_{\mathrm{uncond}}^i(v)$, and let $z_f$ denote the unnormalized
forget-target score. Then

$$
z_f(v)
=
z_c(v)
-
\beta\left[z_c(v)-z_u(v)\right]_+.
$$

Here $z_c$ and $z_u$ are normalized log probabilities, not raw model logits.
This distinction matters because the positive-part operation is not invariant
to a vocabulary-wide additive constant.

Using $p_{\mathrm{cond}}$ as the base is deliberate: the conditional-vs-
prompt-masked difference decides what to suppress, but does not replace the
model's normal conditional behavior with the prompt-masked distribution.

## 4. Training loss

The student minimizes **forward KL** from the selective target:

$$
\mathcal L_{\mathrm{Auto}}
=
\mathbb E_{(x,y),t}
\left[
\frac{1}{|\mathcal M_t|}
\sum_{i\in\mathcal M_t}
D_{\mathrm{KL}}\!\left(
q_{i,t}^{\mathrm{forget}}
\;\|\;
p_\theta^i(\cdot\mid x,y_t)
\right)
\right].
$$

This KL direction is part of the objective: the fixed selective target weights
the student log probabilities, and gradients flow only through $p_\theta$.

## 5. Effect of $\lambda$

| Value | Behavior |
|---:|---|
| $\lambda=0$ | $q^{\mathrm{forget}}=p_{\mathrm{cond}}$: preserve the original conditional teacher. |
| $0<\lambda<1$ | Start from the conditional teacher and suppress only prompt-added probability with strength $\beta=\lambda/(1-\lambda)$. |
| $\lambda\to1^-$ | The target can become extremely sharp because $\beta$ diverges. |
| $\lambda\ge1$ | Unsupported because the convenience mapping to $\beta$ is undefined or negative. |

---

# Part II: Implementation

## 6. Per-batch execution

For each forget batch, the trainer:

1. Samples one diffusion time per example and masks answer positions, producing
   $y_t$ and $\mathcal M_t$.
2. Runs the conditional student on $(x,y_t)$.
3. Runs the frozen teacher on the same $(x,y_t)$ to obtain
   $\log p_{\mathrm{cond}}$.
4. Replaces attended, non-answer, non-padding prompt tokens with the model's mask
   token and runs the same frozen teacher to obtain $\log p_{\mathrm{uncond}}$.
5. Clamps $\log p_{\mathrm{cond}}-\log p_{\mathrm{uncond}}$ at zero, subtracts
   the resulting suppression score from $\log p_{\mathrm{cond}}$, and
   constructs $\log q^{\mathrm{forget}}$ in float32 with `logsumexp`
   normalization.
6. Computes forward KL at masked answer positions and averages first within each
   active example, then across examples.

There is one student forward and two frozen-teacher forwards per batch. Teacher
and target construction run without gradients.

## 7. Numerical and modeling details

- `lam` is validated before trainer setup and must be finite with
  `0 <= lam < 1`.
- Student, teacher, and target distributions are normalized over
  `len(tokenizer)`, excluding padded LM-head-only rows.
- Log probabilities, target construction, and KL accumulation use float32.
- The target is detached; gradients flow only through the conditional student.
- Conditional and prompt-masked teacher passes share the exact sampled answer
  mask, revealed answer tokens, and attention mask.
- Scheduler/CART weights are not applied to the KL. The configured scheduler is
  still used to sample $y_t$.
- No retain dataset, retain loss, or ordinary denoising NLL is used.

## 8. Configuration

The trainer defaults are in
[`configs/trainer/DiffusionAutoUnlearning.yaml`](../configs/trainer/DiffusionAutoUnlearning.yaml):

| Argument | Default | Constraint | Purpose |
|---|---:|---|---|
| `lam` | `0.5` | $0\le\lambda<1$ | Bounded control for $\beta=\lambda/(1-\lambda)$. |

For direct Hydra runs, override it with:

```bash
trainer.method_args.lam=0.25
```

## 9. Running on TOFU

Run the default LLaDA/forget10 experiment with:

```bash
bash scripts/dlm/tofu/run_auto_unlearning.sh
```

Set the bounded suppression control through `LAMBDA`:

```bash
LAMBDA=0.25 bash scripts/dlm/tofu/run_auto_unlearning.sh
```

Dream and other standard TOFU launcher controls remain available:

```bash
MODEL=dream GPUS=0,1 SPLIT=forget05 LAMBDA=0.5 \
  bash scripts/dlm/tofu/run_auto_unlearning.sh
```

The launcher delegates to
[`scripts/dlm/tofu/unlearn.sh`](../scripts/dlm/tofu/unlearn.sh), so
`UNLEARN_EPOCHS`, `TRAIN_BS`, `GRAD_ACCUM`, `NO_EVAL`, `TASK_SUFFIX`, and Hydra
overrides work as they do for other DLLM methods.

## 10. Caveats

- Values close to one strongly amplify prompt-added log-probability. Begin
  with moderate values and inspect target entropy and training stability.
- If revealed answer tokens already expose the forgotten fact, the prompt-masked
  teacher can still assign it high probability.
- The method suppresses prompt-associated positive shifts; that does not by
  itself prove removal under paraphrases or unrelated extraction attacks.

## 11. Source map

| Responsibility | File |
|---|---|
| Selective target and forward-KL trainer | [`src/trainer/dlm/unlearn/auto_unlearning.py`](../src/trainer/dlm/unlearn/auto_unlearning.py) |
| Shared DLLM noising and output handling | [`src/trainer/dlm/base.py`](../src/trainer/dlm/base.py) |
| Frozen-teacher and reduction utilities | [`src/trainer/dlm/unlearn/base.py`](../src/trainer/dlm/unlearn/base.py) |
| Trainer defaults | [`configs/trainer/DiffusionAutoUnlearning.yaml`](../configs/trainer/DiffusionAutoUnlearning.yaml) |
| Retain-free TOFU experiment | [`configs/experiment/unlearn/tofu/dlm_auto_unlearning.yaml`](../configs/experiment/unlearn/tofu/dlm_auto_unlearning.yaml) |
| TOFU launcher | [`scripts/dlm/tofu/run_auto_unlearning.sh`](../scripts/dlm/tofu/run_auto_unlearning.sh) |
