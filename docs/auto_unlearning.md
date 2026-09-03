# Auto Unlearning for Masked Diffusion Language Models

Auto Unlearning is a retain-free objective that turns a difference between the
frozen teacher's conditional and prompt-masked predictions into one closed-form
target distribution. A student is then trained directly toward that target on
normal masked-diffusion states.

This document separates the mathematical definition from code and runtime
details:

- [Part I: Theory](#part-i-theory) derives the anti-conditional target and loss.
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

## 2. Direct objective

For a candidate student distribution $p$, consider

$$
J(p)
=
D_{\mathrm{KL}}(p\|p_{\mathrm{uncond}})
-
\lambda D_{\mathrm{KL}}(p\|p_{\mathrm{cond}}),
\qquad 0\le\lambda<1.
$$

The first term pulls the student toward prompt-masked behavior. The negative
second term pushes it away from the prompt-conditioned behavior associated with
the forgotten prompt.

Expanding the KL terms gives

$$
J(p)
=
\sum_v p(v)
\left[
(1-\lambda)\log p(v)
-\log p_{\mathrm{uncond}}(v)
+\lambda\log p_{\mathrm{cond}}(v)
\right].
$$

Because $1-\lambda>0$, the normalized minimizer has a closed form.

## 3. Closed-form anti-conditional target

At position $i$, define

$$
q_{i,t}^*(v)
=
\operatorname{Softmax}_v\!\left(
\frac{
\log p_{\mathrm{uncond}}^i(v)
-\lambda\log p_{\mathrm{cond}}^i(v)
}{1-\lambda}
\right).
$$

Equivalently, up to a vocabulary-independent normalization constant,

$$
q_{i,t}^*(v)
\propto
\left[p_{\mathrm{uncond}}^i(v)\right]^{1/(1-\lambda)}
\left[p_{\mathrm{cond}}^i(v)\right]^{-\lambda/(1-\lambda)}.
$$

In logit form, vocabulary-independent softmax constants cancel, yielding the
useful interpretation

$$
z^*_{mathrm{forget}}
=
z_{mathrm{uncond}}
-
\frac{\lambda}{1-\lambda}
\left(z_{\mathrm{cond}}-z_{\mathrm{uncond}}\right).
$$

The conditional shift $z_{\mathrm{cond}}-z_{\mathrm{uncond}}$ represents what
the prompt adds. The target extrapolates in the opposite direction.

## 4. Training loss

The direct objective and closed-form target satisfy

$$
J(p)
=
(1-\lambda)D_{\mathrm{KL}}(p\|q^*)+\text{constant}.
$$

Therefore the student minimizes **reverse KL**:

$$
\mathcal L_{\mathrm{Auto}}
=
\mathbb E_{(x,y),t}
\left[
\frac{1}{|\mathcal M_t|}
\sum_{i\in\mathcal M_t}
D_{\mathrm{KL}}\!\left(
p_\theta^i(\cdot\mid x,y_t)
\;\|\;
q_{i,t}^*
\right)
\right].
$$

This KL direction is part of the derivation. Replacing it with
$D_{\mathrm{KL}}(q^*\|p_\theta)$ would define a different optimization problem.

## 5. Effect of $\lambda$

| Value | Behavior |
|---:|---|
| $\lambda=0$ | $q^*=p_{\mathrm{uncond}}$: match the prompt-masked teacher. |
| $0<\lambda<1$ | Extrapolate away from the conditional shift with strength $\lambda/(1-\lambda)$. |
| $\lambda\to1^-$ | The target can become extremely sharp because the extrapolation factor diverges. |
| $\lambda\ge1$ | Unsupported: the smooth closed-form minimizer used here is no longer valid. |

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
5. Constructs $\log q^*$ in float32 with `logsumexp` normalization.
6. Computes reverse KL at masked answer positions and averages first within each
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
| `lam` | `0.5` | $0\le\lambda<1$ | Strength of extrapolation away from the conditional teacher. |

For direct Hydra runs, override it with:

```bash
trainer.method_args.lam=0.25
```

## 9. Running on TOFU

Run the default LLaDA/forget10 experiment with:

```bash
bash scripts/dlm/tofu/run_auto_unlearning.sh
```

Set the anti-conditional strength through `LAMBDA`:

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

- Values close to one strongly amplify small teacher-logit differences. Begin
  with moderate values and inspect target entropy and training stability.
- If revealed answer tokens already expose the forgotten fact, the prompt-masked
  teacher can still assign it high probability.
- The method removes the prompt-associated distributional shift; that does not
  by itself prove removal under paraphrases or unrelated extraction attacks.

## 11. Source map

| Responsibility | File |
|---|---|
| Closed-form target and reverse-KL trainer | [`src/trainer/dlm/unlearn/auto_unlearning.py`](../src/trainer/dlm/unlearn/auto_unlearning.py) |
| Shared DLLM noising and output handling | [`src/trainer/dlm/base.py`](../src/trainer/dlm/base.py) |
| Frozen-teacher and reduction utilities | [`src/trainer/dlm/unlearn/base.py`](../src/trainer/dlm/unlearn/base.py) |
| Trainer defaults | [`configs/trainer/DiffusionAutoUnlearning.yaml`](../configs/trainer/DiffusionAutoUnlearning.yaml) |
| Retain-free TOFU experiment | [`configs/experiment/unlearn/tofu/dlm_auto_unlearning.yaml`](../configs/experiment/unlearn/tofu/dlm_auto_unlearning.yaml) |
| TOFU launcher | [`scripts/dlm/tofu/run_auto_unlearning.sh`](../scripts/dlm/tofu/run_auto_unlearning.sh) |
