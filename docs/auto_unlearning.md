# Auto Unlearning for Masked Diffusion Language Models

Auto Unlearning is a retain-free objective for suppressing prompt-specific
knowledge while minimizing changes to the frozen teacher's other behavior. It
compares the teacher's prediction with the forget prompt visible against its
prediction with that prompt masked, then trains the student toward the resulting
selective target on ordinary masked-diffusion states.

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

## 3. Optimization view and selective target

Let the nonnegative suppression score be

$$
s_{i,t}(v)=\Delta_{i,t,+}(v),
$$

and let $\beta\ge 0$ directly control suppression strength. At every masked
position, define the target as the solution of

$$
q_{i,t}^{*}
=
\underset{q\in\Delta(\mathcal V)}{\arg\min}
\left[
D_{\mathrm{KL}}\!\left(q\,\|\,p_{\mathrm{cond}}^i\right)
+\beta\,\mathbb E_{v\sim q}\!\left[s_{i,t}(v)\right]
\right].
$$

The first term keeps the target close to the original conditional teacher. The
second penalizes probability mass on vocabulary tokens identified as
prompt-specific. Thus the objective expresses **minimal behavioral change
subject to suppressing prompt-specific information**.

Solving the constrained optimization over probability distributions gives

$$
q_{i,t}^{*}(v)
\propto
p_{\mathrm{cond}}^i(v)
\exp\!\left[-\beta s_{i,t}(v)\right].
$$

Equivalently, the normalized forget target is

$$
q_{i,t}^{\mathrm{forget}}(v)
=
\operatorname{Softmax}_v\!\left(
\log p_{\mathrm{cond}}^i(v)
-\beta s_{i,t}(v)
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

The student minimizes **reverse KL** to the selective target:

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
q_{i,t}^{\mathrm{forget}}
\right)
\right].
$$

This KL direction is part of the objective. Replacing it with
$D_{\mathrm{KL}}(q^{\mathrm{forget}}\|p_\theta)$ would define a different
optimization problem.

## 5. Effect of $\beta$

| Value | Behavior |
|---:|---|
| $\beta=0$ | $q^{\mathrm{forget}}=p_{\mathrm{cond}}$: preserve the original conditional teacher. |
| $0<\beta<1$ | Apply mild suppression to prompt-added probability. |
| $\beta=1$ | Cancel the positive conditional log-probability shift before normalization. |
| $\beta>1$ | Penalize prompt-specific tokens more strongly; large values can produce a sharp target. |

---

# Part II: Implementation

## Special-token exclusion

Auto Unlearning does not treat sequence-control tokens as semantic alternatives
at ordinary response positions. The implementation defines the forbidden set

$$
\mathcal S
=\{\text{all tokenizer-declared special-token IDs}\}
\cup\{\text{derived assistant-turn terminator}\}.
$$

Thus $\mathcal S$ includes EOT/EOS, PAD, BOS, and MASK whenever the tokenizer
defines them. Deriving the assistant-turn terminator separately is necessary
because Dream's `im_end` and LLaDA's `eot_id` are not always reported by
`all_special_ids`.

At a masked response position whose clean target is lexical, the conditional
teacher, prompt-masked teacher, forget target, and student distribution are all
conditioned on $\mathcal V_{\mathrm{lex}}=\mathcal V\setminus\mathcal S$:

$$
\widetilde r(v)=
\begin{cases}
r(v), & v\notin\mathcal S,\\
0, & v\in\mathcal S,
\end{cases}
\qquad
r_{\mathrm{lex}}(v)=
\frac{\widetilde r(v)}{\sum_{u\notin\mathcal S}\widetilde r(u)}.
$$

The reverse KL therefore compares the student and target on the same lexical
support; it never evaluates positive student mass against a zero-probability
special-token target. If the clean token at a masked response position is itself
in $\mathcal S$, that position is omitted from the forget loss. This preserves
the sequence-format role of EOT/EOS rather than trying to unlearn it.

## 6. Per-batch execution

For each forget batch, the trainer:

1. Samples one diffusion time per example from $t\sim\operatorname{Beta}(3,1)$,
   scaled to $[\epsilon,1)$, and masks answer positions, producing $y_t$ and
   $\mathcal M_t$. This emphasizes high-noise states while retaining support
   across the full training-time interval.
2. Runs the conditional student on $(x,y_t)$.
3. Runs the frozen teacher on the same $(x,y_t)$ to obtain
   $\log p_{\mathrm{cond}}$.
4. Replaces attended, non-answer, non-padding prompt tokens with the model's mask
   token and runs the same frozen teacher to obtain $\log p_{\mathrm{uncond}}$.
5. Removes the forbidden special-token support and renormalizes both teacher
   distributions over lexical tokens.
6. Clamps $\log p_{\mathrm{cond}}-\log p_{\mathrm{uncond}}$ at zero, subtracts
   the resulting suppression score from $\log p_{\mathrm{cond}}$, and
   constructs $\log q^{\mathrm{forget}}$ in float32 with `logsumexp`
   normalization over the same lexical support.
7. Computes reverse KL at masked lexical answer positions and averages first
   within each active example, then across examples.

There is one student forward and two frozen-teacher forwards per batch. Teacher
and target construction run without gradients.

## 7. Numerical and modeling details

- `beta` is validated before trainer setup and must be finite with `beta >= 0`.
- Student, teacher, and target distributions first drop padded LM-head-only rows,
  then exclude special/control IDs and normalize over the remaining tokenizer
  vocabulary.
- Log probabilities, target construction, and KL accumulation use float32.
- The target is detached; gradients flow only through the conditional student.
- Conditional and prompt-masked teacher passes share the exact sampled answer
  mask, revealed answer tokens, and attention mask.
- Auto Unlearning samples $t\sim\operatorname{Beta}(3,1)$ rather than uniformly;
  with the linear scheduler, its expected masking rate is approximately $0.75$
  (up to the small `time_epsilon` rescaling).
- Scheduler/CART weights are not applied to the KL. The configured scheduler is
  still used to sample $y_t$.
- No retain dataset, retain loss, or ordinary denoising NLL is used.

## 8. Configuration

The trainer defaults are in
[`configs/trainer/DiffusionAutoUnlearning.yaml`](../configs/trainer/DiffusionAutoUnlearning.yaml):

| Argument | Default | Constraint | Purpose |
|---|---:|---|---|
| `beta` | `1.0` | $\beta\ge0$ and finite | Direct strength of prompt-specific suppression. |

For direct Hydra runs, override it with:

```bash
trainer.method_args.beta=0.5
```

This is a breaking parameter rename. To reproduce a run that used the old
bounded parameter $\lambda$, convert it with
$\beta=\lambda/(1-\lambda)$. For example, the old `lam=0.5` is the new
`beta=1.0`.

## 9. Running on TOFU

Run the default LLaDA/forget10 experiment with:

```bash
bash scripts/dlm/tofu/run_auto_unlearning.sh
```

Set the suppression strength through `BETA`:

```bash
BETA=0.5 bash scripts/dlm/tofu/run_auto_unlearning.sh
```

Dream and other standard TOFU launcher controls remain available:

```bash
MODEL=dream GPUS=0,1 SPLIT=forget05 BETA=1.0 \
  bash scripts/dlm/tofu/run_auto_unlearning.sh
```

The launcher delegates to
[`scripts/dlm/tofu/unlearn.sh`](../scripts/dlm/tofu/unlearn.sh), so
`UNLEARN_EPOCHS`, `TRAIN_BS`, `GRAD_ACCUM`, `NO_EVAL`, `TASK_SUFFIX`, and Hydra
overrides work as they do for other DLLM methods.

## 10. Caveats

- Large $\beta$ values strongly suppress prompt-added log-probability. Begin
  with moderate values and inspect target entropy and training stability.
- If revealed answer tokens already expose the forgotten fact, the prompt-masked
  teacher can still assign it high probability.
- The method suppresses prompt-associated positive shifts; that does not by
  itself prove removal under paraphrases or unrelated extraction attacks.

## 11. Source map

| Responsibility | File |
|---|---|
| Selective target and reverse-KL trainer | [`src/trainer/dlm/unlearn/auto_unlearning.py`](../src/trainer/dlm/unlearn/auto_unlearning.py) |
| Shared DLLM noising and output handling | [`src/trainer/dlm/base.py`](../src/trainer/dlm/base.py) |
| Frozen-teacher and reduction utilities | [`src/trainer/dlm/unlearn/base.py`](../src/trainer/dlm/unlearn/base.py) |
| Trainer defaults | [`configs/trainer/DiffusionAutoUnlearning.yaml`](../configs/trainer/DiffusionAutoUnlearning.yaml) |
| Retain-free TOFU experiment | [`configs/experiment/unlearn/tofu/dlm_auto_unlearning.yaml`](../configs/experiment/unlearn/tofu/dlm_auto_unlearning.yaml) |
| TOFU launcher | [`scripts/dlm/tofu/run_auto_unlearning.sh`](../scripts/dlm/tofu/run_auto_unlearning.sh) |
