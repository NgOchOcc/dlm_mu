# Entity-Counterfactual Selective Unlearning for Masked Diffusion Language Models

Selective Unlearning removes entity-dependent semantics from a masked diffusion
language model (DLLM) while preserving the original answer's non-secret
structure.

The method uses only one answer sequence: the ground-truth answer $y$. It first
applies the ordinary DLLM corruption process to obtain $y_t$, then additionally
masks the answer tokens that expose the target entity or fact. A frozen teacher
is evaluated on this same answer state with either the full query or an
entity-masked query.

The full-query teacher identifies which vocabulary tokens are supported by the
entity. The entity-masked teacher supplies a coherent counterfactual
distribution. These two signals form one target distribution for the student.

There is no generated secret-free answer, second answer trajectory, or
cross-answer positional alignment.

This document separates the mathematical definition from the code and runtime
details:

- [Part I: Theory](#part-i-theory) defines the entity-counterfactual target and
  student objective.
- [Part II: Implementation](#part-ii-implementation) explains the ADU-derived
  entity masks, numerical behavior, configuration, and TOFU usage.

---

# Part I: Theory

## 1. Setup

Let the forget dataset contain query-answer pairs

$$
\mathcal D_f=\{(x,y)\},
$$

where $x$ is the original query and $y=(y_1,\ldots,y_L)$ is its ground-truth
answer. Let $p_{\theta_0}$ be a frozen copy of the original model and
$p_\theta$ the trainable student initialized from it.

For each example, define two entity masks:

- $\mathcal I_x(x)$: entity-bearing token positions in the query;
- $\mathcal I_y(y)$: answer positions that directly expose the entity or fact
  to be forgotten.

The two masks refer to the same forget concept but act on different sequences.
They may contain different tokenizations and numbers of positions.

## 2. Entity-masked query

Define the counterfactual query

$$
m_{\mathrm{ent}}(x)_j
=
\begin{cases}
\texttt{[MASK]}, & j\in\mathcal I_x(x),\\
x_j, & j\notin\mathcal I_x(x).
\end{cases}
$$

Only the entity-bearing query tokens are masked. All remaining query tokens stay
visible, preserving instructions, question type, syntax, and non-secret
context.

The resulting teacher distribution is called **entity-counterfactual**, not
fully unconditional. It describes how the frozen model behaves when the target
entity is hidden but the rest of the query remains available.

---

## 3. A single aligned answer trajectory

Sample a diffusion time $t$ and a standard DLLM answer mask $\mathcal M_t$.
Applying the ordinary forward corruption process to the ground-truth answer
produces

$$
y_t
\sim
q_t(\,\cdot\mid y).
$$

Next, mask the answer positions that reveal the entity or fact:

$$
\widetilde{\mathcal M}_t
=
\mathcal M_t
\cup
\mathcal I_y(y),
$$

and

$$
\widetilde y_t
=
m_{\mathrm{ent}}(y_t)
=
\operatorname{Mask}
\left(y,\widetilde{\mathcal M}_t\right).
$$

Equivalently, $\widetilde y_t$ is the ordinary noised ground-truth answer with
all remaining entity-bearing answer tokens hidden. If an entity position was
already selected by $\mathcal M_t$, masking it again has no effect.

This construction prevents visible answer tokens from leaking the forgotten
entity into the teacher or student context. At the same time, all non-secret
answer tokens left visible by the ordinary diffusion mask continue to provide
the original answer's structural scaffold.

Every teacher and student distribution in the objective is evaluated on this
same $\widetilde y_t$. Consequently, position $i$ always refers to the same
ground-truth answer position.

---

## 4. Notation

| Symbol | Meaning |
|---|---|
| $\mathcal D_f$ | Forget dataset containing pairs $(x,y)$. |
| $x$ | Original query with the target entity visible. |
| $y$ | Original ground-truth answer. |
| $\mathcal I_x(x)$ | Entity-bearing query-token positions. |
| $\mathcal I_y(y)$ | Entity- or fact-bearing answer-token positions. |
| $m_{\mathrm{ent}}(x)$ | Query with positions in $\mathcal I_x(x)$ masked. |
| $t$ | Diffusion time sampled by the ordinary DLLM process. |
| $\mathcal M_t$ | Answer positions selected by ordinary DLLM corruption. |
| $\widetilde{\mathcal M}_t$ | Union of the ordinary and answer-entity masks. |
| $y_t$ | Ordinary noised ground-truth answer. |
| $\widetilde y_t=m_{\mathrm{ent}}(y_t)$ | The same answer state with entity-bearing positions also masked. |
| $\theta_0$ | Frozen original model used as the teacher. |
| $\theta$ | Trainable student initialized from $\theta_0$. |
| $p_{\mathrm{cond}}^{i,t}$ | Teacher distribution from $(x,\widetilde y_t)$. |
| $p_{\mathrm{cf}}^{i,t}$ | Entity-counterfactual teacher distribution from $(m_{\mathrm{ent}}(x),\widetilde y_t)$. |
| $s_{i,t}(v)$ | Entity-dependence score of vocabulary token $v$. |
| $q_{i,t}^*$ | Selective-unlearning target distribution. |
| $\beta$ | Strength of entity-dependent semantic suppression. |
| $\mathcal V$ | Model vocabulary. |

---

## 5. Conditional and entity-counterfactual teachers

For every training position $i\in\widetilde{\mathcal M}_t$ and vocabulary token
$v\in\mathcal V$, evaluate the frozen teacher twice on the same answer state.

### 5.1 Full-query conditional distribution

Keep the original query visible:

$$
p_{\mathrm{cond}}^{i,t}(v)
=
p_{\theta_0}^i
\left(v\mid x,\widetilde y_t\right).
$$

This distribution contains the teacher's original entity-dependent behavior.

### 5.2 Entity-counterfactual distribution

Mask only the entity-bearing query tokens:

$$
p_{\mathrm{cf}}^{i,t}(v)
=
p_{\theta_0}^i
\left(v\mid m_{\mathrm{ent}}(x),\widetilde y_t\right).
$$

The entity-counterfactual teacher cannot directly observe the target entity in
either the query or the revealed answer context. It still sees the non-secret
query and answer tokens, so its distribution supplies a coherent local
linguistic reference.

The only difference between $p_{\mathrm{cond}}^{i,t}$ and
$p_{\mathrm{cf}}^{i,t}$ is the visibility of the query entity. Their comparison
therefore estimates the contribution of that entity while holding the answer
trajectory fixed.

---

## 6. Entity-dependent semantic score

Define the positive conditional log-probability shift

$$
s_{i,t}(v)
=
\left[
\log p_{\mathrm{cond}}^{i,t}(v)
-
\log p_{\mathrm{cf}}^{i,t}(v)
\right]_+
=
\left[
\log
\frac{
p_{\theta_0}^i(v\mid x,\widetilde y_t)
}{
p_{\theta_0}^i(v\mid m_{\mathrm{ent}}(x),\widetilde y_t)
}
\right]_+.
$$

The score is evaluated over the complete vocabulary:

- $s_{i,t}(v)\approx0$ means that revealing the entity does not increase the
  probability of token $v$;
- $s_{i,t}(v)\gg0$ means that token $v$ receives substantial probability from
  the visible entity and should be suppressed more strongly.

The positive part is deliberate. If revealing the entity decreases a token's
probability, the method does not reward or amplify that token merely to reverse
the difference.

---

## 7. KL-regularized target distribution

At each $i\in\widetilde{\mathcal M}_t$, define the target as

$$
\boxed{
q_{i,t}^*
=
\underset{q\in\Delta(\mathcal V)}{\arg\min}
\left[
D_{\mathrm{KL}}
\left(
q\,\|\,p_{\mathrm{cf}}^{i,t}
\right)
+
\beta
\mathbb E_{v\sim q}
\left[s_{i,t}(v)\right]
\right].
}
$$

The two terms have distinct roles:

- $D_{\mathrm{KL}}(q\|p_{\mathrm{cf}}^{i,t})$ keeps the target in the coherent
  distributional region supplied by the frozen entity-counterfactual teacher;
- $\beta\mathbb E_{v\sim q}[s_{i,t}(v)]$ removes mass from vocabulary tokens
  whose probability depends on the visible entity.

The KL term prevents the suppression objective from moving probability toward
arbitrary zero-score tokens. No generated secret-free answer is required: the
frozen teacher preserves structure using the non-secret context already present
in $(m_{\mathrm{ent}}(x),\widetilde y_t)$.

### 7.1 Closed-form solution

Introduce a Lagrange multiplier $\lambda$ for $\sum_v q(v)=1$. The Lagrangian
is

$$
\mathcal J(q,\lambda)
=
\sum_{v\in\mathcal V}
q(v)
\left[
\log\frac{q(v)}{p_{\mathrm{cf}}^{i,t}(v)}
+
\beta s_{i,t}(v)
\right]
+
\lambda
\left(
\sum_{v\in\mathcal V}q(v)-1
\right).
$$

Stationarity gives

$$
\log\frac{q(v)}{p_{\mathrm{cf}}^{i,t}(v)}
+1+\beta s_{i,t}(v)+\lambda=0,
$$

so

$$
q(v)
\propto
p_{\mathrm{cf}}^{i,t}(v)
\exp[-\beta s_{i,t}(v)].
$$

After normalization,

$$
\boxed{
q_{i,t}^*(v)
=
\frac{
p_{\mathrm{cf}}^{i,t}(v)
\exp[-\beta s_{i,t}(v)]
}{
\displaystyle
\sum_{u\in\mathcal V}
p_{\mathrm{cf}}^{i,t}(u)
\exp[-\beta s_{i,t}(u)]
}.
}
$$

Equivalently,

$$
q_{i,t}^*(v)
\;\propto\;
\underbrace{
  p_{\mathrm{cf}}^{i,t}(v)
}_{
  \substack{
    \text{entity-counterfactual}\\
    \text{structural anchor}
  }
}
\qquad\times\qquad
\underbrace{
  \exp[-\beta s_{i,t}(v)]
}_{
  \substack{
    \text{entity-dependent}\\
    \text{semantic suppression}
  }
}.
$$

---

## 8. Student objective

The student receives the full original query and the same entity-masked answer
state used by both teachers:

$$
p_\theta^i
\left(v\mid x,\widetilde y_t\right).
$$

It minimizes forward KL to the fixed target at the masked positions:

$$
\boxed{
\mathcal L_{\mathrm{Selective}}
=
\mathbb E_{(x,y)\sim\mathcal D_f,\,t}
\left[
\frac{1}{|\widetilde{\mathcal M}_t|}
\sum_{i\in\widetilde{\mathcal M}_t}
D_{\mathrm{KL}}
\left(
q_{i,t}^*
\,\|\,
p_\theta^i(\cdot\mid x,\widetilde y_t)
\right)
\right].
}
$$

The student sees the entity in the query but not in the answer context. Matching
$q_{i,t}^*$ teaches it to replace the original entity-dependent prediction with
the frozen teacher's coherent counterfactual behavior.

There is one student answer trajectory and one student loss. The two teacher
passes construct the target but do not create separate student objectives.

---

## 9. Effect of $\beta$

The suppression strength $\beta\geq0$ controls the tradeoff inside the target:

$$
q_{i,t}^*(v)
\propto
p_{\mathrm{cf}}^{i,t}(v)
e^{-\beta s_{i,t}(v)}.
$$

| Regime | Theoretical behavior |
|---:|---|
| $\beta=0$ | $q_{i,t}^*=p_{\mathrm{cf}}^{i,t}$: pure entity-counterfactual distillation. |
| Small $\beta>0$ | Tokens mildly supported by the entity receive additional suppression. |
| Large $\beta$ | The target concentrates on the lowest entity-dependence region of $p_{\mathrm{cf}}^{i,t}$. |

The counterfactual teacher already removes direct entity conditioning. Positive
$\beta$ additionally suppresses tokens that regain probability when the entity
is restored in the query. Excessively large $\beta$ may make the target generic
or overly concentrated.

---

## 10. Why cross-answer alignment is unnecessary

The method never compares position $i$ in the original answer with position
$i$ in a generated alternative answer. All quantities use
$\widetilde y_t=m_{\mathrm{ent}}(y_t)$:

$$
\begin{aligned}
p_{\mathrm{cond}}^{i,t}
&=p_{\theta_0}^i(\cdot\mid x,\widetilde y_t),\\
p_{\mathrm{cf}}^{i,t}
&=p_{\theta_0}^i(\cdot\mid m_{\mathrm{ent}}(x),\widetilde y_t),\\
p_\theta^{i,t}
&=p_\theta^i(\cdot\mid x,\widetilde y_t).
\end{aligned}
$$

Therefore:

- sequence length is unchanged;
- token positions are inherited from the ground-truth answer;
- the ordinary diffusion mask and entity mask have explicit, separate roles;
- the structural and suppression signals are combined at the same position and
  diffusion state.

---

## 11. Assumptions and limitations

1. **Answer-side supervision is required.** The method needs
   $\mathcal I_y(y)$, an accurate mask for answer tokens that reveal the target
   entity or fact.
2. **Surface masking may be incomplete.** Entity names alone may not cover
   aliases, descriptions, relations, or paraphrases that reveal the same fact.
3. **The teacher may reconstruct the entity.** Remaining query or answer clues
   can allow $p_{\mathrm{cf}}$ to infer the hidden information despite masking.
4. **Entity positions are always noised.** The effective training mask
   $\widetilde{\mathcal M}_t$ differs from ordinary DLLM corruption because it
   deterministically includes $\mathcal I_y(y)$.
5. **The structural anchor is local.** It preserves the teacher's behavior for
   the masked ground-truth answer scaffold; it does not explicitly teach a new
   complete response style or refusal policy.

These tradeoffs are the cost of eliminating the generated secret-free answer
trajectory. In return, the objective is positionally aligned, uses a single
answer sequence, and obtains both structure and suppression from the frozen
teacher.

---

# Part II: Implementation

The trainer is implemented in
[`src/trainer/dlm/unlearn/selective_unlearning.py`](../src/trainer/dlm/unlearn/selective_unlearning.py).
It uses the tokenizer-surface entity heuristics from the `adu` branch to
construct practical approximations of the theoretical masks
$\mathcal I_x(x)$ and $\mathcal I_y(y)$. The TOFU dataset does not provide
explicit token-level entity annotations to this trainer.

## 12. Entity-mask construction

Entity masks are computed from the clean, tokenized query-answer sequence after
the normal Dream/LLaDA preprocessing. Query positions are attended,
non-padding positions whose labels equal `-100`; answer positions are those
whose labels do not equal `-100`.

Decoded subword surfaces are grouped into whitespace-delimited words before
matching. This preserves all tokens in a split word, so an entity split across
multiple tokenizer pieces is masked as one unit. Matching lowercases each word,
removes chat-template markers, keeps alphanumeric fragments, and ignores
one-character fragments.

### 12.1 Query entity mask

[`query_entity_mask`](../src/trainer/dlm/unlearn/selective_unlearning.py)
constructs $\mathcal I_x(x)$ as the union of:

1. query words whose normalized form also occurs in the answer;
2. capitalized, non-sentence-initial words not present in the ADU query
   allowlist; and
3. words containing a digit, which covers dates and other numeric identifiers.

The first rule catches shared names and titles even when capitalization alone
is insufficient. The second catches proper-noun identity words not repeated in
the answer. The allowlist prevents common question, function, and TOFU-domain
words such as `What`, `author`, `book`, and `genre` from being treated as
entities solely because of capitalization.

### 12.2 Answer entity mask

[`answer_entity_mask`](../src/trainer/dlm/unlearn/selective_unlearning.py)
constructs $\mathcal I_y(y)$ as the union of:

1. answer words whose normalized form occurs in the query; and
2. capitalized, non-sentence-initial answer words not present in the ADU answer
   allowlist.

This catches names and shared titles on the answer side, including words split
across multiple tokens. Because this is a surface heuristic, it may over-mask
ordinary query-overlap words and under-mask aliases, paraphrases, lowercased
facts, or implicit relations. Supplying annotated entity masks would require a
separate dataset/collator interface and is not part of the current
implementation.

## 13. Shared answer-state construction

For each forget batch, the trainer performs the following steps:

1. Applies optional EOS response-block extension and model-specific
   preprocessing. Dream inputs are prepended/right-shifted through the shared
   DLM implementation; LLaDA remains answer-aligned.
2. Constructs the query and answer entity masks from the clean processed
   sequence.
3. Samples one diffusion time per example from
   $t\sim\mathcal U(\epsilon,1)$ and converts it to a mask probability using the
   configured scheduler.
4. Samples the ordinary answer mask $\mathcal M_t$ and forms
   $\widetilde{\mathcal M}_t=\mathcal M_t\cup\mathcal I_y(y)$.
5. Replaces every position in $\widetilde{\mathcal M}_t$ with the tokenizer's
   mask token, producing $\widetilde y_t$.
6. Runs the student on the full query and $\widetilde y_t$.
7. Runs the frozen teacher on the same input for $p_{\mathrm{cond}}$.
8. Masks only $\mathcal I_x(x)$ in that shared input and runs the frozen teacher
   again for $p_{\mathrm{cf}}$.
9. Constructs $q^*$ and computes forward KL at every position in
   $\widetilde{\mathcal M}_t$.

There is one student forward and two frozen-teacher forwards per batch. No
secret-free answer is generated, and no retain batch or denoising NLL is added
to the objective.

The corresponding execution flow is:

```text
Input: forget batch, student p_theta, frozen teacher p_theta0

1. I_x <- ADU_ENTITY_MASK(query, answer)
2. I_y <- ADU_ENTITY_MASK(answer, query)
3. M_t <- SAMPLE_ORDINARY_DIFFUSION_MASK(answer)
4. M_tilde <- M_t UNION I_y
5. y_tilde <- MASK(answer, M_tilde)

6. log p_student <- student(full query, y_tilde)
7. log p_cond    <- teacher(full query, y_tilde)
8. log p_cf      <- teacher(entity-masked query, y_tilde)

9. score <- RELU(log p_cond - log p_cf)
10. log q <- LOG_SOFTMAX(log p_cf - beta * score)
11. loss <- MEAN_i KL(q_i || p_student_i), i in M_tilde
12. update theta only
```

## 14. Numerical and distributed behavior

- Teacher and student logits are truncated to `len(tokenizer)`, excluding any
  padded rows that exist only in the model's LM head.
- Teacher log probabilities, the entity-dependence score, target normalization,
  and KL accumulation use float32 even when model forwards use bfloat16.
- The score uses normalized teacher log probabilities rather than raw logits;
  this is necessary because the positive-part operation is not invariant to
  different log-normalization constants.
- The target is normalized with `log_softmax` after subtracting the suppression
  term. At `beta=0`, it is exactly the entity-counterfactual teacher
  distribution up to floating-point precision.
- The implementation follows the theory's complete tokenizer-vocabulary
  support. Unlike several other unlearning trainers in this repository, it does
  not condition the target on a lexical-only vocabulary.
- Position losses are averaged within each active example and then across
  examples. Scheduler/CART loss weights are not applied to this KL, although
  the scheduler still determines the ordinary diffusion mask.
- The reference model is a frozen copy of the initial student. Teacher passes
  and target construction run under `torch.no_grad()`, so gradients flow only
  through the student distribution.
- The shared reference-model preparation supports the repository's normal
  Accelerate and DeepSpeed execution paths.

Right padding is required. The implementation reuses the shared EOS extension,
Dream input preprocessing, and output-logit postprocessing so entity masks and
prediction positions stay aligned for both supported DLLMs.

## 15. Configuration

Trainer defaults live in
[`configs/trainer/DiffusionSelectiveUnlearning.yaml`](../configs/trainer/DiffusionSelectiveUnlearning.yaml):

| Argument | Default | Purpose |
|---|---:|---|
| `beta` | `1.0` | Nonnegative strength of entity-dependent vocabulary suppression. `0` performs pure entity-counterfactual distillation. |

`beta` is converted to a float and validated during trainer initialization. A
negative, infinite, or NaN value raises `ValueError` before training begins.

The TOFU experiment configuration is
[`configs/experiment/unlearn/tofu/dlm_selective_unlearning.yaml`](../configs/experiment/unlearn/tofu/dlm_selective_unlearning.yaml).
It selects `DiffusionSelectiveUnlearning`, anchors training on the forget split,
and removes the retain dataset entirely.

## 16. Running on TOFU

Launch the default LLaDA `forget10` run with:

```bash
bash scripts/dlm/tofu/run_selective_unlearning.sh
```

Override the model, devices, split, or suppression strength with environment
variables:

```bash
MODEL=dream GPUS=0,1 SPLIT=forget05 BETA=0.5 \
  bash scripts/dlm/tofu/run_selective_unlearning.sh
```

The launcher runs training and evaluation through
[`scripts/dlm/tofu/unlearn.sh`](../scripts/dlm/tofu/unlearn.sh), detaches the
process with `setsid`, and writes output to
`logs/selective_unlearning_selective_beta<BETA>.log` by default. Extra
command-line arguments are forwarded as Hydra overrides. Shared controls such
as `UNLEARN_EPOCHS`, `TRAIN_BS`, `GRAD_ACCUM`, `NO_EVAL`, `RESUME`, and
`TASK_SUFFIX` remain available.

## 17. Source map

| Responsibility | File |
|---|---|
| Entity masks, shared state, target, and forward-KL loss | [`src/trainer/dlm/unlearn/selective_unlearning.py`](../src/trainer/dlm/unlearn/selective_unlearning.py) |
| Shared DLLM preprocessing and diffusion scheduler | [`src/trainer/dlm/base.py`](../src/trainer/dlm/base.py) |
| Frozen reference-model preparation and loss reduction | [`src/trainer/dlm/unlearn/base.py`](../src/trainer/dlm/unlearn/base.py) |
| Trainer defaults | [`configs/trainer/DiffusionSelectiveUnlearning.yaml`](../configs/trainer/DiffusionSelectiveUnlearning.yaml) |
| TOFU experiment | [`configs/experiment/unlearn/tofu/dlm_selective_unlearning.yaml`](../configs/experiment/unlearn/tofu/dlm_selective_unlearning.yaml) |
| Trainer registry | [`src/trainer/__init__.py`](../src/trainer/__init__.py) |
| TOFU launcher | [`scripts/dlm/tofu/run_selective_unlearning.sh`](../scripts/dlm/tofu/run_selective_unlearning.sh) |
