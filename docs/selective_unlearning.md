# Subject-Preserving Selective Unlearning for Masked Diffusion Language Models

Selective Unlearning removes entity-dependent semantics from a masked diffusion
language model (DLLM) while preserving query-provided subjects and the original
answer's non-secret structure.

The method uses only one answer sequence: the ground-truth answer $y$. It first
applies the ordinary DLLM corruption process to obtain $y_t$, then additionally
masks both answer tokens that copy a query entity and answer tokens that expose
the fact to forget. A frozen teacher is evaluated on this same answer state with
either the full query or an entity-masked query.

The full-query teacher supplies a preservation target for copied subjects and
ordinary answer structure. At fact positions, the full-query teacher identifies
which vocabulary tokens are supported by the entity and the entity-masked
teacher supplies a coherent counterfactual distribution. Suppression is never
applied to an answer mention merely because it copies an entity already given
in the query.

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

For each example, define one query mask and two disjoint answer masks:

- $\mathcal I_x(x)$: entity-bearing token positions in the query;
- $\mathcal I_y^{\mathrm{copy}}(x,y)$: answer positions that refer to an entity
  explicitly supplied in the query;
- $\mathcal I_y^{\mathrm{forget}}(x,y)$: answer positions that expose the fact
  to be forgotten.

The answer masks are required to be disjoint:

$$
\mathcal I_y^{\mathrm{copy}}
\cap
\mathcal I_y^{\mathrm{forget}}
=
\varnothing.
$$

The copy mask is entity- or coreference-based rather than arbitrary lexical
overlap. For example, for the pair `Where is Hanoi?` and `Hanoi is in Vietnam.`,
`Hanoi` belongs to $\mathcal I_y^{\mathrm{copy}}$, `Vietnam` belongs to
$\mathcal I_y^{\mathrm{forget}}$, and the repeated function word `is` belongs
to neither mask.

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

Define the answer-side context-hiding mask

$$
\mathcal I_y^{\mathrm{hide}}
=
\mathcal I_y^{\mathrm{copy}}
\cup
\mathcal I_y^{\mathrm{forget}}.
$$

Both parts are hidden in the shared answer state:

$$
\widetilde{\mathcal M}_t
=
\mathcal M_t
\cup
\mathcal I_y^{\mathrm{hide}}(x,y),
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
all copied-subject and forget-fact positions hidden. If one of these positions
was already selected by $\mathcal M_t$, masking it again has no effect.

Copied subjects remain hidden from the counterfactual teacher so it cannot
recover the query identity from the answer. They are nevertheless preserved by
the full-query teacher target defined below. This is a hide-in-context,
preserve-in-target construction. Non-secret answer tokens left visible by the
ordinary diffusion mask continue to provide the original answer's structural
scaffold.

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
| $\mathcal I_y^{\mathrm{copy}}(x,y)$ | Answer positions that copy or corefer with an entity explicitly present in the query. |
| $\mathcal I_y^{\mathrm{forget}}(x,y)$ | Answer positions containing the fact to forget, excluding copied query entities. |
| $\mathcal I_y^{\mathrm{hide}}(x,y)$ | Union of copied-subject and forget-fact positions hidden from the shared answer context. |
| $m_{\mathrm{ent}}(x)$ | Query with positions in $\mathcal I_x(x)$ masked. |
| $t$ | Diffusion time sampled by the ordinary DLLM process. |
| $\mathcal M_t$ | Answer positions selected by ordinary DLLM corruption. |
| $\widetilde{\mathcal M}_t$ | Union of the ordinary and answer context-hiding masks. |
| $y_t$ | Ordinary noised ground-truth answer. |
| $\widetilde y_t$ | The same answer state with copied-subject and forget-fact positions also masked. |
| $\theta_0$ | Frozen original model used as the teacher. |
| $\theta$ | Trainable student initialized from $\theta_0$. |
| $p_{\mathrm{cond}}^{i,t}$ | Teacher distribution from $(x,\widetilde y_t)$. |
| $p_{\mathrm{cf}}^{i,t}$ | Entity-counterfactual teacher distribution from $(m_{\mathrm{ent}}(x),\widetilde y_t)$. |
| $s_{i,t}(v)$ | Entity-dependence score of vocabulary token $v$. |
| $q_{i,t}^{\mathrm{sel}}$ | Selective-unlearning target used only at forget-fact positions. |
| $q_{i,t}^{\mathrm{final}}$ | Piecewise student target: selective at forget positions and conditional elsewhere. |
| $\beta$ | Strength of entity-dependent semantic suppression. |
| $\lambda_{\mathrm{pres}}$ | Weight of conditional-teacher preservation relative to selective forgetting. |
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

## 7. KL-regularized selective target

At each forget-fact position
$i\in\mathcal I_y^{\mathrm{forget}}$, define the selective target as

$$
\boxed{
q_{i,t}^{\mathrm{sel}}
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
q_{i,t}^{\mathrm{sel}}(v)
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
q_{i,t}^{\mathrm{sel}}(v)
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

## 8. Subject-preserving student objective

The student receives the full original query and the same entity-masked answer
state used by both teachers:

$$
p_\theta^i
\left(v\mid x,\widetilde y_t\right).
$$

Partition the effective training positions into the forget set

$$
\mathcal F_t
=
\widetilde{\mathcal M}_t
\cap
\mathcal I_y^{\mathrm{forget}}
=
\mathcal I_y^{\mathrm{forget}},
$$

because every forget position is deterministically masked, and the preservation
set

$$
\mathcal P_t
=
\widetilde{\mathcal M}_t
\setminus
\mathcal I_y^{\mathrm{forget}}.
$$

$\mathcal P_t$ contains every copied-subject position and every ordinary
diffusion position outside the forget set. Define the final per-position target

$$
\boxed{
q_{i,t}^{\mathrm{final}}
=
\begin{cases}
q_{i,t}^{\mathrm{sel}},
& i\in\mathcal F_t,\\
p_{\mathrm{cond}}^{i,t},
& i\in\mathcal P_t.
\end{cases}
}
$$

Thus, an answer mention of the query entity is still absent from
$\widetilde y_t$, but the student is trained to reconstruct it from the full
query exactly as the original teacher would. Entity-counterfactual suppression
is used only for the answer content designated for forgetting.

Forgetting and preservation are reduced separately so that a large number of
ordinary scaffold positions cannot dilute the forget gradient:

$$
\mathcal L_{\mathrm{forget}}
=
\mathbb E_{(x,y)\sim\mathcal D_f,\,t}
\left[
\frac{1}{|\mathcal F_t|}
\sum_{i\in\mathcal F_t}
D_{\mathrm{KL}}
\left(
q_{i,t}^{\mathrm{sel}}
\,\|\,
p_\theta^i(\cdot\mid x,\widetilde y_t)
\right)
\right],
$$

and


$$
\mathcal L_{\mathrm{preserve}}
=
\mathbb E_{(x,y)\sim\mathcal D_f,\,t}
\left[
\frac{1}{|\mathcal P_t|}
\sum_{i\in\mathcal P_t}
D_{\mathrm{KL}}
\left(
p_{\mathrm{cond}}^{i,t}
\,\|\,
p_\theta^i(\cdot\mid x,\widetilde y_t)
\right)
\right].
$$

An empty set contributes zero loss. The combined objective is

$$
\boxed{
\mathcal L_{\mathrm{Selective}}
=
\mathcal L_{\mathrm{forget}}
+
\lambda_{\mathrm{pres}}
\mathcal L_{\mathrm{preserve}}.
}
$$

The student sees the entity in the query but not in the answer context. At a
copied-subject position, conditional distillation teaches the student to retain
the query entity. At a forget-fact position, matching
$q_{i,t}^{\mathrm{sel}}$ teaches it to replace the original entity-dependent
prediction with the frozen teacher's counterfactual behavior.

There is one student answer trajectory and one student forward pass. Both loss
terms reuse the same student and teacher distributions; only their position
masks and reductions differ.

---

## 9. Effect of $\beta$

The suppression strength $\beta\geq0$ controls the tradeoff inside the target:

$$
q_{i,t}^{\mathrm{sel}}(v)
\propto
p_{\mathrm{cf}}^{i,t}(v)
e^{-\beta s_{i,t}(v)}.
$$

| Regime | Theoretical behavior |
|---:|---|
| $\beta=0$ | $q_{i,t}^{\mathrm{sel}}=p_{\mathrm{cf}}^{i,t}$ at forget positions; preservation positions remain anchored to $p_{\mathrm{cond}}^{i,t}$. |
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
$\widetilde y_t$:

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

1. **Answer-side supervision is required.** The method needs an accurate
   partition into $\mathcal I_y^{\mathrm{copy}}$ and
   $\mathcal I_y^{\mathrm{forget}}$. A copied query entity must not be mistaken
   for a forget fact, and a fact token must not be incorrectly protected.
2. **Surface masking may be incomplete.** Entity names alone may not cover
   aliases, coreferences, descriptions, relations, or paraphrases that reveal
   the same fact.
3. **The teacher may reconstruct the entity.** Remaining query or answer clues
   can allow $p_{\mathrm{cf}}$ to infer the hidden information despite masking.
4. **Copied-subject and forget positions are always noised.** The effective
   training mask $\widetilde{\mathcal M}_t$ differs from ordinary DLLM
   corruption because it deterministically includes
   $\mathcal I_y^{\mathrm{hide}}$.
5. **The structural anchor is local.** It preserves the teacher's behavior for
   the masked ground-truth answer scaffold; it does not explicitly teach a new
   complete response style or refusal policy.
6. **Preserving the subject does not guarantee abstention.** The modified
   objective prevents a copied subject such as `Hanoi` from being trained toward
   an entity-ablated distribution, but the selective target at a fact position
   may still redistribute mass from `Vietnam` to another location. Preventing
   false replacement facts requires an explicit abstention or safe-answer
   target, which is outside this single-trajectory method.

These tradeoffs are the cost of eliminating the generated secret-free answer
trajectory. In return, the objective is positionally aligned, uses a single
answer sequence, and obtains both structure and suppression from the frozen
teacher.

---

# Part II: Implementation

The trainer is implemented in
[`src/trainer/dlm/unlearn/selective_unlearning.py`](../src/trainer/dlm/unlearn/selective_unlearning.py).
It uses tokenizer-surface entity heuristics adapted from the `adu` branch to
construct practical approximations of $\mathcal I_x(x)$,
$\mathcal I_y^{\mathrm{copy}}(x,y)$, and
$\mathcal I_y^{\mathrm{forget}}(x,y)$. The TOFU dataset does not provide
explicit token-level entity or fact annotations to this trainer.

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
constructs $\mathcal I_x(x)$ from capitalized words and numeric identifiers not
present in the query allowlist. Sentence-initial entities are eligible; common
question, function, and TOFU-domain words such as `What`, `author`, `book`, and
`genre` are excluded by the allowlist.

### 12.2 Copied-subject and forget-fact masks

[`answer_copy_mask`](../src/trainer/dlm/unlearn/selective_unlearning.py)
constructs $\mathcal I_y^{\mathrm{copy}}$ by matching answer words only against
the detected query-entity words. It therefore protects `Hanoi` but not the
ordinary repeated word `is` in the running example.

[`answer_entity_mask`](../src/trainer/dlm/unlearn/selective_unlearning.py)
constructs an answer entity/fact candidate mask as the union of:

1. copied query-entity words;
2. capitalized answer words not present in the answer allowlist; and
3. answer words containing a digit.

The implementation subtracts the copy mask from this candidate mask to obtain
$\mathcal I_y^{\mathrm{forget}}$, then unions the two masks to obtain
$\mathcal I_y^{\mathrm{hide}}$. Because this is a surface heuristic, it may
under-mask aliases, lowercased facts, relations, or implicit facts. Supplying
annotated fact and coreference masks would require a separate dataset/collator
interface and remains the preferred high-precision alternative.

## 13. Shared answer-state construction

For each forget batch, the trainer performs the following steps:

1. Applies optional EOS response-block extension and model-specific
   preprocessing. Dream inputs are prepended/right-shifted through the shared
   DLM implementation; LLaDA remains answer-aligned.
2. Constructs the query-entity, copied-subject, forget-fact, and answer-hiding
   masks from the clean processed sequence.
3. Samples one diffusion time per example from
   $t\sim\mathcal U(\epsilon,1)$ and converts it to a mask probability using the
   configured scheduler.
4. Samples the ordinary answer mask $\mathcal M_t$ and forms
   $\widetilde{\mathcal M}_t=\mathcal M_t\cup
   \mathcal I_y^{\mathrm{hide}}(x,y)$.
5. Replaces every position in $\widetilde{\mathcal M}_t$ with the tokenizer's
   mask token, producing $\widetilde y_t$.
6. Runs the student on the full query and $\widetilde y_t$.
7. Runs the frozen teacher on the same input for $p_{\mathrm{cond}}$.
8. Masks only $\mathcal I_x(x)$ in that shared input and runs the frozen teacher
   again for $p_{\mathrm{cf}}$.
9. Routes forget positions to $q^{\mathrm{sel}}$ and all remaining effective
   positions to $p_{\mathrm{cond}}$.
10. Separately reduces the forget and preservation forward-KL losses, then
    combines them using $\lambda_{\mathrm{pres}}$.

There is one student forward and two frozen-teacher forwards per batch. No
secret-free answer is generated, and no retain batch or denoising NLL is added
to the objective.

The corresponding execution flow is:

```text
Input: forget batch, student p_theta, frozen teacher p_theta0

1. I_x      <- QUERY_ENTITY_MASK(query)
2. I_copy   <- ANSWER_QUERY_ENTITY_MATCH(answer, I_x)
3. I_entity <- ANSWER_ENTITY_MASK(answer) UNION I_copy
4. I_forget <- I_entity MINUS I_copy
5. I_hide   <- I_copy UNION I_forget
6. M_t      <- SAMPLE_ORDINARY_DIFFUSION_MASK(answer)
7. M_tilde  <- M_t UNION I_hide
8. y_tilde  <- MASK(answer, M_tilde)

9.  log p_student <- student(full query, y_tilde)
10. log p_cond    <- teacher(full query, y_tilde)
11. log p_cf      <- teacher(entity-masked query, y_tilde)

12. score     <- RELU(log p_cond - log p_cf)
13. log q_sel <- LOG_SOFTMAX(log p_cf - beta * score)
14. q_final   <- q_sel on I_forget; p_cond otherwise
15. L_forget  <- MEAN_i KL(q_sel_i || p_student_i), i in I_forget
16. L_pres    <- MEAN_i KL(p_cond_i || p_student_i), i in M_tilde MINUS I_forget
17. loss      <- L_forget + preserve_weight * L_pres
18. update theta only
```

## 14. Numerical and distributed behavior

- Teacher and student logits are truncated to `len(tokenizer)`, excluding any
  padded rows that exist only in the model's LM head.
- Teacher log probabilities, the entity-dependence score, target normalization,
  and KL accumulation use float32 even when model forwards use bfloat16.
- The score uses normalized teacher log probabilities rather than raw logits;
  this is necessary because the positive-part operation is not invariant to
  different log-normalization constants.
- The selective target is normalized with `log_softmax` after subtracting the
  suppression term. At `beta=0`, it is exactly the entity-counterfactual
  teacher distribution at forget positions. Preservation positions always use
  the full-query teacher distribution.
- The implementation follows the theory's complete tokenizer-vocabulary
  support. Unlike several other unlearning trainers in this repository, it does
  not condition the target on a lexical-only vocabulary.
- Forget and preservation position losses are each averaged within active
  examples and then across examples before being combined. This prevents a
  large preservation set from diluting the forget term. Scheduler/CART loss
  weights are not applied to either KL, although the scheduler still determines
  the ordinary diffusion mask.
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
| `beta` | `1.0` | Nonnegative strength of entity-dependent vocabulary suppression. At forget positions, `0` performs pure entity-counterfactual distillation. |
| `preserve_weight` | `1.0` | Nonnegative weight $\lambda_{\mathrm{pres}}$ on conditional-teacher preservation at copied-subject and ordinary non-forget positions. |

Both arguments are converted to floats and validated during trainer
initialization. A negative, infinite, or NaN value raises `ValueError` before
training begins.

The TOFU experiment configuration is
[`configs/experiment/unlearn/tofu/dlm_selective_unlearning.yaml`](../configs/experiment/unlearn/tofu/dlm_selective_unlearning.yaml).
It selects `DiffusionSelectiveUnlearning`, anchors training on the forget split,
and removes the retain dataset entirely.

## 16. Running on TOFU

Launch the default LLaDA `forget10` run with:

```bash
bash scripts/dlm/tofu/run_selective_unlearning.sh
```

Override the model, devices, split, suppression strength, or preservation
weight with environment variables:

```bash
MODEL=dream GPUS=0,1 SPLIT=forget05 BETA=0.5 PRESERVE_WEIGHT=1.0 \
  bash scripts/dlm/tofu/run_selective_unlearning.sh
```

The launcher runs training and evaluation through
[`scripts/dlm/tofu/unlearn.sh`](../scripts/dlm/tofu/unlearn.sh), detaches the
process with `setsid`, and writes output to
`logs/selective_unlearning_selective_beta<BETA>_preserve<PRESERVE_WEIGHT>.log`
by default. Extra command-line arguments are forwarded as Hydra overrides.
Shared controls such as `UNLEARN_EPOCHS`, `TRAIN_BS`, `GRAD_ACCUM`, `NO_EVAL`,
`RESUME`, and `TASK_SUFFIX` remain available.

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
| Subject-mask and target-routing tests | [`tests/test_selective_unlearning.py`](../tests/test_selective_unlearning.py) |
