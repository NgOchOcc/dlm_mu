# Selective Unlearning for Masked Diffusion Language Models

Selective Unlearning teaches a masked diffusion language model (DLLM) to
produce a secret-free answer when it receives a prompt containing information
that should be forgotten.

The method has three distinct operations:

1. Mask only the entity tokens in the query and let the frozen teacher generate
   a relevant secret-free answer using ordinary DLLM decoding.
2. Separately mask the full query to create the unconditional context used to
   measure query-dependent semantics.
3. Use the normally masked ground-truth answer to measure how much semantics
   should be suppressed, while using the secret-free answer only as a
   structural anchor for the target distribution.

There is no unlearning-specific procedure that selects or masks answer tokens.
Answer masking is used only by the model's normal generation and training
processes, exactly as it is when an original DLLM is decoded or trained.

---

## 1. Setup

Let the forget dataset contain prompt-answer pairs

$$
\mathcal D_f=\{(x,y)\},
$$

where $x$ is a query and $y$ is its original answer. Let $p_{\theta_0}$ be a
frozen copy of the original model and $p_\theta$ the trainable student
initialized from it.

Let $\mathcal I(x)$ be the entity-token positions in the query and
$\mathcal Q(x)$ the complete set of attended query-token positions.

Define the entity-masked query

$$
m_{\mathrm{ent}}(x)_j
=
\begin{cases}
\texttt{[MASK]}, & j\in\mathcal I(x),\\
x_j, & j\notin\mathcal I(x),
\end{cases}
$$

and the fully masked query

$$
x_{\emptyset,j}
=
\begin{cases}
\texttt{[MASK]}, & j\in\mathcal Q(x),\\
x_j, & j\notin\mathcal Q(x).
\end{cases}
$$

Both transformations act only on the query. Neither selects, removes, or masks
tokens in the original answer. The term **unconditional** below means
conditioned on $x_\emptyset$; it is not an unconditional language model in the
strict probabilistic sense.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $\mathcal D_f$ | Forget dataset of prompt-answer pairs $(x,y)$. |
| $x$ | Full original query, visible in the conditional context. |
| $\mathcal I(x)$ | Entity-token positions in the query. |
| $\mathcal Q(x)$ | Attended query-token positions. |
| $m_{\mathrm{ent}}(x)$ | Query with only positions in $\mathcal I(x)$ masked. |
| $x_\emptyset$ | Unconditional context with all positions in $\mathcal Q(x)$ masked. |
| $y$ | Original answer associated with $x$. |
| $y^{\mathrm{sf}}$ | Secret-free answer generated from $m_{\mathrm{ent}}(x)$. |
| $t$ | Diffusion time sampled by the ordinary DLLM training process. |
| $\mathcal M_t$ | Answer positions masked by the ordinary DLLM corruption process. |
| $y_t$ | Standard noised version of the ground-truth answer $y$. |
| $y_t^{\mathrm{sf}}$ | Standard noised version of the secret-free answer. |
| $\theta_0$ | Frozen original model used as the teacher. |
| $\theta$ | Trainable student initialized from $\theta_0$. |
| $p_{\mathrm{cond}}^{i,t}$ | Teacher distribution from $(x,y_t)$ with the full query visible. |
| $p_{\mathrm{uncond}}^{i,t}$ | Teacher distribution from $(x_\emptyset,y_t)$ with the full query masked. |
| $p_{\mathrm{sf}}^{i,t}$ | Structural teacher distribution from $(m_{\mathrm{ent}}(x),y_t^{\mathrm{sf}})$. |
| $s_{i,t}(v)$ | Semantic suppression score measured only on the noised ground-truth answer. |
| $q_{i,t}^*$ | KL-regularized target anchored to the secret-free distribution. |
| $\beta$ | Strength of secret-dependent token suppression. |
| $\mathcal V$ | Model vocabulary. |

---

## 3. Generate a secret-free answer

The frozen teacher first receives the entity-masked query
$m_{\mathrm{ent}}(x)$ and generates a new answer:

$$
\boxed{
y^{\mathrm{sf}}
=
\operatorname{Generate}_{\theta_0}(m_{\mathrm{ent}}(x)).
}
$$

Because only the entity is hidden, the teacher retains the non-secret query
context needed to generate a relevant and grammatically coherent answer while
lacking direct access to the query-specific identity.

The generation operator is the ordinary DLLM reverse process. It begins from
the model's normal masked answer state and progressively reveals answer tokens
according to the standard decoding schedule. Any answer masks used inside
$\operatorname{Generate}$ belong to normal DLLM generation; they are not
chosen by an unlearning selector and are not part of
$m_{\mathrm{ent}}(x)$.

The original answer $y$ is not edited token by token to create
$y^{\mathrm{sf}}$. The teacher generates a new answer from the counterfactual
query.

---

## 4. Construct two ordinary DLLM answer states

Training uses the same forward corruption process as ordinary DLLM training.
Sample a diffusion time $t$ and an answer mask $\mathcal M_t$ from the original
scheduler. Apply that same mask pattern to two clean answers:

$$
y_t
\sim
q_t(\,\cdot\mid y),
$$

and

$$
y_t^{\mathrm{sf}}
\sim
q_t(\,\cdot\mid y^{\mathrm{sf}}).
$$

Here $y_t$ is the noised ground-truth answer and $y_t^{\mathrm{sf}}$ is the
noised secret-free answer. The two answers are assumed to occupy the same
answer positions, and the same $\mathcal M_t$ is used so position $i$ has the
same diffusion status in both branches.

The mask set is not derived from token importance, prompt dependence, PCSS, or
an entity selector. It is the standard random answer mask used when training an
original DLLM.

The two answer states have different responsibilities:

- **Suppression state $y_t$:** used exclusively to measure how much each
  vocabulary token depends on the original query.
- **Structural state $y_t^{\mathrm{sf}}$:** used exclusively to construct a
  coherent secret-free reference distribution.

This separation is essential. The generated secret-free answer does not decide
which semantics should be suppressed, and the ground-truth answer does not
serve as the structural target.

---

## 5. Secret-dependence score

The suppression branch uses only the normally masked ground-truth answer
$y_t$. For each $i\in\mathcal M_t$ and $v\in\mathcal V$, the frozen teacher is
evaluated under two query conditions while $y_t$ remains fixed.

The **conditional** distribution keeps the full query visible:

$$
p_{\mathrm{cond}}^{i,t}(v)
=
p_{\theta_0}^i(v\mid x,y_t).
$$

The **unconditional** distribution masks the full query:

$$
p_{\mathrm{uncond}}^{i,t}(v)
=
p_{\theta_0}^i(v\mid x_\emptyset,y_t).
$$

Because both passes use the same masked ground-truth answer, their difference
isolates the probability contributed by the visible query. Define the positive
conditional log-probability shift

$$
s_{i,t}(v)
=
\left[
\log p_{\mathrm{cond}}^{i,t}(v)
-
\log p_{\mathrm{uncond}}^{i,t}(v)
\right]_+
=
\left[
\log
\frac{p_{\mathrm{cond}}^{i,t}(v)}
     {p_{\mathrm{uncond}}^{i,t}(v)}
\right]_+.
$$

This score is evaluated over the whole vocabulary:

- $s_{i,t}(v)\approx0$ means that revealing the query does not make token $v$
  more likely.
- $s_{i,t}(v)\gg0$ means that token $v$ receives substantial probability from
  the original query and should be suppressed more strongly.

Only positive shifts are penalized. A token is not rewarded merely because the
original query makes it less likely.

The secret-free answer $y^{\mathrm{sf}}$ and its noised state
$y_t^{\mathrm{sf}}$ do not appear in $s_{i,t}(v)$. They play no role in
measuring how much semantic information should be forgotten.

---

## 6. KL-regularized selective-unlearning objective

The secret-free branch supplies a structural reference distribution:

$$
p_{\mathrm{sf}}^{i,t}(v)
=
p_{\theta_0}^i
\left(v\mid m_{\mathrm{ent}}(x),y_t^{\mathrm{sf}}\right).
$$

Unlike the suppression score, this distribution is evaluated on the noised
secret-free answer. Its purpose is to give the target a coherent linguistic
shape so that suppressing the forgotten semantics does not push the student
toward arbitrary or gibberish output.

The target distribution is defined by

$$
\boxed{
q_{i,t}^*
=
\underset{q\in\Delta(\mathcal V)}{\arg\min}
\left[
D_{\mathrm{KL}}
\left(
q\,\|\,p_{\mathrm{sf}}^{i,t}
\right)
+
\beta\,
\mathbb E_{v\sim q}
\left[s_{i,t}(v)\right]
\right].
}
$$

No local-preservation loss, retain loss, ground-truth demotion loss, or
PCSS-selected answer-token loss is part of this formulation.

The two terms in the objective have complementary roles:

- $D_{\mathrm{KL}}(q\|p_{\mathrm{sf}}^{i,t})$ keeps the target close to the
  frozen teacher's distribution on the secret-free answer state, preserving
  coherent answer structure.
- $\beta\mathbb E_{v\sim q}[s_{i,t}(v)]$ removes probability mass from tokens
  whose likelihood is increased by the full visible query on the masked
  ground-truth answer.

The first term prevents unstructured degeneration. The second term determines
how much query-dependent semantics should be forgotten. Neither term replaces
the other.

### 6.1 Closed-form target

Introduce a Lagrange multiplier $\lambda$ for the constraint
$\sum_v q(v)=1$. The Lagrangian is

$$
\mathcal J(q,\lambda)
=
\sum_{v\in\mathcal V}
q(v)
\left[
\log\frac{q(v)}{p_{\mathrm{sf}}^{i,t}(v)}
+
\beta s_{i,t}(v)
\right]
+
\lambda
\left(
\sum_{v\in\mathcal V}q(v)-1
\right).
$$

Stationarity with respect to $q(v)$ gives

$$
\log\frac{q(v)}{p_{\mathrm{sf}}^{i,t}(v)}
+1+\beta s_{i,t}(v)+\lambda=0.
$$

Therefore

$$
q(v)
\propto
p_{\mathrm{sf}}^{i,t}(v)
\exp[-\beta s_{i,t}(v)],
$$

and the normalized solution is

$$
\boxed{
q_{i,t}^*(v)
=
\frac{
p_{\mathrm{sf}}^{i,t}(v)
\exp[-\beta s_{i,t}(v)]
}{
\displaystyle
\sum_{u\in\mathcal V}
p_{\mathrm{sf}}^{i,t}(u)
\exp[-\beta s_{i,t}(u)]
}.
}
$$

Equivalently,

$$
q_{i,t}^*(v)
\;\propto\;
\underbrace{
  p_{\mathrm{sf}}^{i,t}(v)
}_{
  \substack{
    \text{secret-free}\\
    \text{structural anchor}
  }
}
\qquad\times\qquad
\underbrace{
  \exp[-\beta s_{i,t}(v)]
}_{
  \substack{
    \text{secret-dependent}\\
    \text{semantic suppression}
  }
}.
$$

This closed form is the solution of the stated objective; it does not introduce
an additional unlearning loss.

---

## 7. Training the student with the target

The student receives the original query $x$ and the ordinarily corrupted
ground-truth answer $y_t$. At positions in the standard DLLM mask set
$\mathcal M_t$, $q_{i,t}^*$ replaces the one-hot ground-truth target with a soft
target distribution.

The student is fitted to this soft target with the ordinary DLLM denoising
training rule. Since $q_{i,t}^*$ is fixed with respect to $\theta$, soft-label
cross-entropy and forward KL target matching differ only by the constant
entropy of $q_{i,t}^*$.

The diffusion time sampling, answer-mask sampling, and masked-position
reduction remain the ordinary DLLM training procedure. The only change is the
denoising target: the student learns the counterfactual distribution $q^*$
instead of reconstructing the original query-dependent answer.

---

## 8. Effect of the suppression strength

The parameter $\beta\geq0$ controls the additional suppression applied to
secret-dependent vocabulary tokens:

$$
q_{i,t}^*(v)
\propto
p_{\mathrm{sf}}^{i,t}(v)e^{-\beta s_{i,t}(v)}.
$$

| Regime | Theoretical behavior |
|---:|---|
| $\beta=0$ | $q_{i,t}^*=p_{\mathrm{sf}}^{i,t}$; training follows only the secret-free structural anchor. |
| Small $\beta>0$ | Tokens mildly supported by the visible query receive additional suppression. |
| Large $\beta$ | The target concentrates on tokens whose probabilities do not increase when the full query is revealed. |

The secret-free base supplies coherent answer structure. Increasing $\beta$
suppresses tokens whose probability rises when the full query is restored in
the ground-truth answer state. If $\beta$ is too large, the target may become
overly generic because it favors only the lowest-dependence region of the
vocabulary.

---

## 9. Why the target is broader than ground-truth deletion

Ground-truth demotion only decreases the probability of a labeled answer token.
It does not directly constrain aliases, alternate tokenizations, or related
continuations that express the same secret.

Selective Unlearning computes $s_{i,t}(v)$ for every
$v\in\mathcal V$. Consequently, any vocabulary token whose probability is
increased by the visible query is downweighted in $q_{i,t}^*$, even when it
does not equal a token in the original answer.

The method therefore aims to remove query-dependent output behavior rather
than delete a predefined set of answer tokens.

---

## 10. Separation of the query and answer states

The formulation uses two query transformations and two answer states that
should not be conflated:

1. **Entity intervention:** $m_{\mathrm{ent}}(x)$ masks only the query entity.
   It preserves the remaining query context used to generate a relevant
   secret-free answer.
2. **Unconditional baseline:** $x_\emptyset$ masks the complete query. It is
   used only in the conditional-versus-unconditional comparison that measures
   how much query-dependent semantics to suppress.
3. **Secret-free answer generation:** the frozen teacher uses the ordinary DLLM
   reverse process, whose answer slots follow the model's normal decoding
   schedule.
4. **Ground-truth suppression state:** the ordinary DLLM forward process creates
   $y_t$ from $y$. Only this state is used in
   $p_{\mathrm{cond}}^{i,t}/p_{\mathrm{uncond}}^{i,t}$ and therefore in
   $s_{i,t}(v)$.
5. **Secret-free structural state:** the same ordinary mask pattern creates
   $y_t^{\mathrm{sf}}$ from $y^{\mathrm{sf}}$. This state is used only in
   $p_{\mathrm{sf}}^{i,t}$, the KL anchor.

The query comparison determines what to suppress. The secret-free answer keeps
the learned distribution structured. Both answer corruptions are unchanged
DLLM mechanics; there is no separate answer-token selector, importance mask,
or selective answer-masking stage in the unlearning theory.
