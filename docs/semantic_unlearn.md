# Semantic_Unlearn for Masked Diffusion Language Models

Semantic_Unlearn is a forget-only objective for masked discrete diffusion
language models (DLLMs). It freezes the original model as a teacher
$p_{\theta_0}$, constructs a semantic target distribution at masked forget-answer
positions, and trains a student $p_\theta$ to match that target.

This document separates the mathematical definition from the code and runtime
details:

- [Part I: Theory](#part-i-theory) defines the target distributions and loss.
- [Part II: Implementation](#part-ii-implementation) explains configuration,
  numerical details, and TOFU usage.

The method uses only the forget dataset. It has no retain loss, ordinary
denoising cross-entropy term, or separate retain examples.

---

# Part I: Theory

## 1. Motivation

Exact-token suppression can move probability to a synonym, spelling variant, or
another token in the same semantic region. Semantic_Unlearn instead changes the
teacher distribution according to the frozen input-embedding relationship
between each vocabulary token and the forgotten gold token.

Unlike unbounded gradient ascent, the method supplies a finite, normalized
replacement distribution. Because the target is evaluated on randomly noised
answer sequences, the objective covers the diffusion states encountered during
normal DLLM denoising.

## 2. Supported target distributions

For a masked answer position $i$, let

$$
p_i^0(v)=p_{\theta_0}(v\mid x_f,y_t),
\qquad
c_i=E_{y_{f,i}},
$$

where $E$ is the frozen teacher input-embedding matrix. Every mode uses

$$
q_i^*(v)
=
\frac{1}{Z_i}p_i^0(v)\exp[-\eta s_i(v)],
\qquad
Z_i=\sum_{v'\in\mathcal V}p_i^0(v')\exp[-\eta s_i(v')].
$$

The supported choices for $s_i(v)$ are:

| `target_mode` | Semantic score $s_i(v)$ | Target distribution | Interpretation |
|---|---|---|---|
| `rbf` (default) | $\displaystyle k_i(v)=\exp\!\left(-\frac{\lVert E_v-c_i\rVert_2^2}{h_i}\right)$ | $\displaystyle q_i^*(v)=\frac{1}{Z_i}p_i^0(v)\exp[-\eta k_i(v)]$ | Downweights tokens inside the RBF neighborhood of the gold token. |
| `cosine` | $\displaystyle \cos(E_v,c_i)-1$ | $\displaystyle q_i^*(v)=\frac{1}{Z_i}p_i^0(v)\exp\{-\eta[\cos(E_v,c_i)-1]\}$ | Downweights higher-cosine tokens relative to lower-cosine tokens, without an RBF bandwidth. |

The `cosine` formula is implemented exactly as shown. Its constant `-1` adds the
same $+\eta$ to every unnormalized log weight, so it cancels in $Z_i$. Therefore
the normalized distribution is equivalently

$$
q_i^*(v)\propto p_i^0(v)\exp[-\eta\cos(E_v,c_i)].
$$

Keeping the shifted form in the definition makes the self-score zero because
$\cos(c_i,c_i)-1=0$.

## 3. Notation

| Symbol | Meaning |
|---|---|
| $\mathcal D_f$ | Forget dataset of prompt-answer pairs $(x_f,y_f)$. |
| $t$ | Random diffusion time sampled from the trainer's normal interval. |
| $\mathcal M_t$ | Answer-token positions masked at time $t$. |
| $y_t$ | Partially masked sequence obtained from $y_f$ at time $t$. |
| $\theta_0$ | Frozen original model used as teacher and embedding source. |
| $\theta$ | Trainable student initialized from the original model. |
| $\mathcal V$ | Tokenizer vocabulary; padded LM-head-only rows are excluded. |
| $E_v$ | Frozen input embedding of vocabulary token $v$. |
| $c_i=E_{y_{f,i}}$ | Semantic center at masked answer position $i$. |
| $s_i(v)$ | Mode-specific semantic score from the table above. |
| $\eta$ | Nonnegative target-tilt strength. |
| $q_i^*$ | Normalized target distribution at position $i$. |

## 4. Target construction

### 4.1 Shared diffusion state

For each forget example, sample

$$
t\sim\mathcal U(\epsilon,1),
$$

convert $t$ to a mask probability with the configured diffusion scheduler, and
mask answer positions to obtain $y_t$. Prompt tokens remain visible. The student
and frozen teacher receive the same $y_t$ and attention mask.

### 4.2 Frozen semantic center

At each $i\in\mathcal M_t$, the clean gold token defines

$$
c_i=E_{y_{f,i}}.
$$

The embedding matrix is taken from the frozen teacher and detached from
autograd, so the target cannot change by moving its geometry.

### 4.3 RBF score

The RBF mode computes

$$
d_i(v)^2=\lVert E_v-c_i\rVert_2^2,
\qquad
m_i^2=\operatorname{median}_{v\in\mathcal V}d_i(v)^2,
\qquad
h_i=\frac{m_i^2}{\log|\mathcal V|},
$$

then uses

$$
s_i(v)=k_i(v)=\exp\!\left(-\frac{d_i(v)^2}{h_i}\right).
$$

If $h_i$ is zero or non-finite, it falls back to
$2\sigma_{\mathrm{fallback}}^2$. The median is exact over the tokenizer
vocabulary.

### 4.4 Shifted-cosine score

The cosine mode normalizes both embedding vectors and uses

$$
s_i(v)
=
\frac{E_v^\top c_i}{\lVert E_v\rVert_2\lVert c_i\rVert_2}-1.
$$

This score lies in $[-2,0]$ up to numerical precision. It needs no bandwidth or
fallback scale.

### 4.5 Normalization and loss

The implementation constructs the target in log space:

$$
\log q_i^*(v)
=
\log p_i^0(v)-\eta s_i(v)
-
\operatorname{logsumexp}_{v'}
\left[\log p_i^0(v')-\eta s_i(v')\right].
$$

When $\eta=0$, both modes recover the teacher distribution. The target and its
log probabilities are detached before evaluating the forward KL objective:

$$
\mathcal L_{\mathrm{Semantic\_Unlearn}}
=
\frac{1}{|\mathcal M_t|}
\sum_{i\in\mathcal M_t}
D_{\mathrm{KL}}\!\left(q_i^*\,\|\,p_\theta(\cdot\mid x_f,y_t)_i\right).
$$

Per-example masked-position means are averaged across active examples. A sample
with no masked answer token contributes no loss. Scheduler/CART token weights
are not applied to this KL, although the scheduler still determines $y_t$.

---

# Part II: Implementation

## 5. Selecting a target mode

The trainer arguments live in
[`configs/trainer/DiffusionSemanticUnlearn.yaml`](../configs/trainer/DiffusionSemanticUnlearn.yaml):

| Argument | Default | Applies to | Purpose |
|---|---:|---|---|
| `target_mode` | `rbf` | Both | Selects `rbf` or `cosine`. |
| `eta` | `1.0` | Both | Strength of the target tilt; `0` distills the unchanged teacher. |
| `sigma_fallback` | `0.05` | RBF only | Fixed scale used when a median bandwidth degenerates. |
| `center_chunk_size` | `8` | Both | Distinct gold-token centers processed together. |
| `vocab_chunk_size` | `4096` | Both | Vocabulary embedding rows converted to float32 together. |

Hydra overrides can select the new mode directly:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/dlm_semantic_unlearn \
  trainer.method_args.target_mode=cosine
```

An unsupported value fails during trainer initialization with a `ValueError`.

## 6. Execution flow

```text
Input: forget batch, student, frozen teacher, frozen teacher embeddings E

1. Sample one diffusion time per example and construct y_t.
2. Run student and teacher on the same y_t.
3. Deduplicate gold-token centers among masked answer positions.
4. For each center chunk:
   a. RBF mode: compute full-vocabulary distances, bandwidth, and RBF scores.
   b. Cosine mode: normalize embeddings and compute shifted-cosine scores.
5. Form log q = log p_theta0 - eta * score and normalize with logsumexp.
6. Compute KL(q || p_theta) at masked answer positions.
7. Average within each active example, then across examples; update only theta.
```

The code reuses
[`DiffusionFinetuneTrainer._sample_and_denoise`](../src/trainer/dlm/base.py),
including LLaDA/Dream preprocessing and right-shift behavior.

## 7. Numerical and distributed behavior

- Teacher, target, and student distributions are restricted to
  `len(tokenizer)`, excluding padded LM-head-only rows.
- Log-softmax, semantic-score construction, normalization, and KL accumulation
  use float32 even when the model uses bfloat16.
- Cosines are clamped to $[-1,1]$ after the normalized matrix product to remove
  floating-point overshoot.
- RBF squared distances are clamped at zero to remove roundoff negatives.
- Vocabulary and center chunking avoid a
  `[masked_positions, vocabulary, hidden_size]` tensor without approximating the
  target.
- Repeated gold-token centers reuse one score vector within a batch.
- Under ZeRO-3, the frozen input embedding is materialized with
  `deepspeed.zero.GatheredParameters`.
- Teacher and target operations run under `torch.no_grad()`; gradients flow only
  through student log probabilities.

## 8. Running on TOFU

RBF remains the launcher default:

```bash
bash scripts/dlm/tofu/run_semantic_unlearn.sh
```

Select the shifted-cosine target with:

```bash
TARGET_MODE=cosine ETA=2.0 \
  bash scripts/dlm/tofu/run_semantic_unlearn.sh
```

The launcher passes extra arguments to
[`scripts/dlm/tofu/unlearn.sh`](../scripts/dlm/tofu/unlearn.sh), so controls such
as `UNLEARN_EPOCHS`, `TRAIN_BS`, `GRAD_ACCUM`, `NO_EVAL`, and `TASK_SUFFIX`
remain available. The existing RBF suffix is preserved; the cosine default adds
`_cosine_` to keep the two modes' runs distinct.

The TOFU experiment configuration is
[`configs/experiment/unlearn/tofu/dlm_semantic_unlearn.yaml`](../configs/experiment/unlearn/tofu/dlm_semantic_unlearn.yaml).
It removes `data.retain` entirely.

## 9. Source map

| Responsibility | File |
|---|---|
| Target scores, normalization, and loss | [`src/trainer/dlm/unlearn/semantic_unlearn.py`](../src/trainer/dlm/unlearn/semantic_unlearn.py) |
| DLLM noising and Dream/LLaDA handling | [`src/trainer/dlm/base.py`](../src/trainer/dlm/base.py) |
| Frozen teacher and reduction | [`src/trainer/dlm/unlearn/base.py`](../src/trainer/dlm/unlearn/base.py) |
| Trainer defaults | [`configs/trainer/DiffusionSemanticUnlearn.yaml`](../configs/trainer/DiffusionSemanticUnlearn.yaml) |
| TOFU experiment | [`configs/experiment/unlearn/tofu/dlm_semantic_unlearn.yaml`](../configs/experiment/unlearn/tofu/dlm_semantic_unlearn.yaml) |
| TOFU launcher | [`scripts/dlm/tofu/run_semantic_unlearn.sh`](../scripts/dlm/tofu/run_semantic_unlearn.sh) |
