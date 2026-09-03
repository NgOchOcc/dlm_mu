# Semantic_Unlearn for Masked Diffusion Language Models

## 1. Overview

**Semantic_Unlearn** is a forget-only unlearning objective for masked discrete diffusion language models (DLLMs). It starts from an original model, freezes a copy as the teacher $p_{\theta_0}$, and updates a student $p_\theta$ on normal diffusion states sampled from the forget answers.

For every masked answer position, the method constructs a valid vocabulary distribution from the frozen teacher and exponentially downweights tokens whose frozen token embeddings are close to the forgotten gold token. The student then minimizes forward KL from this semantic target:

$$
q_i^*(v)
\propto
p_{\theta_0}(v\mid x_f,y_t)
\exp\!\left[-\eta k_i(v)\right].
$$

Unlike exact-token deletion, this target suppresses the local semantic neighborhood around the forgotten answer token. Unlike gradient ascent, it remains a normalized distribution and gives the student an explicit replacement target. The method uses only the forget dataset: it has no retain loss, no standard denoising cross-entropy term, and no separate retain examples.

The implementation is in [`src/trainer/dlm/unlearn/semantic_unlearn.py`](../src/trainer/dlm/unlearn/semantic_unlearn.py). The default trainer and TOFU experiment configurations are [`configs/trainer/DiffusionSemanticUnlearn.yaml`](../configs/trainer/DiffusionSemanticUnlearn.yaml) and [`configs/experiment/unlearn/tofu/dlm_semantic_unlearn.yaml`](../configs/experiment/unlearn/tofu/dlm_semantic_unlearn.yaml).

---

## 2. Motivation

### 2.1 Exact-token suppression can move probability to a synonym

Suppressing only the gold token does not necessarily remove the underlying fact. Probability can move to a synonym, spelling variant, or another token in the same semantic region. Semantic_Unlearn defines that region through the original model's frozen input-token embedding geometry.

### 2.2 Unbounded ascent does not define replacement behavior

Maximizing the forgotten answer's negative log-likelihood can continue without a finite target. Semantic_Unlearn instead edits the original teacher distribution and renormalizes it. Training is therefore a bounded forward-KL fit to a well-defined probability distribution.

### 2.3 DLLM associations exist across diffusion states

A masked DLLM denoises partially masked sequences. Semantic_Unlearn samples a random diffusion time and applies its objective to the masked answer positions in that state, exposing the student to the same family of states used by ordinary DLLM training.

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| $\mathcal D_f$ | Forget dataset of prompt-answer pairs $(x_f,y_f)$. |
| $t$ | Random diffusion time sampled from the trainer's normal interval. |
| $\mathcal M_t$ | Answer-token positions masked at time $t$. |
| $y_t$ | Partially masked sequence obtained from $y_f$ at time $t$. |
| $\theta_0$ | Frozen original model used as the teacher and embedding source. |
| $\theta$ | Trainable student initialized from the original model. |
| $\mathcal V$ | Tokenizer vocabulary; padded LM-head-only rows are excluded. |
| $E_v$ | Frozen input-token embedding of vocabulary token $v$ from $\theta_0$. |
| $c_i=E_{y_{f,i}}$ | Semantic center at answer position $i$. |
| $d_i(v)^2$ | Squared Euclidean distance $\lVert E_v-c_i\rVert_2^2$. |
| $h_i$ | Per-center median-heuristic RBF bandwidth. |
| $k_i(v)$ | RBF similarity between vocabulary token $v$ and center $c_i$. |
| $\eta$ | Nonnegative semantic suppression strength. |
| $q_i^*$ | Normalized semantic target at masked position $i$. |

---

## 4. Method

### 4.1 Sample a normal DLLM noising state

For each forget example $(x_f,y_f)$, sample

$$
t\sim\mathcal U(\epsilon,1),
$$

convert $t$ to a masking probability through the configured diffusion scheduler, and independently mask answer positions to form $y_t$. Prompt tokens remain visible. The student and frozen teacher are evaluated on the **same** $y_t$ and attention mask.

The implementation reuses [`DiffusionFinetuneTrainer._sample_and_denoise`](../src/trainer/dlm/base.py), including LLaDA/Dream preprocessing and right-shift handling.

### 4.2 Define a frozen semantic center

At every masked answer position $i\in\mathcal M_t$, use the clean gold token id retained by the batch and set

$$
c_i=E_{y_{f,i}}.
$$

$E$ is the original model's frozen input-embedding matrix, restricted to the tokenizer vocabulary. The geometry is detached from autograd, so the target cannot be optimized by moving the embedding matrix.

### 4.3 Compute the per-center median bandwidth

For one center $c_i$, compute its squared distance to every eligible vocabulary embedding:

$$
d_i(v)^2=\lVert E_v-c_i\rVert_2^2,
\qquad v\in\mathcal V.
$$

Semantic_Unlearn follows the repository's SVGD-style median heuristic. Let

$$
m_i^2=\operatorname{median}_{v\in\mathcal V}d_i(v)^2,
\qquad
h_i=\frac{m_i^2}{\log |\mathcal V|}.
$$

The RBF kernel is

$$
k_i(v)=\exp\!\left(-\frac{d_i(v)^2}{h_i}\right).
$$

Thus $k_i(y_{f,i})=1$, nearby tokens receive values near one, and distant tokens approach zero. If the median degenerates to zero or is non-finite, the implementation falls back to

$$
h_i=2\sigma_{\mathrm{fallback}}^2.
$$

The median is exact over the full tokenizer vocabulary for each distinct center token in the batch. Vocabulary computations are chunked and repeated gold-token centers are deduplicated to control memory without changing the result.

### 4.4 Construct the semantic target

Let

$$
p_i^0(v)=p_{\theta_0}(v\mid x_f,y_t)
$$

be the frozen teacher distribution at masked position $i$. Define the unnormalized target

$$
\widetilde q_i(v)
=
p_i^0(v)\exp[-\eta k_i(v)]
$$

and partition function

$$
Z_i
=
\sum_{v'\in\mathcal V}
p_i^0(v')\exp[-\eta k_i(v')].
$$

The target is

$$
q_i^*(v)
=
\frac{1}{Z_i}
p_i^0(v)\exp[-\eta k_i(v)].
$$

The implementation evaluates this in log space:

$$
\log q_i^*(v)
=
\log p_i^0(v)-\eta k_i(v)
-
\operatorname{logsumexp}_{v'}
\left[\log p_i^0(v')-\eta k_i(v')\right].
$$

This preserves the teacher's support and ordering away from the semantic neighborhood while reducing the odds of high-kernel tokens relative to distant tokens. In particular, $\eta=0$ recovers $q_i^*=p_i^0$. Both $q_i^*$ and $\log q_i^*$ are detached before the student loss is computed.

### 4.5 Optimize forward KL on masked answer positions

For one example and noising state, the loss is exactly

$$
\mathcal L_{\mathrm{Semantic\_Unlearn}}
=
\frac{1}{|\mathcal M_t|}
\sum_{i\in\mathcal M_t}
D_{\mathrm{KL}}
\left(
q_i^*
\;\|\;
p_\theta(\cdot\mid x_f,y_t)_i
\right),
$$

where

$$
D_{\mathrm{KL}}(q_i^*\|p_i^\theta)
=
\sum_{v\in\mathcal V}
q_i^*(v)
\left[
\log q_i^*(v)-\log p_i^\theta(v)
\right].
$$

The training objective is the expectation over forget examples and sampled diffusion times. Per-example masked-position means are averaged across active examples. A rare sample with no masked answer token contributes no loss; an all-empty batch returns a differentiable zero.

The explicit $1/|\mathcal M_t|$ definition means scheduler/CART token weights are not applied to this KL, although the configured scheduler still determines the noising state.

---

## 5. Pseudocode

```text
Input:
    forget batch (x_f, y_f)
    trainable student p_theta
    frozen teacher p_theta0
    frozen teacher input embeddings E
    suppression strength eta

Sample one random diffusion time per example
Mask answer tokens to obtain y_t and masked set M_t
Run the student on (x_f, y_t)
Run the frozen teacher on the identical (x_f, y_t)

For every distinct gold token g appearing in M_t:
    c = E[g]
    d2[v] = ||E[v] - c||^2 for every tokenizer token v
    h = median_v(d2[v]) / log(|V|)
    if h is zero or non-finite:
        h = 2 * sigma_fallback^2
    kernel[v] = exp(-d2[v] / h)

    For each masked position i whose gold token is g:
        log_q_unnorm[v] = log p_theta0(v | x_f, y_t)_i
                            - eta * kernel[v]
        log_q[v] = log_q_unnorm[v] - logsumexp(log_q_unnorm)
        q[v] = exp(log_q[v])
        kl[i] = sum_v q[v] * (log_q[v] - log p_theta[v])

For each active example:
    average kl[i] over i in M_t
Average the example losses
Update only theta
```

---

## 6. Numerical and distributed-training details

- **Tokenizer support:** teacher, target, and student distributions are normalized over `len(tokenizer)`, not the wider padded LM head used by LLaDA and Dream.
- **Float32 target math:** log-softmax, distances, bandwidths, kernels, normalization, and KL accumulation use float32 even when the model runs in bfloat16.
- **Stable distances:** squared distances are computed as $\lVert E_v\rVert^2+\lVert c\rVert^2-2E_v^\top c$ and clamped at zero to remove roundoff negatives.
- **Memory bound:** the code never materializes a tensor shaped `[masked_positions, vocabulary, hidden_size]`. It chunks the vocabulary matrix during distance computation and chunks distinct center tokens during target construction.
- **Repeated centers:** masked positions with the same gold token reuse one kernel and bandwidth in the current batch.
- **ZeRO-3:** the frozen teacher's sharded input embedding is materialized with `deepspeed.zero.GatheredParameters` and cloned inside the gather context.
- **Fixed target:** all teacher and semantic-target operations run under `torch.no_grad()`; gradients flow only through the student log-probabilities.
- **Shared state:** teacher and student always receive the same noised token ids, preventing target differences caused by independent masking draws.

---

## 7. Configuration

The main method arguments are:

| Argument | Default | Purpose |
|---|---:|---|
| `eta` | `1.0` | Strength of semantic downweighting. `0` leaves the teacher target unchanged. |
| `sigma_fallback` | `0.05` | Fixed RBF scale used only when a per-center median bandwidth degenerates. |
| `center_chunk_size` | `8` | Number of distinct gold-token centers processed together. |
| `vocab_chunk_size` | `4096` | Vocabulary rows converted to float32 together during distance computation. |

The TOFU experiment is retain-free: [`dlm_semantic_unlearn.yaml`](../configs/experiment/unlearn/tofu/dlm_semantic_unlearn.yaml) removes `data.retain` entirely.

---

## 8. Running on TOFU

Use the method-specific launcher:

```bash
bash scripts/dlm/tofu/run_semantic_unlearn.sh
```

Override the semantic strength or chunk sizes with environment variables:

```bash
ETA=2.0 CENTER_CHUNK_SIZE=4 VOCAB_CHUNK_SIZE=2048 \
  bash scripts/dlm/tofu/run_semantic_unlearn.sh
```

The launcher delegates to [`scripts/dlm/tofu/unlearn.sh`](../scripts/dlm/tofu/unlearn.sh), so normal controls such as `UNLEARN_EPOCHS`, `TRAIN_BS`, `GRAD_ACCUM`, `NO_EVAL`, and `TASK_SUFFIX` remain available.

---

## 9. Implementation map

| Component | File |
|---|---|
| Algorithm and target construction | [`src/trainer/dlm/unlearn/semantic_unlearn.py`](../src/trainer/dlm/unlearn/semantic_unlearn.py) |
| DLLM noising and Dream/LLaDA handling | [`src/trainer/dlm/base.py`](../src/trainer/dlm/base.py) |
| Frozen teacher and shared reduction | [`src/trainer/dlm/unlearn/base.py`](../src/trainer/dlm/unlearn/base.py) |
| Package export | [`src/trainer/dlm/unlearn/__init__.py`](../src/trainer/dlm/unlearn/__init__.py) |
| Trainer registry | [`src/trainer/__init__.py`](../src/trainer/__init__.py) |
| Trainer defaults | [`configs/trainer/DiffusionSemanticUnlearn.yaml`](../configs/trainer/DiffusionSemanticUnlearn.yaml) |
| TOFU experiment | [`configs/experiment/unlearn/tofu/dlm_semantic_unlearn.yaml`](../configs/experiment/unlearn/tofu/dlm_semantic_unlearn.yaml) |
| TOFU launcher | [`scripts/dlm/tofu/run_semantic_unlearn.sh`](../scripts/dlm/tofu/run_semantic_unlearn.sh) |
