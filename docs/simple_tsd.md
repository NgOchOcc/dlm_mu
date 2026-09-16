# Simple Token Selection and Trajectory-Selective Distillation

This document separates the mathematical method from its configuration and
source-code details:

- [Part I: Theory](#part-i-theory) explains the motivation, token selection,
  target construction, and training objective.
- [Part II: Implementation](#part-ii-implementation) covers shipped defaults,
  implementation differences, variants, and source locations.

---

# Part I: Theory

## 1. Overview

**Simple Trajectory-Selective Distillation (SimpleTSD)** is a retain-set-free unlearning method for masked discrete diffusion language models (DLLMs).

The method starts from a pretrained or fine-tuned model, freezes a copy as the teacher $p_{\theta_0}$, and trains another copy $p_\theta$ to:

1. forget answer tokens whose prediction depends strongly on the forget prompt; and
2. preserve the teacher's behavior on the remaining answer tokens.

SimpleTSD therefore separates an answer into two kinds of positions:

- **forget positions**, which carry prompt-specific information that should be removed; and
- **self-distillation positions**, which preserve grammar, fluency, and non-sensitive content.

---

## 2. What problem does SimpleTSD solve?

### 2.1 Whole-answer unlearning is too broad

A forget-set answer usually contains both fact-bearing content and generic language. For example:

> "The author was born in London in 1984."

Only some tokens identify the fact to erase. Function words, punctuation, and predictable syntax should normally remain unchanged. Applying the forget objective to every answer token can damage fluency and unrelated capabilities.

SimpleTSD therefore asks:

> Which answer positions are specifically made predictable by the forget question?

It uses a prompt-dependence score to select only those positions.

### 2.2 Exact-token suppression does not define a useful replacement

Simply maximizing the loss of the original answer can continuously push probability away from the gold token without specifying a well-formed alternative. This may produce unstable training, arbitrary token redistribution, or degraded generation.

SimpleTSD instead constructs an **edited teacher target**. At a selected forget position, it:

1. takes the frozen teacher distribution;
2. removes the probability assigned to the forgotten gold token; and
3. renormalizes the remaining distribution.

The student is trained toward this valid probability distribution rather than trained with an unbounded ascent objective.

### 2.3 Retain-free training still needs a utility constraint

Many unlearning objectives preserve utility using a separate retain dataset. SimpleTSD is designed to operate on the forget dataset alone.

It obtains its preservation signal from the same forget example: masked answer positions that were **not** selected for forgetting are trained to match the frozen teacher. This local self-distillation protects surrounding language behavior without loading a retain corpus.

### 2.4 DLLM behavior must be changed across noising states

A masked DLLM generates by denoising partially masked sequences. Forgetting only one fixed state, such as the fully masked answer, does not necessarily remove the association from other points on the denoising trajectory.

SimpleTSD repeatedly samples normal diffusion mask states and applies its forget and preservation objectives within those states. The method therefore targets behavior along the model's denoising trajectory rather than at only one input pattern.

---

## 3. Method at a glance

SimpleTSD has two stages:

1. **Prompt-Conditional Score Selection (PCSS)** identifies the answer positions to forget.
2. **Trajectory-selective distillation** trains on sampled diffusion states:
   - an edited teacher target is used at selected forget positions; and
   - ordinary teacher-to-student distillation is used at the other masked answer positions.

```text
forget question x + answer y
             │
             ▼
compare teacher predictions with and without x
             │
             ▼
select prompt-dependent positions Â
             │
             ▼
sample a partially masked diffusion state y_M
             │
             ├── F = M ∩ Â
             │     fit student to a gold-demoted teacher target
             │
             └── R = M \ Â
                   fit student to the original teacher distribution
             │
             ▼
combine the two losses and update the student
```

---

## 4. Notation

| Symbol | Meaning |
|---|---|
| $\mathcal D_f$ | Forget dataset containing question-answer pairs $(x,y)$. |
| $x$ | Forget question or prompt. |
| $y=(y_1,\ldots,y_L)$ | Answer containing $L$ tokens. |
| $\widetilde y$ | Fully masked answer in the simplified selection algorithm. |
| $\theta_0$ | Frozen original model used as the teacher. |
| $\theta$ | Trainable student model initialized from $\theta_0$. |
| $x_\varnothing$ | Input with the prompt content removed or replaced by masks. |
| $s_i$ | Prompt-dependence score for answer token $y_i$. |
| $\tau$ | Fixed selection threshold, when fixed-threshold selection is used. |
| $\widehat{\mathcal A}$ | Selected forget-token positions. |
| $\mathcal M$ | Answer-token positions masked in one denoising state. |
| $y_{\mathcal M}$ | Partially masked answer produced from $\mathcal M$. |
| $p^0_{\mathcal M,i}$ | Frozen teacher distribution at masked position $i$. |
| $p^\theta_{\mathcal M,i}$ | Student distribution at masked position $i$. |
| $q_{\mathcal M,i}$ | Edited teacher target at a selected forget position. |
| $F$ | Masked forget positions, $F=\mathcal M\cap\widehat{\mathcal A}$. |
| $R$ | Masked self-distillation positions, $R=\mathcal M\setminus\widehat{\mathcal A}$. |
| $\rho$ | Mask rate obtained from the DLLM diffusion schedule. |
| $\alpha$ | Weight assigned to the forget loss. The default is $0.5$. |
| $\epsilon$ | Small numerical-stability constant, such as $10^{-8}$. |
| $\mathcal V$ | Model vocabulary. |

---

## 5. Stage 1: select the positions to forget

### 5.1 Prompt-dependence score

For each answer position $i$, SimpleTSD compares the frozen teacher's probability of the gold token in two contexts:

- **conditional context:** the question is visible; and
- **promptless context:** the question content is blanked.

The score is

$$
s_i =
\mathbb E_{\mathcal M:i\in\mathcal M}
\left[
\log p_{\theta_0}(y_i\mid x,y_{\mathcal M})
-
\log p_{\theta_0}(y_i\mid x_\varnothing,y_{\mathcal M})
\right].
$$

A high score means that seeing the question substantially increases the teacher's belief in token $y_i$. Such a token is treated as prompt-specific and is a candidate for forgetting.

The current implementation estimates the expectation with `sel_mc` partial-mask draws. Partial masking is intentional: under a fully masked answer, the promptless teacher can become nearly uniform, causing almost every answer token to appear prompt-dependent.

The selection implementation is in [`_select_forget_positions`](../src/trainer/dlm/unlearn/simple_tsd.py#L163-L200).

### 5.2 Convert scores into a selected set

The selected set is

$$
\widehat{\mathcal A}
=
\{i : y_i\text{ is selected by the configured score rule}\}.
$$

Three rules are supported:

- **`otsu`** — the default; computes an adaptive threshold separately for each example;
- **`topk`** — selects the configured top fraction of answer positions; and
- **`tau`** — selects positions satisfying $s_i\geq\tau$.

Otsu and threshold selection are implemented in [`_otsu_threshold` and `_threshold_scores`](../src/trainer/dlm/unlearn/simple_tsd.py#L202-L237).

When `drop_question_overlap=true`, answer words that merely repeat words already present in the question are removed from $\widehat{\mathcal A}$. The prompt already exposes those surface forms, so treating them as hidden information can add unnecessary forgetting pressure. See [`_question_overlap_mask`](../src/trainer/dlm/unlearn/simple_tsd.py#L239-L263).

### 5.3 Manual-span option

If token-level forget spans are already known, selection can be skipped. A supplied answer mask is used directly as $\widehat{\mathcal A}$, intersected with valid answer positions.

This path is implemented in [`_forget_context`](../src/trainer/dlm/unlearn/simple_tsd.py#L418-L467).

### 5.4 Selection pseudocode

```text
Input:
    frozen teacher p_θ0
    question x
    answer y
    number of score draws sel_mc

Initialize score_sum[i] = 0 and count[i] = 0

Repeat sel_mc times:
    sample a normal diffusion mask rate
    partially mask answer positions to form y_M

    run teacher with question x
    run teacher with blanked question x_empty

    for each masked answer position i:
        score_sum[i] += log p_θ0(y_i | x, y_M)
                        - log p_θ0(y_i | x_empty, y_M)
        count[i] += 1

score[i] = score_sum[i] / max(count[i], 1)
select Â using Otsu, top-k, or fixed tau
optionally remove answer words copied from the question
return Â
```

---

## 6. Stage 2: trajectory-selective distillation

### 6.1 Sample a diffusion state

For each active example, the method samples masked answer positions $\mathcal M$. It then constructs the partially masked answer $y_{\mathcal M}$ and evaluates both the frozen teacher and trainable student on the same state.

The implementation supports two masking modes:

- **`stratified`** — samples answer positions and forces at least one selected forget position into each active draw;
- **`cover`** — samples a diffusion mask budget, splits that budget between selected and non-selected positions, and samples selected positions with probabilities informed by their PCSS scores.

The masking implementations are [`_sample_mask`](../src/trainer/dlm/unlearn/simple_tsd.py#L265-L288) and [`_sample_masks_cover`](../src/trainer/dlm/unlearn/simple_tsd.py#L313-L362).

### 6.2 Split the masked positions

The masked positions are partitioned into

$$
F=\mathcal M\cap\widehat{\mathcal A}
$$

and

$$
R=\mathcal M\setminus\widehat{\mathcal A}.
$$

- $F$ receives the unlearning objective.
- $R$ receives the preservation objective.

An example with $\widehat{\mathcal A}=\varnothing$ contributes no training loss.

### 6.3 Construct the forget target

For every $i\in F$, start from the frozen teacher distribution:

$$
q_{\mathcal M,i}\leftarrow p^0_{\mathcal M,i}.
$$

In the default `minimal` mode, remove the forgotten gold token:

$$
q_{\mathcal M,i}(y_i)\leftarrow 0.
$$

Then renormalize over the surviving vocabulary:

$$
Z_i=\sum_{v\in\mathcal V}q_{\mathcal M,i}(v),
\qquad
q_{\mathcal M,i}(v)
\leftarrow
\frac{q_{\mathcal M,i}(v)}{Z_i}.
$$

The implementation performs the relevant probability calculations in float32 and constructs the normalized target in log space for numerical stability.

With `demotion="belief"`, the method additionally removes the teacher's top `belief_top_k` predictions. This is a stronger variant intended to suppress a local belief set rather than only the exact gold token.

Target construction is implemented in [`_demote_mask` and `_forget_kl`](../src/trainer/dlm/unlearn/simple_tsd.py#L364-L416).

### 6.4 Forget loss

The student is fitted to the edited target using

$$
\mathcal L_f
=
\frac{1}{|F|}
\sum_{i\in F}
D_{\mathrm{KL}}
\left(
q_{\mathcal M,i}
\;\|\;
p^\theta_{\mathcal M,i}
\right).
$$

The direction of this KL is important: the edited teacher target is the first argument, and the student is the second.

### 6.5 Self-distillation loss

At masked positions not selected for forgetting, the student should preserve the original model:

$$
\mathcal L_d
=
\frac{1}{|R|}
\sum_{i\in R}
D_{\mathrm{KL}}
\left(
p^0_{\mathcal M,i}
\;\|\;
p^\theta_{\mathcal M,i}
\right).
$$

This term is the retain-free utility guard. It preserves behavior on non-selected parts of the answer without requiring examples from a separate retain dataset.

### 6.6 Combined loss

When $R\neq\varnothing$, the per-example loss is

$$
\mathcal L_{\mathrm{ex}}
=
\alpha\mathcal L_f
+
(1-\alpha)\mathcal L_d.
$$

When $R=\varnothing$, the method uses

$$
\mathcal L_{\mathrm{ex}}=\mathcal L_f.
$$

The fallback avoids shrinking the forget objective merely because a sampled state contains no self-distillation positions.

The reduction is implemented in [`_combine_draw`](../src/trainer/dlm/unlearn/simple_tsd.py#L500-L580).

### 6.7 Multiple mask draws

The configuration can use `loss_mc > 1` independent mask draws for the same selected set $\widehat{\mathcal A}$. Their losses are averaged to reduce pathwise gradient variance.

During training, the code backpropagates each draw before creating the next student graph. This keeps peak activation memory close to one draw under DeepSpeed ZeRO-3. See [`compute_loss` and `training_step`](../src/trainer/dlm/unlearn/simple_tsd.py#L582-L643).

### 6.8 Training pseudocode

```text
Input:
    frozen teacher p_θ0
    student p_θ initialized from θ0
    forget examples (x, y)
    forget weight α

For each training step:
    sample a minibatch

    for each example:
        obtain selected positions Â using PCSS or a manual span
        if Â is empty:
            skip the example

        sample one or more diffusion mask sets M

        for each mask draw M:
            ensure the draw contains a selected forget position
            form partially masked answer y_M
            run teacher and student on (x, y_M)

            F = M ∩ Â
            R = M \ Â

            for each i in F:
                copy teacher distribution into q
                remove the gold token (and optional belief tokens)
                renormalize q
                accumulate KL(q || student)

            for each i in R:
                accumulate KL(teacher || student)

            average each loss over its active positions
            combine forget and distillation losses using α

    average active examples and mask draws
    update θ

Return θ
```

---

## 7. Why the method works

SimpleTSD combines three constraints:

1. **Selection:** apply forgetting where the prompt reveals answer-specific information.
2. **Controlled replacement:** replace the original target with a normalized teacher distribution that excludes the forgotten token or belief set.
3. **Local preservation:** match the original teacher on nearby, non-selected answer positions.

This makes the method more targeted than whole-answer gradient ascent and allows it to train without a separate retain set.

---

# Part II: Implementation

The implementation is in
[`src/trainer/dlm/unlearn/simple_tsd.py`](../src/trainer/dlm/unlearn/simple_tsd.py),
and its default configuration is in
[`configs/trainer/DiffusionSimpleTSD.yaml`](../configs/trainer/DiffusionSimpleTSD.yaml).

## Special-token exclusion

SimpleTSD's forget branch operates only on lexical tokens. Its forbidden set
$\mathcal S$ contains every tokenizer-declared special ID plus the assistant-turn
terminator derived from the chat template. This explicitly covers EOT/EOS, PAD,
BOS, and MASK, including Dream/LLaDA turn-end IDs that a tokenizer may omit from
`all_special_ids`.

PCSS cannot select a position whose clean answer token is in $\mathcal S$. At a
selected lexical position, the student and edited-teacher distributions are
conditioned on $\mathcal V_{\mathrm{lex}}=\mathcal V\setminus\mathcal S$ before
gold/belief demotion and renormalization. Equivalently, each forget target first
sets

$$
\widetilde q(v)=0\quad(v\in\mathcal S),
\qquad
q_{\mathrm{lex}}(v)=
\frac{\widetilde q(v)}{\sum_{u\notin\mathcal S}\widetilde q(u)}.
$$

The same rule is inherited by the QMask and UTSD variants: QMask conditions its
prompt-masked teacher on lexical support, while UTSD is uniform over
$\mathcal V_{\mathrm{lex}}$, not the full vocabulary. Special-token answer
positions remain eligible for the non-forget self-distillation branch, whose
full-vocabulary teacher target preserves sequence-format behavior.

## 8. Current configuration

The shipped defaults are defined in [`configs/trainer/DiffusionSimpleTSD.yaml`](../configs/trainer/DiffusionSimpleTSD.yaml).

| Argument | Default | Purpose |
|---|---:|---|
| `alpha` | `0.5` | Balance between forgetting and self-distillation. |
| `select_method` | `otsu` | Convert PCSS scores into selected positions. |
| `otsu_relax` | `0.5` | Lower the Otsu threshold to select more positions. |
| `sel_mc` | `8` | Number of partial-mask draws used to estimate PCSS. |
| `loss_mc` | `4` | Number of mask draws used for each training loss. |
| `mask_mode` | `cover` | Strategy used to construct training mask sets. |
| `retain_frac` | `0.5` | Share of a cover-mode mask budget assigned to self-distillation positions. |
| `select_temp` | `1.0` | Temperature for score-weighted forget-position sampling. |
| `topk_frac` | `0.5` | Selected fraction when `select_method=topk`. |
| `drop_question_overlap` | `true` | Remove answer words already exposed by the question. |
| `tau` | `0.0` | Threshold when `select_method=tau`. |
| `demotion` | `minimal` | Remove only the gold token or a larger teacher belief set. |
| `belief_top_k` | `5` | Number of teacher top predictions removed in belief mode. |
| `mask_weight_power` | `0.0` | Optional emphasis on high-mask, early-generation states. |
| `gamma` | `1.0` | Overall SimpleTSD loss scale. |

The TOFU experiment wiring is in [`configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml`](../configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml). It removes the retain dataset entirely and uses only the forget split during training.

---

## 9. Simplified specification versus current implementation

The original simplified specification (not included in this checkout) presents
the core method in a deliberately simplified form:

- it scores tokens using one fully masked answer;
- it uses a fixed threshold $\tau$;
- it samples one mask set per example; and
- it removes only the gold token from the forget target.

The current implementation preserves the same selection/forget/distill structure but adds:

- partial-mask Monte Carlo selection;
- Otsu and top-k selection;
- question-overlap filtering;
- stratified and cover masking;
- score-weighted position sampling;
- minimal and belief demotion;
- multi-draw loss averaging; and
- optional weighting of high-mask diffusion states.

This document describes the **current implementation** while retaining the mathematical core of the original specification.

---

## 10. Related variants

- [`DiffusionSimpleTSDQMask`](../src/trainer/dlm/unlearn/simple_tsd_qmask.py) reuses the selection and training pipeline but uses the promptless teacher distribution as its forget target.
- [`DiffusionUTSD`](../src/trainer/dlm/unlearn/uniform_tsd.py) is a related uniform trajectory-selective variant.
- [`DiffusionSBRUTSD`](../src/trainer/dlm/unlearn/sbru_tsd.py) replaces gold-token demotion with bounded semantic embedding-space repulsion while retaining SimpleTSD's selection and self-distillation pipeline.

---

## 11. Implementation map

| Component | Location |
|---|---|
| Trainer class and objective | [`src/trainer/dlm/unlearn/simple_tsd.py`](../src/trainer/dlm/unlearn/simple_tsd.py) |
| Trainer defaults | [`configs/trainer/DiffusionSimpleTSD.yaml`](../configs/trainer/DiffusionSimpleTSD.yaml) |
| TOFU experiment | [`configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml`](../configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml) |
| Trainer registry | [`src/trainer/__init__.py`](../src/trainer/__init__.py) |
| Shared DLLM trainer | [`src/trainer/dlm/base.py`](../src/trainer/dlm/base.py) |
| Shared DLLM unlearning utilities | [`src/trainer/dlm/unlearn/base.py`](../src/trainer/dlm/unlearn/base.py) |
| Primary launch wrapper | [`scripts/dlm/tofu/simple_tsd.sh`](../scripts/dlm/tofu/simple_tsd.sh) |
