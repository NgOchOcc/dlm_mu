# Offline Monte Carlo Query-Entity Masking for SDU

`DiffusionSDU` replaces the query-side surface heuristic used by
`DiffusionSelectiveUnlearning` with an offline, model-based estimate of which
query words help the frozen model recover the memorized answer. The selected
words are cached once and converted back into token positions during training
by [`_query_entity_mask`](../src/trainer/dlm/unlearn/sdu.py).

The name *query entity mask* is inherited from Selective Unlearning. In SDU,
however, selection is not a named-entity-recognition rule. A word is selected
because masking it reduces the base model's ability to reconstruct the true
answer. The resulting set can therefore contain a name, identifier, relation
word, or other answer-predictive query content, subject to the post-selection
filters described below.

Only the query-side mask is changed. SDU inherits the answer-copy mask,
answer-entity mask, entity-counterfactual teacher, selective target, and loss
from `DiffusionSelectiveUnlearning` unchanged.

This document separates the algorithm from its concrete implementation:

- [Part I: Theory](#part-i-theory) defines Monte Carlo query-leak importance,
  rank-based selection, stop-word filtering, name-span expansion, and the final
  token mask.
- [Part II: Implementation](#part-ii-implementation) describes offline cache
  generation, runtime reconstruction, configuration, failure modes, and usage.

---

# Part I: Theory

## 1. Setup and objective

Let one forget example be a tokenized query-answer pair

$$
(x,y),
$$

where $x$ is the query and $y=(y_1,\ldots,y_m)$ is the ground-truth answer.
Let $p_{\theta_0}$ be the frozen base model that will later act as the teacher
for Selective Unlearning.

The goal of the query-mask algorithm is to identify query words whose
visibility materially helps $p_{\theta_0}$ recover $y$. If hiding a query word
lowers the model's log-probability of the true answer, that word is treated as
leaking information about the memorized answer and becomes a candidate for the
entity-counterfactual mask.

The output is a set of query token positions

$$
\mathcal I_x^{\mathrm{SDU}}(x,y),
$$

which replaces the heuristic query mask $\mathcal I_x(x)$ in
`DiffusionSelectiveUnlearning`.

Unlike the original heuristic, the SDU mask depends on all of the following:

- the query $x$;
- the ground-truth answer $y$;
- the frozen checkpoint $p_{\theta_0}$;
- the tokenizer and chat template;
- the Monte Carlo seed and number of trials;
- the selected top-$k$ percentage.

It is therefore an example-specific and checkpoint-specific mask.

## 2. Query words and token groups

The algorithm operates on words rather than directly on tokenizer pieces.
Let

$$
W(x)=(w_1,\ldots,w_n)
$$

be the ordered query-word list, and let $P_i$ be the absolute token positions
belonging to word $w_i$.

Words are approximated by grouping adjacent decoded token surfaces at
whitespace boundaries. Tokens containing only punctuation or chat-template
control syntax are removed from each group. Bare role names such as `user`,
`assistant`, and `system` are also removed when they occur at the beginning or
end of a group.

Consequently:

$$
P_i \subseteq \mathcal I_{\mathrm{prompt}},
$$

where $\mathcal I_{\mathrm{prompt}}$ contains query-side positions rather than
supervised answer positions. A split word may contain several tokenizer pieces,
but it receives one importance score and is masked as one unit.

Stop and function words remain in $W(x)$ during importance estimation. They are
not removed from the candidate population before Monte Carlo sampling.

## 3. Occluded query-answer states

Let $A$ be the set of answer token positions and let $y_j$ denote the true
token at answer position $j\in A$. Before scoring query words, every answer
position is replaced with the model's mask token.

For a subset of word indices $S\subseteq\{1,\ldots,n\}$, define the occluded
input $z(S)$ by

$$
z(S)_j=
\begin{cases}
\texttt{[MASK]}, & j\in A,\\
\texttt{[MASK]}, & j\in P_i\text{ for some }i\in S,\\
x_j, & \text{otherwise}.
\end{cases}
$$

Thus the answer is always fully masked, while $S$ controls which query words
are hidden. The attention mask is unchanged.

This differs from SDU training, where the answer follows the normal selective
unlearning state construction and is only masked at the sampled diffusion
positions plus the answer-side hiding mask. Offline scoring deliberately uses a
fixed, fully masked answer so the score measures recovery from query evidence
rather than from visible answer tokens.

## 4. Answer-recovery score

For one occlusion set $S$, define the mean true-answer log-probability

$$
L(S)
=
\frac{1}{|A|}
\sum_{j\in A}
\log p_{\theta_0}^{j}
\left(
y_j\mid z(S)
\right).
$$

All true answer tokens are scored in one model forward pass. The score is the
mean of their individual log-probabilities, not sequence-level autoregressive
likelihood: masked diffusion predicts the answer positions in parallel.

A larger $L(S)$ means the model assigns higher probability to the memorized
answer under that query-word occlusion.

## 5. Monte Carlo coalition sampling

For each trial $r\in\{1,\ldots,R\}$, SDU samples a masking probability

$$
\rho_r\sim\mathcal U(0,1).
$$

Conditioned on $\rho_r$, every query word is independently included in the
masked coalition $S_r$:

$$
\mathbf 1[i\in S_r]
\sim
\operatorname{Bernoulli}(\rho_r).
$$

The trial is then scored with $L(S_r)$.

Sampling a new $\rho_r$ for every trial exposes the model to coalitions ranging
from nearly no query masking to nearly complete query masking. It does not use
one fixed masking rate such as $0.5$.

Because $\rho_r$ is shared by all words within a trial, word-mask indicators are
conditionally independent given $\rho_r$ but correlated after marginalizing
over $\rho_r$. The estimator is an occlusion-based approximation inspired by
coalition attribution; it is not an exact enumeration of Shapley values.

## 6. Query-leak importance

For word $w_i$, partition the trials into:

$$
\mathcal R_i^{\mathrm{visible}}
=
\{r:i\notin S_r\},
$$

and

$$
\mathcal R_i^{\mathrm{masked}}
=
\{r:i\in S_r\}.
$$

The Monte Carlo importance is

$$
\widehat a_i
=
\frac{1}{|\mathcal R_i^{\mathrm{visible}}|}
\sum_{r\in\mathcal R_i^{\mathrm{visible}}}L(S_r)
-
\frac{1}{|\mathcal R_i^{\mathrm{masked}}|}
\sum_{r\in\mathcal R_i^{\mathrm{masked}}}L(S_r).
$$

Interpretation:

- $\widehat a_i>0$: the true answer is easier to recover when $w_i$ is
  visible, so masking that word removes answer-relevant evidence;
- $\widehat a_i\approx0$: visibility of $w_i$ has little estimated effect;
- $\widehat a_i<0$: the sampled trials happened to score better when $w_i$ was
  masked.

The implementation does not clamp negative importance values. They remain in
the ranking and are selected only if they still fall inside the requested
top-$k$ fraction.

If a finite Monte Carlo sample never observes both states for a word, its
importance is set to zero. This can occur with very few trials or extreme
sampled masking probabilities.

## 7. Fixed-fraction rank selection

Let $\kappa$ be `k_percent_query`. For $n>0$ query words, the implementation
computes

$$
K
=
\max\left(
1,
\operatorname{round}\left(\frac{\kappa}{100}n\right)
\right).
$$

It sorts words by decreasing $\widehat a_i$ and initially selects the first
$K$ indices:

$$
S_{\mathrm{rank}}
=
\operatorname{TopK}
\left(
\widehat a_1,\ldots,\widehat a_n;K
\right).
$$

At least one word is selected when the query-word list is nonempty. Python's
stable sorting preserves the earlier word index when importance values tie.
The code assumes a sensible percentage range; it does not explicitly validate
or clamp `k_percent_query`.

## 8. Post-ranking stop-word filter

Let $\mathcal K$ be the query allowlist inherited from
`DiffusionSelectiveUnlearning._QUERY_KEEP`. A selected word is considered a
stop/function word if all of its normalized alphanumeric fragments belong to
$\mathcal K$.

The filtered set is

$$
S_{\mathrm{content}}
=
\left\{
i\in S_{\mathrm{rank}}:
\operatorname{fragments}(w_i)\not\subseteq\mathcal K
\right\}.
$$

This filter is applied **after** top-$K$ ranking. Stop words participate in all
Monte Carlo trials and count toward $n$ and $K$, but a selected pure stop word
is dropped rather than replaced by the next-ranked content word. Therefore the
final mask may contain fewer than $K$ words or even be empty.

## 9. Multi-word proper-name expansion

Monte Carlo scoring treats each whitespace-delimited word independently. This
can select only part of a multi-word name, leaving enough visible text to reveal
the identity. SDU therefore expands partially selected name spans.

A word is considered a name part when:

1. it contains at least one normalized alphanumeric fragment;
2. its alphabetic surface has at least two letters;
3. its first alphabetic character is uppercase; and
4. not all of its fragments belong to the query allowlist.

Adjacent name parts form a maximal span. If
$S_{\mathrm{content}}$ intersects that span, every word in the span is added:

$$
S_{\mathrm{final}}
=
S_{\mathrm{content}}
\cup
\bigcup_{N:\,N\cap S_{\mathrm{content}}\neq\varnothing}N.
$$

For example, if `Hsiao Yun-Hwa` forms one three-word name span and only
`Yun-Hwa` enters the top-$K$ set, expansion adds the other name words as well.

Numeric words do not receive special span expansion. They can still be selected
by their Monte Carlo score and survive the stop-word filter.

Name expansion can make the final selection larger than $K$.

## 10. Final token-level query mask

The offline cache stores word ranks, not token IDs or absolute token positions.
For a cached example, the final token mask is reconstructed as

$$
\mathcal I_x^{\mathrm{SDU}}(x,y)
=
\bigcup_{i\in S_{\mathrm{final}}}P_i.
$$

At training time the result is intersected with the current prompt mask:

$$
\mathcal I_x^{\mathrm{runtime}}
=
\mathcal I_x^{\mathrm{SDU}}
\cap
\mathcal I_{\mathrm{prompt}}.
$$

This final intersection prevents cached word indices from marking answer,
padding, or unattended positions.

## 11. Effect inside Selective Unlearning

SDU passes the reconstructed mask into the unchanged parent algorithm. It has
two downstream roles:

1. **Counterfactual query construction.** Positions in
   $\mathcal I_x^{\mathrm{runtime}}$ are replaced by `[MASK]` for the frozen
   entity-counterfactual teacher.
2. **Copied-subject detection.** The parent trainer decodes the selected query
   tokens and protects matching answer words as copied query content rather than
   treating them as forget-fact positions.

All answer-side masks still use the surface heuristic from
`DiffusionSelectiveUnlearning`. In particular, SDU does not read an answer mask
from the JSON cache.

The remainder of the objective is unchanged: the same corrupted answer state
is used for the student, full-query teacher, and entity-masked-query teacher;
forget positions use the selective target; and other effective positions use
the full-query preservation target.

## 12. Assumptions and limitations

The query-mask algorithm relies on several assumptions:

1. **Answer recovery indicates leakage.** A query word is considered important
   when hiding it reduces true-answer probability. This measures predictive
   dependence, not causal identity or factual correctness.
2. **The base checkpoint is the correct reference.** Importance is specific to
   the checkpoint used during precomputation. Reusing a cache with a different
   model can select words that are no longer informative for that model.
3. **Fully masked answers are representative.** Precomputation measures recall
   when every answer token is hidden, while training uses a sampled diffusion
   mask plus deterministic answer-side hiding.
4. **Whitespace grouping is stable.** The cache stores word indices. A change in
   tokenizer, chat template, truncation length, prompt text, or preprocessing
   can change the word list and invalidate those indices.
5. **Monte Carlo estimates are noisy.** More trials reduce sampling variance
   but increase preprocessing cost linearly.
6. **Marginal score differences include interactions.** Visible and masked
   averages are taken over different sampled coalitions; correlated query words
   can share, dilute, or redistribute importance.
7. **The final percentage is not exact.** Post-ranking stop-word removal can
   shrink the mask, while name-span expansion can enlarge it.
8. **Capitalization expansion is heuristic.** It protects multi-word names but
   can merge adjacent capitalized content words or miss lowercase aliases.
9. **Cache metadata is advisory.** The runtime loader reads `entries` but does
   not verify the cached model, split, seed, maximum length, or word count.

For reliable use, generate the cache with the same base checkpoint, tokenizer,
TOFU split, chat template, and `max_length` used by training.

---

# Part II: Implementation

## 13. Offline preprocessing pipeline

The cache generator is
[`scripts/dlm/sdu_precompute.py`](../scripts/dlm/sdu_precompute.py). For each
example it performs the following steps:

1. Loads and tokenizes the TOFU query-answer example.
2. Defines answer positions as `labels != -100` and query positions as
   `labels == -100`.
3. Builds the ordered query-word list with `build_word_groups`.
4. Fully masks every answer token.
5. Samples `num_mc_trials` random query-word coalitions.
6. Evaluates all trial rows as one model batch.
7. Computes each word's visible-minus-masked mean answer score.
8. Selects the top `k_percent_query` percent by rank.
9. Drops selected pure stop/function words.
10. Expands any partially selected multi-word proper-name span.
11. Writes the sorted final word indices into the JSON cache.

The corresponding execution flow is:

```text
Input: tokenized (query, answer), frozen model p_theta0

1. query_words   <- BUILD_WORD_GROUPS(query token positions)
2. answer_masked <- MASK_ALL(answer positions)

3. for trial r = 1 ... R:
       rho_r <- Uniform(0, 1)
       S_r   <- independently mask each query word with probability rho_r
       L_r   <- mean true-answer log probability under S_r

4. for each query word i:
       importance_i <- mean(L_r | i visible) - mean(L_r | i masked)
       if either sample group is empty: importance_i <- 0

5. selected <- TOP_PERCENT_BY_RANK(importance, k_percent_query)
6. selected <- DROP_SELECTED_STOPWORDS(selected)
7. selected <- EXPAND_TOUCHED_CAPITALIZED_NAME_SPANS(selected)
8. cache[dataset_index].query <- SORT(selected)
```

### 13.1 Batched model scoring

`mc_query_leak_importance` constructs one input row per Monte Carlo trial and
stacks all rows into a single tensor. `_gather_logp` then:

1. runs one model forward over the trial batch;
2. truncates logits to `len(tokenizer)`;
3. converts logits to float32;
4. applies `log_softmax` over the vocabulary;
5. gathers the true token at every answer position; and
6. averages those log-probabilities across answer positions.

The memory cost therefore grows with `num_mc_trials × sequence_length`, while
the number of model forwards per example remains one.

### 13.2 Randomness and repeatability

The command-line `seed` initializes both Python's `random` module and PyTorch.
Coalition probabilities and memberships are sampled through the Python random
generator passed to `mc_query_leak_importance`. With the same examples,
checkpoint, tokenizer, arguments, and deterministic model execution, this
reproduces the same sampled masks and rankings.

## 14. Word-group construction

[`build_word_groups`](../src/trainer/dlm/unlearn/query_importance.py) is shared
between precomputation and runtime reconstruction. It:

1. decodes each candidate token independently;
2. groups decoded surfaces when a new surface begins with whitespace;
3. removes token pieces without alphanumeric content after stripping special
   markers;
4. removes leading or trailing token runs spelling `user`, `assistant`, or
   `system`; and
5. stores both the reconstructed word text and its absolute token positions.

The role-name removal is necessary because some chat templates place bare role
tokens next to neighboring text without a whitespace boundary. Such tokens
describe conversation structure and must not become query-mask candidates.

The grouping is tokenizer-surface based, not a linguistic tokenizer. Languages
without whitespace boundaries and tokenizers with unusual decoded spacing may
produce groups that do not correspond to semantic words.

## 15. Cache format and identity contract

The generated JSON has this structure:

```json
{
  "meta": {
    "model_dir": "saves/finetune/tofu_LLaDA-1.5_retain100_150epochs",
    "split": "forget10",
    "k_percent_query": 20,
    "num_mc_trials": 32,
    "max_length": 512,
    "seed": 0
  },
  "entries": {
    "0": {
      "query": [3, 4],
      "n_query_words": 12
    }
  }
}
```

`entries` is keyed by the dataset's stable per-example `index`, not by batch
position or dataloader order. `QADataset` adds this index before tokenization,
and the index-emitting collator preserves it in each training batch.

The `query` array contains zero-based indices into the ordered word list from
`build_word_groups`. It does not contain tokenizer IDs or absolute sequence
positions. `n_query_words` and `meta` are recorded for inspection, but the
current runtime loader does not validate them.

`--start-index` and `--num-samples` control which dataset rows are processed.
Even for a partial range, the cache key remains the row's dataset `index`.

## 16. Runtime `_query_entity_mask`

[`DiffusionSDU._query_entity_mask`](../src/trainer/dlm/unlearn/sdu.py) converts
the cached word ranks into a boolean tensor for each training batch.

Before the method is called, `DiffusionSDU._sample_selective_state` requires an
`index` field and stores the batch's dataset indices in `_current_index`. The
parent implementation then performs normal EOS extension and model-specific
input preprocessing before invoking `_query_entity_mask`.

For each batch row, `_query_entity_mask`:

1. constructs `prompt_mask` from non-answer, non-padding, attended positions;
2. retrieves the selected word-index set using that example's dataset index;
3. rebuilds query words over the current prompt positions with
   `build_word_groups`;
4. maps every selected word index to all of its token positions;
5. silently skips cached word indices beyond the reconstructed word-list
   length; and
6. intersects the result with `prompt_mask` before returning it.

In tensor form, the returned mask has the same shape as `input_ids`, uses a
boolean value per token position, and is always false outside the prompt.

An empty cached selection produces an all-false query mask. A missing cache
entry raises `KeyError`. A batch without `index` raises `ValueError` before the
parent selective-state construction begins.

## 17. Relationship to the heuristic algorithm

The parent `DiffusionSelectiveUnlearning._query_entity_mask` selects words from
surface form alone:

- any word containing a digit is selected;
- otherwise the alphabetic surface must start uppercase, contain at least two
  letters, and not consist entirely of allowlisted fragments.

SDU replaces that decision rule with the cached Monte Carlo ranking. Both
methods ultimately return token-level masks over grouped query words, and both
use the same query allowlist for excluding pure stop/function words.

| Property | SelectiveUnlearning heuristic | SDU Monte Carlo mask |
|---|---|---|
| Selection signal | Capitalization, digits, allowlist | Change in true-answer log-probability |
| Depends on answer | No | Yes |
| Depends on checkpoint | No | Yes |
| Computed | During every training batch | Once offline, reconstructed online |
| Cached unit | None | Word-list index |
| Multi-word name handling | Each qualifying word selected directly | Touched capitalized spans expanded |
| Answer-side masking | Surface heuristic | Same inherited surface heuristic |

## 18. Configuration

The trainer configuration is
[`configs/trainer/DiffusionSDU.yaml`](../configs/trainer/DiffusionSDU.yaml). It
inherits `beta`, `preserve_weight`, and the remaining selective-unlearning
arguments, and adds:

| Argument | Required | Purpose |
|---|---:|---|
| `mask_cache_path` | Yes | JSON cache generated by `sdu_precompute.py`. |

The following parameters belong to cache generation rather than the trainer:

| Precompute argument | Default | Purpose |
|---|---:|---|
| `--model-dir` | required | Frozen checkpoint used to estimate answer leakage. |
| `--split` | `forget10` | TOFU forget split to cache. |
| `--start-index` | `0` | First dataset row to process. |
| `--num-samples` | whole split | Number of rows to process. |
| `--num-mc-trials` | `32` | Random query-mask coalitions per example. |
| `--k-percent-query` | `20` | Initial top-ranked query-word percentage. |
| `--seed` | `0` | Python and PyTorch random seed. |
| `--max-length` | `512` | Tokenization/truncation length used while building word ranks. |
| `--out` | required | Destination JSON cache. |

The SDU experiment configuration selects
`DataCollatorForSupervisedDatasetwithIndex`, which is required to pass stable
dataset indices from `QADataset` to the trainer.

## 19. Running precomputation and training

Generate a cache using the same base checkpoint and split that training will
use. For example, for LLaDA-1.5 and TOFU `forget10`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/dlm/sdu_precompute.py \
  --model-dir saves/finetune/tofu_LLaDA-1.5_retain100_150epochs \
  --split forget10 \
  --num-mc-trials 32 \
  --k-percent-query 20 \
  --max-length 512 \
  --out saves/sdu_cache/llada1.5_forget10.json
```

Then launch SDU with that cache:

```bash
MODEL=llada1.5 \
GPUS=0,1,2,3 \
SPLIT=forget10 \
MASK_CACHE=saves/sdu_cache/llada1.5_forget10.json \
bash scripts/dlm/tofu/run_sdu.sh
```

For a sweep across models and splits, use
[`scripts/dlm/tofu/run_sdu_all.sh`](../scripts/dlm/tofu/run_sdu_all.sh). Its
defaults currently use 32 Monte Carlo trials, a 30-percent query selection, and
`PRECOMPUTE_MAX_LENGTH=256`, which differ from the standalone precompute
script's defaults. Cache metadata should therefore be checked when comparing
runs generated by the two entry points.

## 20. Failure modes and safeguards

- If `mask_cache_path` is absent, Hydra cannot instantiate `DiffusionSDU`
  because the trainer config marks it as required.
- If the training collator omits `index`, the trainer raises `ValueError`.
- If the cache has no entry for a batch example's dataset index, the trainer
  raises `KeyError` and asks for the split/index range to be recomputed.
- If a cached word index is too large for the reconstructed word list, it is
  silently skipped rather than raising an error.
- If an entry's `query` list is empty, no query positions are masked for that
  example.
- The loader expects a top-level `entries` object and each entry to contain a
  `query` array. Malformed caches fail during loading or access.
- `num_mc_trials` should be positive. The CLI currently does not validate it;
  zero trials would fail when the trial batch is stacked.
- The code does not validate that `k_percent_query` lies in `[0,100]`. Use a
  sensible positive percentage.

## 21. Source map

| Responsibility | File |
|---|---|
| SDU runtime cache loading and `_query_entity_mask` | [`src/trainer/dlm/unlearn/sdu.py`](../src/trainer/dlm/unlearn/sdu.py) |
| Word grouping, Monte Carlo scoring, ranking, filtering, and span expansion | [`src/trainer/dlm/unlearn/query_importance.py`](../src/trainer/dlm/unlearn/query_importance.py) |
| Offline cache generation | [`scripts/dlm/sdu_precompute.py`](../scripts/dlm/sdu_precompute.py) |
| Parent query heuristic and unchanged answer-side logic | [`src/trainer/dlm/unlearn/selective_unlearning.py`](../src/trainer/dlm/unlearn/selective_unlearning.py) |
| Dataset index creation | [`src/data/qa.py`](../src/data/qa.py) and [`src/data/utils.py`](../src/data/utils.py) |
| Index-preserving batch collation | [`src/data/collators.py`](../src/data/collators.py) |
| Trainer defaults | [`configs/trainer/DiffusionSDU.yaml`](../configs/trainer/DiffusionSDU.yaml) |
| TOFU SDU experiment | [`configs/experiment/unlearn/tofu/dlm_sdu.yaml`](../configs/experiment/unlearn/tofu/dlm_sdu.yaml) |
| Single-run launcher | [`scripts/dlm/tofu/run_sdu.sh`](../scripts/dlm/tofu/run_sdu.sh) |
| Multi-model/split launcher | [`scripts/dlm/tofu/run_sdu_all.sh`](../scripts/dlm/tofu/run_sdu_all.sh) |
