# Offline Monte Carlo Query-Entity Masking

This module automatically chooses which query words to hide during Selective
Unlearning. It is a preprocessing pipeline, not a new unlearning loss:

```text
forget examples + frozen checkpoint
                |
                v
      offline Monte Carlo scoring
                |
                v
       cached query-word masks
                |
                v
 DiffusionSelectiveUnlearning training
```

The repository calls the runtime adapter `DiffusionSDU`. That adapter keeps the
Selective Unlearning objective and replaces its rule-based query mask with the
cached mask. The current code does not connect this pipeline to a
`DiffusionADU` trainer.

The central question is: **which words in the query help the frozen model
recover the memorized answer?** Words whose visibility improves answer recovery
receive higher importance and are more likely to be masked during training.

---

# Part I: Theory

## 1. Goal and output

For one forget example, let $x$ be the query, $y=(y_1,\ldots,y_m)$ the true
answer, and $p_{\theta_0}$ the frozen checkpoint. The preprocessor returns a set
of query token positions

$$
\mathcal I_x^{\mathrm{MC}}(x,y).
$$

These are the positions that will be hidden from the counterfactual teacher.
Unlike the original capitalization-and-digit rule, this mask depends on both
the query and answer and on the behavior of $p_{\theta_0}$. It is computed once
and cached before training.

## 2. Candidate words

Let

$$
W(x)=(w_1,\ldots,w_n)
$$

be the ordered query-word list. Let $P_i$ be the token positions belonging to
word $w_i$.

The implementation approximates words by grouping decoded tokenizer pieces at
whitespace boundaries. It removes punctuation-only pieces, chat-template
syntax, and bare role names such as `user` and `assistant`. A word split across
several tokens receives one importance score, and all its tokens are masked
together.

All query words, including stop words, participate in Monte Carlo scoring.

## 3. Scoring one masked query

Let $A$ be the answer-token positions and let
$S\subseteq\{1,\ldots,n\}$ be a set of query words to hide. The occluded input
$z(S)$ is constructed by:

1. masking every answer position in $A$;
2. masking every query position in $\bigcup_{i\in S}P_i$; and
3. leaving all other query words visible.

The coalition score is the mean log-probability of the true masked answer:

$$
L(S)=\frac{1}{|A|}\sum_{j\in A}
\log p_{\theta_0}^{j}\!\left(y_j\mid z(S)\right).
$$

A larger $L(S)$ means that the visible part of the query still lets the model
recover the answer. The answer is fully masked so that it cannot reveal itself
to the model.

## 4. Estimating each word's importance

For each trial $r=1,\ldots,R$, sample

$$
\rho_r\sim\mathcal U(0,1),
\qquad
\mathbf 1[i\in S_r]\sim\operatorname{Bernoulli}(\rho_r).
$$

The sampled set $S_r$ is scored with $L(S_r)$. For each word, trials are split
according to whether that word was visible or masked:

$$
\widehat a_i=
\operatorname{mean}_{r:\,i\notin S_r}L(S_r)
-
\operatorname{mean}_{r:\,i\in S_r}L(S_r).
$$

A positive $\widehat a_i$ means that answer recovery was better when $w_i$ was
visible. The word therefore carries evidence about the answer. A value near
zero means that the sampled trials found little dependence on the word. The
implementation retains negative values and assigns zero when one comparison
group is empty.

This quantity is an occlusion importance estimate, not an exact Shapley value.
All words share the same trial-specific $\rho_r$, so a word is more likely to be
masked in trials where many other words are also masked. Its score can therefore
include coalition-size and interaction effects.

For example, if hiding `Hsiao Yun-Hwa` sharply lowers the probability of the
true answer while hiding `what` has little effect, the name receives a larger
importance score than the question word.

## 5. Entity calibration and final selection

The current implementation combines the Monte Carlo score with a weak version
of the original entity heuristic. Let $h_i=1$ when $w_i$ looks like a proper
name or numeric identifier, and let

$$
\Delta_a=\max_j\widehat a_j-\min_j\widehat a_j.
$$

With `entity_boost` $=b$, the ranking score is

$$
\widetilde a_i
=
\widehat a_i+(b-1)\Delta_a h_i.
$$

The default $b=1.6$ favors entity-like words without discarding the learned
importance ranking. Setting $b=1$ disables this calibration. Thus the current
mask is model-guided, but not purely model-derived.

For `k_percent_query` $=\kappa$ and $n>0$, the initial selection size is

$$
K=\max\!\left(1,
\operatorname{round}\!\left(\frac{\kappa n}{100}\right)\right).
$$

The preprocessor takes the $K$ largest calibrated scores, then applies two
cleanup steps:

1. **Stop-word removal.** A selected word is removed when all of its normalized
   fragments belong to the query allowlist. Removed words are not replaced,
   so the result may contain fewer than $K$ words.
2. **Name-span expansion.** If the selection touches a consecutive span of
   capitalized, non-stop words, the entire span is selected. This prevents a
   multi-word identity such as `Hsiao Yun-Hwa` from being only partly hidden.

The final number of selected words is therefore not guaranteed to equal $K$.

## 6. Cache and token-mask reconstruction

The preprocessor stores selected word indices rather than tokenizer IDs or
absolute sequence positions. At training time it rebuilds the same word groups
and returns

$$
\mathcal I_x^{\mathrm{MC}}(x,y)
=
\left(\bigcup_{i\in S_{\mathrm{final}}}P_i\right)
\cap\mathcal I_{\mathrm{prompt}}.
$$

The prompt intersection ensures that the reconstructed mask cannot include
answer, padding, or unattended positions. Each cache entry is keyed by the
dataset's stable example index.

## 7. Effect during Selective Unlearning

`DiffusionSDU` substitutes the cached result for the parent's rule-based query
mask:

$$
\mathcal I_x^{\mathrm{rule}}(x)
\quad\longrightarrow\quad
\mathcal I_x^{\mathrm{MC}}(x,y).
$$

The reconstructed mask has two effects:

1. **Counterfactual teacher.** Selected query tokens are replaced with `[MASK]`
   when constructing the entity-counterfactual teacher input.
2. **Copied-answer detection.** Answer words matching selected query content are
   treated as copied subjects and preserved rather than assigned to the forget
   mask.

The second effect is important: changing the query selector can also change
which answer positions are preserved or forgotten. The answer-entity heuristic,
teacher target, and loss equations otherwise remain those of
`DiffusionSelectiveUnlearning`. Offline scoring contributes no gradient.

## 8. Assumptions and limitations

- **Predictive dependence is not identity recognition.** The mask may select a
  name, date, relation, topic word, or any other content that helps recover the
  answer.
- **The cache is checkpoint-specific.** A mask generated from another model or
  tokenizer may not represent the dependencies used during training.
- **Offline and training corruption differ.** Precomputation fully masks the
  answer, whereas training uses ordinary diffusion noise plus deterministic
  answer-entity masking.
- **The estimator is noisy and coalition-dependent.** More trials reduce
  variance but do not remove bias caused by correlated coalition sizes.
- **Word indices require stable preprocessing.** Changes to the prompt, chat
  template, tokenizer, truncation, or grouping can invalidate cached indices.
- **Cleanup changes the requested fraction.** Stop-word removal can shrink the
  mask and name expansion can enlarge it.
- **Runtime metadata validation is limited.** The current loader does not prove
  that a cache matches the active checkpoint and preprocessing configuration.

For a valid run, generate the cache from the same base checkpoint, dataset
split, tokenizer, chat template, and sequence-length settings used by training.

---

# Part II: Implementation

## 9. Offline preprocessing pipeline

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
8. Adds the configured prior to entity-like words.
9. Selects the top `k_percent_query` percent by calibrated rank.
10. Drops selected pure stop/function words.
11. Expands any partially selected multi-word proper-name span.
12. Writes the sorted final word indices into the JSON cache.

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

5. ranking  <- ADD_ENTITY_CALIBRATION(importance, entity_boost)
6. selected <- TOP_PERCENT_BY_RANK(ranking, k_percent_query)
7. selected <- DROP_SELECTED_STOPWORDS(selected)
8. selected <- EXPAND_TOUCHED_CAPITALIZED_NAME_SPANS(selected)
9. cache[dataset_index].query <- SORT(selected)
```

### 9.1 Batched model scoring

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

### 9.2 Randomness and repeatability

The command-line `seed` initializes both Python's `random` module and PyTorch.
Coalition probabilities and memberships are sampled through the Python random
generator passed to `mc_query_leak_importance`. With the same examples,
checkpoint, tokenizer, arguments, and deterministic model execution, this
reproduces the same sampled masks and rankings.

## 10. Word-group construction

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

## 11. Cache format and identity contract

The generated JSON has this structure:

```json
{
  "meta": {
    "model_dir": "saves/finetune/tofu_LLaDA-1.5_retain100_150epochs",
    "split": "forget10",
    "k_percent_query": 20,
    "entity_boost": 1.6,
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

## 12. Runtime `query_entity_mask`

[`DiffusionSDU.query_entity_mask`](../src/trainer/dlm/unlearn/sdu.py) converts
the cached word ranks into a boolean tensor for each training batch.

Before the method is called, `DiffusionSDU.sample_selective_state` requires an
`index` field and stores the batch's dataset indices in `_current_index`. The
parent implementation then performs normal EOS extension and model-specific
input preprocessing before invoking `query_entity_mask`.

For each batch row, `query_entity_mask`:

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

## 13. Relationship to the heuristic algorithm

The parent `DiffusionSelectiveUnlearning.query_entity_mask` selects words from
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

## 14. Configuration

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
| `--entity-boost` | `1.6` | Additive ranking prior for entity-like words; `1.0` disables it. |
| `--seed` | `0` | Python and PyTorch random seed. |
| `--max-length` | `512` | Tokenization/truncation length used while building word ranks. |
| `--out` | required | Destination JSON cache. |

The SDU experiment configuration selects
`DataCollatorForSupervisedDatasetwithIndex`, which is required to pass stable
dataset indices from `QADataset` to the trainer.

## 15. Running precomputation and training

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

## 16. Failure modes and safeguards

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

## 17. Source map

| Responsibility | File |
|---|---|
| SDU runtime cache loading and `query_entity_mask` | [`src/trainer/dlm/unlearn/sdu.py`](../src/trainer/dlm/unlearn/sdu.py) |
| Word grouping, Monte Carlo scoring, ranking, filtering, and span expansion | [`src/trainer/dlm/unlearn/query_importance.py`](../src/trainer/dlm/unlearn/query_importance.py) |
| Offline cache generation | [`scripts/dlm/sdu_precompute.py`](../scripts/dlm/sdu_precompute.py) |
| Parent query heuristic and unchanged answer-side logic | [`src/trainer/dlm/unlearn/selective_unlearning.py`](../src/trainer/dlm/unlearn/selective_unlearning.py) |
| Dataset index creation | [`src/data/qa.py`](../src/data/qa.py) and [`src/data/utils.py`](../src/data/utils.py) |
| Index-preserving batch collation | [`src/data/collators.py`](../src/data/collators.py) |
| Trainer defaults | [`configs/trainer/DiffusionSDU.yaml`](../configs/trainer/DiffusionSDU.yaml) |
| TOFU SDU experiment | [`configs/experiment/unlearn/tofu/dlm_sdu.yaml`](../configs/experiment/unlearn/tofu/dlm_sdu.yaml) |
| Single-run launcher | [`scripts/dlm/tofu/run_sdu.sh`](../scripts/dlm/tofu/run_sdu.sh) |
| Multi-model/split launcher | [`scripts/dlm/tofu/run_sdu_all.sh`](../scripts/dlm/tofu/run_sdu_all.sh) |
