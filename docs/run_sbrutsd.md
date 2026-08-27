# SBRU-TSD TOFU unlearning workflow

This document explains what
[`scripts/dlm/tofu/run_sbrutsd.sh`](../scripts/dlm/tofu/run_sbrutsd.sh)
does, the unlearning problem it addresses, and the SBRU-TSD method that it
launches.

> **Scope:** `run_sbrutsd.sh` is a launcher, not the implementation of the
> learning algorithm. It selects defaults and Hydra overrides, starts the common
> TOFU unlearning/evaluation pipeline, redirects its output to a log, and exits.
> The method itself is implemented by
> [`DiffusionSBRUTSD`](../src/trainer/dlm/unlearn/sbru_tsd.py), which subclasses
> [`DiffusionSimpleTSD`](../src/trainer/dlm/unlearn/simple_tsd.py).

## 1. Problem being solved

### 1.1 Machine unlearning on TOFU

The run starts from a masked discrete-diffusion language model (LLaDA or Dream)
that was fine-tuned on the complete TOFU training set. Given a TOFU forget
split, the goal is to update the model so that it no longer reproduces the
question-answer associations in that split while retaining useful language
behavior and knowledge outside that split.

This is different from training from scratch without the forgotten examples:
the launcher starts from an existing `retain100` fine-tuned checkpoint and tries
to remove selected information from it.

### 1.2 Why whole-answer suppression is undesirable

A forget answer mixes information that should be removed with ordinary syntax,
function words, and reusable language. For example, in
“the author was born in London in 1984,” only part of the answer may identify
the fact being forgotten. Applying an unlearning loss to every answer token can
damage fluency and unrelated behavior.

Masked-diffusion models add another requirement. They generate an answer by
denoising many partially masked states, so changing the model at only one fixed
masking pattern does not ensure that the association is removed throughout the
denoising trajectory.

### 1.3 Why combine SBRU and SimpleTSD

The repository contains two relevant ideas:

- **SBRU (Semantic-Ball Repulsion Unlearning)** moves the student's soft
  predicted embedding away from the forgotten semantic region. Its bounded RBF
  objective avoids the unbounded behavior of gradient ascent and is intended to
  discourage probability mass from simply moving to nearby synonyms. Plain
  retain-free SBRU, however, relied on a promptless anchor as its only utility
  guard. A recorded SBRU sweep found retain ROUGE collapsing from approximately
  `0.96` to `0.00–0.03` when the anchor became ineffective at the same kernel
  scale as the repulsion.
- **SimpleTSD (Simple Trajectory-Selective Distillation)** identifies
  prompt-dependent answer positions and self-distills the frozen original model
  at the remaining masked answer positions. This supplies a local utility guard
  using the forget examples themselves, without loading a separate retain
  corpus.

**SBRU-TSD combines them.** It retains SimpleTSD's token selection,
trajectory masking, and self-distillation, but replaces SimpleTSD's
“demote the gold token” forget target with SBRU's bounded embedding-space
repulsion and optional promptless anchor. The intended result is targeted
semantic forgetting with a stronger retain-free preservation signal than
anchor-only SBRU. The method supports anchor attraction, but the current
`run_sbrutsd.sh` sets `GAMMA_ANC=0.0` by default, so its default run uses the
repulsion and TSD self-distillation terms without anchor attraction.

## 2. Method

### 2.1 Notation

| Symbol | Meaning |
|---|---|
| \(x\) | Forget question or prompt. |
| \(x_\varnothing\) | Promptless input obtained by masking the attended non-answer prompt tokens. |
| \(y=(y_1,\ldots,y_L)\) | Gold answer tokens. |
| \(p_{\theta_0}\) | Frozen original model (teacher). |
| \(p_\theta\) | Trainable student initialized from the original model. |
| \(\widehat{\mathcal A}\) | Answer positions selected for forgetting. |
| \(\mathcal M\) | Answer positions masked in one sampled diffusion state. |
| \(F=\mathcal M\cap\widehat{\mathcal A}\) | Masked positions that receive the SBRU forget objective. |
| \(R=\mathcal M\setminus\widehat{\mathcal A}\) | Other masked answer positions that receive self-distillation. |
| \(E\in\mathbb R^{|\mathcal V|\times d}\) | Detached output-embedding/unembedding matrix. |
| \(\widehat e_i\) | Student's probability-weighted embedding at position \(i\). |
| \(c_i\) | Forget-region center. |
| \(a_i\) | Promptless frozen-teacher anchor. |

### 2.2 Stage 1: select prompt-dependent answer positions

For each answer position, Prompt-Conditional Score Selection (PCSS) compares
how likely the frozen teacher considers the gold token with the question present
and with the question blanked:

\[
s_i = \mathbb E_{\mathcal M:i\in\mathcal M}
\left[
\log p_{\theta_0}(y_i\mid x,y_{\mathcal M})-
\log p_{\theta_0}(y_i\mid x_\varnothing,y_{\mathcal M})
\right].
\]

The implementation estimates this expectation with `sel_mc=8` independently
sampled partial-mask states. Each draw performs two teacher forwards: one with
the question and one with the prompt blanked. Partial masking is important: a
fully masked answer can make the promptless distribution almost uniform and
cause nearly the entire answer to look prompt-dependent.

The default selector applies a per-example Otsu threshold and then lowers that
threshold toward the minimum score by `otsu_relax=0.5`. It also removes answer
words whose content fragments merely repeat words already visible in the
question (`drop_question_overlap=true`). The result is
\(\widehat{\mathcal A}\), the positions eligible for forgetting.

Alternative selection rules are available through Hydra overrides:

- `select_method=topk` selects `topk_frac` of the answer positions;
- `select_method=tau` selects positions with \(s_i\ge\tau\); and
- a data batch may provide an explicit `answer_mask`, bypassing PCSS.

Implementation:
[`_select_forget_positions`](../src/trainer/dlm/unlearn/simple_tsd.py#L153-L185),
[`_threshold_scores`](../src/trainer/dlm/unlearn/simple_tsd.py#L207-L223), and
[`_question_overlap_mask`](../src/trainer/dlm/unlearn/simple_tsd.py#L225-L249).

### 2.3 Stage 2: sample denoising states

The default `mask_mode=cover` creates `loss_mc=4` masks per training batch. For
each example and draw it:

1. samples a diffusion time \(t\) and derives the ordinary MDLM answer mask rate
   \(\rho(t)=1-\operatorname{scheduler}(t)\);
2. computes a mask budget from \(\rho(t)\) and the answer length;
3. assigns `retain_frac=0.5` of that budget to positions outside
   \(\widehat{\mathcal A}\) and the remaining share to selected forget
   positions;
4. guarantees at least one selected forget position when the selected set is
   nonempty; and
5. samples forget positions without replacement, weighted by standardized PCSS
   scores and `select_temp`, while sampling self-distillation positions
   uniformly.

This exposes the objective to multiple points on the normal denoising
trajectory. It gives high-PCSS positions more frequent coverage, but it does
not guarantee that every selected position is masked in a batch.

Implementation:
[`_sample_masks_cover`](../src/trainer/dlm/unlearn/simple_tsd.py#L299-L348).
The alternative `mask_mode=stratified` samples uniformly from answer positions
and forces at least one selected position into a draw.

### 2.4 SBRU forget objective at selected positions

At every position, the student distribution is converted into a continuous
soft predicted embedding:

\[
\widehat e_i=\sum_{v\in\mathcal V}p_\theta(v\mid x,y_{\mathcal M})E_v.
\]

The unembedding matrix \(E\) is detached, so the model cannot reduce the loss by
moving the embedding geometry itself; gradients flow through the student's
probability distribution. Under DeepSpeed ZeRO-3, the sharded matrix is
gathered once per batch and reused across all mask draws.

The forget center is selected by `centre_mode`:

- **`gold` (default):** \(c_i=E_{y_i}\). If `ball_k>0` is set by a raw Hydra
  override, the center is instead the mean of the gold embedding's `ball_k`
  cosine-nearest vocabulary embeddings.
- **`ref_soft`:** \(c_i\) is the frozen teacher's soft embedding on the same
  noised, question-conditioned state.

The promptless anchor is the frozen teacher's soft embedding when the question
portion of that noised state is replaced by mask tokens:

\[
a_i=\sum_{v\in\mathcal V}
 p_{\theta_0}(v\mid x_\varnothing,y_{\mathcal M})E_v.
\]

For each selected masked position \(i\in F\), SBRU-TSD minimizes

\[
\ell_{f,i}=k_{\mathrm{rep}}(\widehat e_i,c_i)
-\gamma_{\mathrm{anc}}k_{\sigma_{\mathrm{anc}}}(\widehat e_i,a_i).
\]

The positive first term is minimized by moving away from the forget center. When
`gamma_anc > 0`, the negative second term is minimized by moving toward the
promptless teacher anchor. Both kernels are bounded, so
\(\ell_{f,i}\in[-\gamma_{\mathrm{anc}},1]\), and their gradients decay away
from the corresponding kernel regions rather than driving an unbounded
negative-log-likelihood ascent. With this wrapper's default `GAMMA_ANC=0`, the
second term contributes nothing and \(\ell_{f,i}=k_{\mathrm{rep}}\in[0,1]\).

With a fixed bandwidth, the kernel is

\[
k_\sigma(u,v)=\exp\left(-\frac{\lVert u-v\rVert^2}{2\sigma^2}\right).
\]

The launcher's default `SIGMA=median` instead uses the SVGD median heuristic at
every draw:

\[
h=\frac{\operatorname{median}_{i<j}
 \lVert\widehat e_i-\widehat e_j\rVert^2}{\log n},
\qquad
k_{\mathrm{rep}}(u,v)=\exp\left(-\frac{\lVert u-v\rVert^2}{h}\right),
\]

where \(n\) is the number of selected-position soft embeddings in the current
batch. If fewer than two points exist or all pairwise distances are zero, the
code falls back to `SIGMA_FALLBACK=0.05`, represented as
\(h=2\sigma_{\mathrm{fallback}}^2\). The anchor always uses the fixed
`SIGMA_ANC=1.0` bandwidth.

Implementation:
[`DiffusionSBRUTSD._forget_kl`](../src/trainer/dlm/unlearn/sbru_tsd.py#L161-L220)
and
[`_median_bandwidth`](../src/trainer/dlm/unlearn/sbru_tsd.py#L125-L138).

### 2.5 Retain-free self-distillation

At the other masked answer positions \(i\in R\), the student matches the
question-conditioned frozen teacher:

\[
\ell_{d,i}=D_{\mathrm{KL}}\left(
 p_{\theta_0}(\cdot\mid x,y_{\mathcal M})
 \;\|\;
 p_\theta(\cdot\mid x,y_{\mathcal M})
\right).
\]

For an example with at least one position in \(R\), the losses are combined as

\[
\mathcal L_{\mathrm{ex}}
=\alpha\operatorname{mean}_{i\in F}\ell_{f,i}
+(1-\alpha)\operatorname{mean}_{i\in R}\ell_{d,i}.
\]

If a draw has no positions in \(R\), the implementation uses only the mean
forget loss instead of scaling it by \(\alpha\). An example with an empty
selected set contributes zero. The default `alpha=0.5` gives equal coefficient
to forgetting and self-distillation when both are present.

This is called **retain-free** because no examples from the TOFU retain split
are loaded during training. The preservation signal comes from non-selected
answer positions in the forget examples. This does not claim that the method
has no effect on the external retain set; retain behavior is measured after
training by the evaluation suite.

Implementation:
[`_combine_draw`](../src/trainer/dlm/unlearn/simple_tsd.py#L475-L555).

### 2.6 Multi-draw optimization

The four loss masks reuse one PCSS selection per batch. During training, each
draw is forwarded and backpropagated before the next student graph is created;
the four scaled gradients sum to the mean loss. This avoids retaining all four
student activation graphs simultaneously under ZeRO-3. The final draw performs
synchronization only when the outer Hugging Face gradient-accumulation step is
at its normal synchronization boundary.

Implementation:
[`training_step`](../src/trainer/dlm/unlearn/simple_tsd.py#L575-L618).

### 2.7 Algorithm summary

```text
Start with the retain100 fine-tuned checkpoint.
Create a frozen teacher and an initially identical trainable student.

For each forget-set minibatch:
    Run PCSS under several partial masks.
    Select prompt-dependent answer positions A_hat.
    Gather the detached unembedding matrix E once.

    For each sampled diffusion mask M:
        Run student and frozen teacher on the same partially masked answer.
        F = M intersect A_hat
        R = M minus A_hat

        At F:
            compute the student's soft predicted embedding
            repel it from the forget center
            if gamma_anc > 0, attract it toward the promptless frozen-teacher anchor

        At R:
            match the original teacher with KL self-distillation

        combine the two means using alpha
        backpropagate this draw's scaled contribution

After unlearning, save the final model and run the TOFU DLM evaluation suite.
```

## 3. What `run_sbrutsd.sh` does

### 3.1 Launcher defaults

The script enables strict Bash behavior with `set -euo pipefail` and changes to
the Git repository root. It then reads the following environment variables:

| Environment variable | Launcher default | Meaning |
|---|---:|---|
| `MODEL` | `llada` | Model family. Accepted by the common launcher: `llada` or `dream`. |
| `GPUS` | `0,1,2,3` | Comma-separated physical GPU IDs made visible to Accelerate. |
| `SPLIT` | `forget10` | TOFU forget split. Common aliases also accept `unlearn90`; supported families are 10%, 5%, and 1% forget. |
| `TASK_SUFFIX` | `_sbrutsd_run1` | Suffix added to the task name, output directory, and log name. Use a unique value for concurrent or repeated runs. |
| `SIGMA` | `median` | Adaptive repulsion bandwidth, or a positive fixed `sigma`. |
| `SIGMA_FALLBACK` | `0.05` | Fixed repulsion sigma used when the median heuristic degenerates. |
| `SIGMA_ANC` | `1.0` | Fixed promptless-anchor bandwidth. |
| `GAMMA_ANC` | `0.0` | Weight of anchor attraction. The current wrapper disables the anchor term by default even though the trainer YAML defaults it to `1.0`. |
| `ALPHA` | `0.5` | Forget/self-distillation trade-off when both position sets are present. |
| `CENTRE_MODE` | `gold` | Forget center: `gold` or `ref_soft`. |

It creates `logs/` and writes the detached process's standard output and error
to:

```text
logs/sbrutsd${TASK_SUFFIX}_no_distil.log
```

With defaults, this is `logs/sbrutsd_sbrutsd_run1_no_distil.log`. The
`_no_distil` label is hard-coded in the log filename only; it does **not** turn
off the inherited self-distillation term. Use `ALPHA=1` (subject to the empty-set
loss behavior described below) if the intent is to remove its weight.

### 3.2 Command launched

Conceptually, the wrapper runs:

```bash
TASK_SUFFIX="$TASK_SUFFIX" setsid bash scripts/dlm/tofu/unlearn.sh \
  "$MODEL" "$GPUS" "$SPLIT" SBRUTSD \
  trainer.method_args.sigma="$SIGMA" \
  trainer.method_args.sigma_fallback="$SIGMA_FALLBACK" \
  trainer.method_args.sigma_anc="$SIGMA_ANC" \
  trainer.method_args.gamma_anc="$GAMMA_ANC" \
  trainer.method_args.alpha="$ALPHA" \
  trainer.method_args.centre_mode="$CENTRE_MODE" \
  <additional arguments>
```

The real command redirects standard input from `/dev/null`, redirects both
output streams to the log, and ends with `&`.

This has several consequences:

- `setsid` starts the job in a new session so it can survive the launching
  terminal closing;
- `&` makes the wrapper return immediately;
- the printed PID is the detached background process PID;
- any extra arguments passed to `run_sbrutsd.sh` are appended verbatim and are
  normally used as Hydra overrides; and
- training or evaluation failures after launch are visible in the log but are
  not returned as the wrapper's exit status.

## 4. Downstream execution flow

```text
run_sbrutsd.sh
  |
  +-- scripts/dlm/tofu/unlearn.sh
        |
        +-- scripts/dlm/lib.sh
        |     resolve model, GPUs, checkpoint, split, epochs, task, and method
        |
        +-- accelerate launch src/train.py --config-name=unlearn.yaml
        |     experiment=unlearn/tofu/dlm_simple_tsd
        |     trainer=DiffusionSBRUTSD
        |       -> DiffusionSBRUTSD
        |          -> inherited DiffusionSimpleTSD pipeline
        |
        +-- save final model under saves/unlearn/<task>
        |
        +-- unless NO_EVAL is enabled:
              accelerate launch src/eval.py eval=tofu_dlm
              -> results under saves/eval/<task>
```

### 4.1 Model resolution

[`dlm_config`](../scripts/dlm/lib.sh#L57-L87) expands the launcher's short model
name:

| `MODEL` input | Hydra model config | Start checkpoint directory | Other automatic settings |
|---|---|---|---|
| `llada` | `LLaDA-8B-Instruct` | `saves/finetune/tofu_LLaDA-8B-Instruct_retain100_150epochs` | no right shift; scheduler loss weighting; gradient checkpointing off |
| `dream` | `Dream-v0-Instruct-7B` | `saves/finetune/tofu_Dream-v0-Instruct-7B_retain100_150_epochs` | right-shift logits; gradient checkpointing on; scheduler loss weighting on this launch path unless explicitly overridden |

Although `dlm_config` initially associates Dream with `cart[geo_p:0.3]`,
[`unlearn.sh`](../scripts/dlm/tofu/unlearn.sh#L4) exports
`LOSS_WEIGHT_OVERRIDE=scheduler` by default before calling it. Consequently,
this wrapper actually launches **both** model families with scheduler weighting
unless the caller sets, for example,
`LOSS_WEIGHT_OVERRIDE='cart[geo_p:0.3]'`. Dream users who want the CART recipe
shown in the experiment comments must request it explicitly.

If the checkpoint directory itself contains `config.json`, it is used directly;
otherwise, the numerically greatest immediate `checkpoint-*` child is selected.
If neither exists, checkpoint resolution returns the expected path and model
loading later fails. The wrapper does not perform an early checkpoint existence
check.

`START_MODEL=/some/checkpoint` can override the resolved start checkpoint via
the common launcher.

### 4.2 Split and epoch resolution

[`tofu_split`](../scripts/dlm/lib.sh#L108-L116) maps the requested split:

| `SPLIT` | Forget split | Retain split used for evaluation naming/config | Holdout split | Default epochs |
|---|---|---|---|---:|
| `forget10` or `unlearn90` | `forget10` | `retain90` | `holdout10` | 20 |
| `forget05` or `unlearn95` | `forget05` | `retain95` | `holdout05` | 30 |
| `forget01` or `unlearn99` | `forget01` | `retain99` | `holdout01` | 50 |

Despite resolving a retain split, SBRU-TSD's experiment config explicitly sets
the training retain dataset to `null`. The retain split remains useful to the
TOFU evaluation configuration.

`UNLEARN_EPOCHS` overrides the default epoch count. The generic suite can also
accept comma-separated split and `EPOCHS` lists, although `run_sbrutsd.sh` is
primarily written for one split per invocation.

### 4.3 Method and Hydra configuration

The string `SBRUTSD` is normalized to `DiffusionSBRUTSD`, validated against
[`configs/trainer/DiffusionSBRUTSD.yaml`](../configs/trainer/DiffusionSBRUTSD.yaml),
and mapped to the experiment
[`configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml`](../configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml).

The effective configuration layers are:

1. [`configs/unlearn.yaml`](../configs/unlearn.yaml), the root unlearning config;
2. [`dlm_simple_tsd.yaml`](../configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml),
   which selects the TOFU forget dataset and removes the retain dataset;
3. [`DiffusionSBRUTSD.yaml`](../configs/trainer/DiffusionSBRUTSD.yaml), which
   inherits [`DiffusionSimpleTSD.yaml`](../configs/trainer/DiffusionSimpleTSD.yaml)
   and ultimately
   [`DiffusionFinetune.yaml`](../configs/trainer/DiffusionFinetune.yaml);
4. the common launcher's model-, split-, batch-, and epoch-specific CLI
   overrides; and
5. SBRU-TSD launcher overrides plus any user-supplied trailing Hydra arguments.

Hydra CLI values have the final precedence. In particular, the trainer YAML has
`sigma: 0.05` and `gamma_anc: 1.0`, but this wrapper changes them to
`sigma=median` and `gamma_anc=0.0` unless `SIGMA` or `GAMMA_ANC` is set.
Consequently, the wrapper's default selected-position objective is repulsion
only; the promptless anchor is still computed and logged, but contributes zero
to the loss. Launching `DiffusionSBRUTSD` through another path can therefore
have different bandwidth and anchor behavior.

The inherited `demotion` and `belief_top_k` options do **not** affect
SBRU-TSD: they belong to SimpleTSD's edited-teacher forget target, and
SBRU-TSD completely overrides that selected-position objective. Selection,
masking, self-distillation, `mask_weight_power`, `gamma`, and `ball_k` remain
relevant.

### 4.4 Distributed training command

The common launcher validates the requested GPU IDs and invokes roughly:

```bash
CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
  --config_file configs/accelerate/dlm_config.yaml \
  --num_processes <number-of-GPU-IDs> \
  --main_process_port <available-port> \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/dlm_simple_tsd \
  model=<resolved-model> \
  model.model_args.pretrained_model_name_or_path=<resolved-SFT-checkpoint> \
  trainer=DiffusionSBRUTSD \
  task_name=<computed-task> \
  forget_split=<forget-split> \
  retain_split=<retain-split> \
  holdout_split=<holdout-split> \
  trainer.args.num_train_epochs=<resolved-epochs> \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.save_strategy=no \
  trainer.args.gradient_checkpointing=<model-specific> \
  trainer.method_args.right_shift_logits=<model-specific> \
  trainer.method_args.loss_weight_type=scheduler \
  <SBRU-TSD overrides>
```

The default effective batch size is
`4 examples/GPU x 4 accumulation steps x number of GPUs`: `64` with the
launcher's four-GPU default. The command shows the effective default
`loss_weight_type=scheduler` used by this launch path. For Dream, set
`LOSS_WEIGHT_OVERRIDE='cart[geo_p:0.3]'` before invoking the wrapper if the CART
weighting associated with Dream in `dlm_config` is intended.

By default, Accelerate uses DeepSpeed ZeRO-3 with CPU optimizer and parameter
offload as configured in
[`configs/accelerate/dlm_config.yaml`](../configs/accelerate/dlm_config.yaml).
Set `OFFLOAD=0` to select `dlm_config_nooffload.yaml`, or set `UNLEARN_ACCEL` to
an explicit Accelerate config.

Although `trainer.args.save_strategy=no` disables periodic checkpoints,
[`src/train.py`](../src/train.py#L104-L113) calls `save_model` after successful
training, so the final model is saved. The consequence of the default is that
there is no periodic recovery checkpoint if the run fails before completion.
Set `UNLEARN_SAVE_STRATEGY=epoch` if intermediate checkpoints are required.

### 4.5 Task names and output locations

With the default model, split, and suffix, the task is:

```text
forget10/tofu_LLaDA-8B-Instruct_DiffusionSBRUTSD_sbrutsd_run1
```

Outputs are therefore expected at:

```text
logs/sbrutsd_sbrutsd_run1_no_distil.log
saves/unlearn/forget10/tofu_LLaDA-8B-Instruct_DiffusionSBRUTSD_sbrutsd_run1/
saves/eval/forget10/tofu_LLaDA-8B-Instruct_DiffusionSBRUTSD_sbrutsd_run1/
```

The unlearning directory contains the final model after successful training.
The evaluation directory contains at least:

```text
TOFU_EVAL.json
TOFU_SUMMARY.json
```

`TASK_SUFFIX` should be changed for a new run unless overwriting/reusing the
same output and log paths is intentional. `OUTPUT_ROOT` affects paths computed
inside the generic suite for its skip check and unlearn/eval checkpoint
argument, but the repository's standard Hydra `paths.output_dir` remains rooted
at `saves/`; using a nondefault output root therefore requires matching Hydra
path overrides as well.

### 4.6 Evaluation

There are two distinct evaluation paths.

**Training-time evaluation.** The inherited trainer configuration keeps
`do_eval=true`, `eval_on_start=true`, and `eval_strategy=epoch`; the unlearning
suite does not override them. [`FinetuneTrainer.evaluate`](../src/trainer/base.py)
can run its custom evaluator only with a single Accelerate process. Therefore:

- on the normal multi-GPU launch, each scheduled call warns that the custom
  evaluator requires one process and returns without running it;
- on a one-GPU launch, the full TOFU evaluator can run before training, at each
  epoch boundary, and once more after training through [`src/train.py`](../src/train.py).

Single-GPU training-time results are placed under
`saves/unlearn/<task>/checkpoint-<step>/evals/`. These are separate from the
post-training suite output described next.

**Post-training evaluation.** After `src/train.py` finishes, `unlearn.sh`
launches [`src/eval.py`](../src/eval.py) against the saved final model unless
`NO_EVAL` is set to a truthy value. The default suite in
[`configs/eval/tofu_dlm.yaml`](../configs/eval/tofu_dlm.yaml) evaluates diffusion
ROUGE and probability-based TOFU metrics, including forget and retain Q/A
scores, paraphrased forget probability/truth ratio, retain truth ratio, real
authors, and world facts. Probability metrics use `mc_num=128` diffusion Monte
Carlo samples by default. Its results are written to `saves/eval/<task>/`.

Post-training evaluation uses `EVAL_GPUS` if set, otherwise the training `GPUS`,
and defaults to
[`configs/accelerate/eval_4gpu.yaml`](../configs/accelerate/eval_4gpu.yaml). The
common launcher overrides its process count to match the GPU list.

## 5. Prerequisites

Before launching, verify the following:

1. **Run from a Git checkout.** The wrapper executes
   `git rev-parse --show-toplevel`; it exits before launching if the current
   location is not within the repository.
2. **Initialize the `dllm` dependency/submodule.** The DLLM trainer imports its
   schedulers and Dream utilities.
3. **Install the project's Python dependencies, including Accelerate and
   DeepSpeed.** Use the environment intended for this repository.
4. **Make the requested NVIDIA GPUs available.** Each integer in `GPUS` must be
   less than the count reported by `nvidia-smi`.
5. **Create or provide the retain100 SFT checkpoint.** The default expected path
   depends on the selected model, as listed above. Use `START_MODEL` to point to
   an alternative compatible checkpoint.
6. **Ensure Hugging Face access/data availability.** The forget dataset config
   reads the requested split from `locuslab/TOFU`.
7. **Budget for two model copies and repeated forwards.** The trainer holds a
   trainable student and frozen teacher. Default PCSS performs up to 16 teacher
   forwards per batch (eight with the prompt and eight promptless), followed by
   four training draws, each with student and teacher work plus promptless
   anchor computation. ZeRO-3/offload and per-draw backpropagation help memory,
   but the method is compute intensive.

## 6. Running the launcher

Run these commands from anywhere inside the Git checkout; the wrapper changes
to the repository root.

### 6.1 Default LLaDA run

```bash
bash scripts/dlm/tofu/run_sbrutsd.sh
```

This launches LLaDA on GPUs `0,1,2,3`, forgets `forget10` for 20 epochs, and
evaluates the final model.

### 6.2 Dream on two GPUs and the 5% split

```bash
MODEL=dream \
GPUS=0,1 \
SPLIT=forget05 \
TASK_SUFFIX=_dream_f05_run1 \
bash scripts/dlm/tofu/run_sbrutsd.sh
```

This uses scheduler loss weighting because `unlearn.sh` sets that as the default
override for this launch path. To use Dream's CART recipe instead, launch with:

```bash
LOSS_WEIGHT_OVERRIDE='cart[geo_p:0.3]' \
MODEL=dream \
GPUS=0,1 \
SPLIT=forget05 \
TASK_SUFFIX=_dream_f05_cart_run1 \
bash scripts/dlm/tofu/run_sbrutsd.sh
```

### 6.3 Fixed repulsion bandwidth

```bash
SIGMA=0.1 \
SIGMA_ANC=2.0 \
GAMMA_ANC=1.0 \
ALPHA=0.5 \
TASK_SUFFIX=_fixed_sigma_01 \
bash scripts/dlm/tofu/run_sbrutsd.sh
```

### 6.4 Additional Hydra overrides

Trailing arguments pass directly to the training command. For example:

```bash
TASK_SUFFIX=_topk_run1 \
bash scripts/dlm/tofu/run_sbrutsd.sh \
  trainer.method_args.select_method=topk \
  trainer.method_args.topk_frac=0.35 \
  trainer.method_args.loss_mc=6 \
  trainer.method_args.ball_k=8
```

A trailing override can supersede a value generated from an environment
variable because it appears later on the Hydra command line. Avoid supplying
the same key twice unless this precedence is intentional.

### 6.5 Save periodic checkpoints and skip evaluation

```bash
UNLEARN_SAVE_STRATEGY=epoch \
NO_EVAL=1 \
TASK_SUFFIX=_train_only \
bash scripts/dlm/tofu/run_sbrutsd.sh
```

### 6.6 Monitor the run

The wrapper prints a ready-to-use command. With the default suffix:

```bash
tail -f logs/sbrutsd_sbrutsd_run1_no_distil.log
```

The script returning successfully means only that the job was launched. Check
the log and the expected `TOFU_SUMMARY.json` to establish end-to-end success.
The common result table can be displayed with:

```bash
python scripts/dlm/show_results.py
```

Useful training metrics include:

| Metric | Interpretation |
|---|---|
| `sbru_tsd/L_rep` | Mean repulsion-kernel value at selected masked positions; minimizing pushes this downward. |
| `sbru_tsd/L_anc` | Mean anchor-kernel similarity; `gamma_anc > 0` encourages this upward. It is diagnostic only at the wrapper default `GAMMA_ANC=0`. |
| `sbru_tsd/dist_c` | Distance from the current soft embedding to the forget center. |
| `sbru_tsd/dist_a` | Distance to the promptless anchor. |
| `sbru_tsd/sigma_rep` | Effective repulsion sigma, including the adaptive median value. |
| `simple_tsd/forget_kl` | Historical metric name for the mean selected-position objective; in SBRU-TSD it is not literally a KL. |
| `simple_tsd/distill_kl` | Frozen-teacher KL on other masked answer positions. |
| `simple_tsd/selected_frac` | Fraction of answer positions selected by PCSS. |
| `simple_tsd/skipped_examples` | Batch examples with no selected forget position. |
| `simple_tsd/mask_rate` | Masked-position count divided by answer-position count. |

## 7. Tuning guidance

- **`ALPHA`** controls the central trade-off. Lower values strengthen
  self-distillation when \(R\) is present; higher values emphasize semantic
  repulsion. It has no scaling effect on draws with an empty \(R\), where the
  implementation deliberately uses the forget loss alone.
- **`SIGMA=median`** adapts to current selected-position embedding distances and
  is the wrapper default. A fixed positive value makes runs easier to compare
  but is sensitive to the model's embedding scale.
- **`SIGMA_FALLBACK`** matters when a batch has fewer than two selected
  positions or coincident embeddings. Frequent fallback can indicate overly
  sparse selection or very small active batches.
- **`SIGMA_ANC`** controls how broad the anchor's attraction basin is when
  `GAMMA_ANC>0`. It should generally be wider than a small fixed repulsion sigma;
  otherwise both kernels can vanish after a short move from the forget center.
  The current wrapper default `GAMMA_ANC=0` disables this attraction, so changing
  `SIGMA_ANC` alone does not change the loss.
- **Selection parameters** (`select_method`, `otsu_relax`, `sel_mc`,
  `drop_question_overlap`) control what is forgotten. Too broad a set erodes the
  self-distillation region; too narrow a set leaves prompt-specific information
  untouched.
- **`loss_mc` and `select_temp`** control trajectory coverage. In cover mode,
  values below four trigger a warning because score-weighted sampling may cover
  low-score selected positions poorly.
- **`mask_weight_power`** can emphasize high-mask, early-generation states.
  Its default `0.0` weights draws uniformly.
- **`ball_k`** broadens the gold center to nearby vocabulary embeddings, but
  nearest-neighbor search is performed over the vocabulary and can add
  substantial computation. The wrapper leaves it at zero.

Hyperparameters should be selected from forget effectiveness and retain/model
utility together, not from training loss alone.

## 8. Caveats and failure modes

1. **Fire-and-forget status:** the wrapper cannot report whether detached
   training or evaluation later fails. Inspect the log and output artifacts.
2. **Log and output collisions:** reusing `TASK_SUFFIX` truncates the same log at
   launch and reuses the same task directory. Concurrent runs need unique
   suffixes and non-overlapping GPU allocations.
3. **No periodic checkpoint by default:** the final model is saved after a
   successful train, but a crash before then has no epoch checkpoint unless
   `UNLEARN_SAVE_STRATEGY` is changed.
4. **Median fallback can change the effective scale abruptly:** sparse PCSS
   selection can alternate between an adaptive bandwidth and the fixed
   fallback.
5. **Retain-free does not mean preservation is guaranteed:** the utility guard
   is local self-distillation on non-selected tokens from forget examples. PCSS
   mistakes or overly aggressive repulsion can still reduce retain metrics.
6. **`alpha=0` does not universally disable forgetting:** a draw without
   self-distillation positions uses the forget objective alone.
7. **Inherited demotion options are inactive:** `demotion` and `belief_top_k`
   are accepted through inheritance but are bypassed by SBRU-TSD's overridden
   forget objective.
8. **Wrapper and YAML objective defaults differ:** this wrapper uses
   `sigma=median` and `gamma_anc=0.0`; the trainer config alone uses fixed
   `sigma=0.05` and `gamma_anc=1.0`. Thus the wrapper disables anchor attraction
   by default despite still computing its diagnostics.
9. **The `_no_distil` log suffix is only a filename:** self-distillation remains
   active at the default `ALPHA=0.5`; the suffix does not configure the trainer.
10. **SFT path spelling differs by model:** the LLaDA default contains
   `150epochs`, whereas Dream contains `150_epochs`.
11. **GPU validation is index-based:** malformed GPU lists may fail later, and a
    race between concurrently selected “free” ports is still theoretically
    possible.
12. **Evaluation is expensive:** the default probability metrics use 128 Monte
    Carlo samples after an already compute-heavy unlearning phase. On one-GPU
    training, inherited `eval_on_start` and epoch evaluation can additionally
    run that custom evaluator multiple times; multi-GPU scheduled evaluations
    only warn and return.
13. **Dream does not automatically get CART weighting here:** `unlearn.sh`
    defaults `LOSS_WEIGHT_OVERRIDE` to `scheduler`. Set
    `LOSS_WEIGHT_OVERRIDE='cart[geo_p:0.3]'` explicitly when that Dream recipe is
    intended.
14. **The generic suite's `SKIP_DONE=1` checks for an existing evaluation
    summary:** with a reused task name, it can skip the entire run when
    `saves/eval/<task>/TOFU_SUMMARY.json` already exists.

## 9. Implementation map

| Component | File |
|---|---|
| User-facing launcher | [`scripts/dlm/tofu/run_sbrutsd.sh`](../scripts/dlm/tofu/run_sbrutsd.sh) |
| Generic unlearn/evaluate suite | [`scripts/dlm/tofu/unlearn.sh`](../scripts/dlm/tofu/unlearn.sh) |
| Shared model, split, GPU, launch, and evaluation helpers | [`scripts/dlm/lib.sh`](../scripts/dlm/lib.sh) |
| SBRU-TSD selected-position objective | [`src/trainer/dlm/unlearn/sbru_tsd.py`](../src/trainer/dlm/unlearn/sbru_tsd.py) |
| PCSS, masking, self-distillation, and multi-draw training | [`src/trainer/dlm/unlearn/simple_tsd.py`](../src/trainer/dlm/unlearn/simple_tsd.py) |
| SBRU-TSD trainer defaults | [`configs/trainer/DiffusionSBRUTSD.yaml`](../configs/trainer/DiffusionSBRUTSD.yaml) |
| Inherited SimpleTSD defaults | [`configs/trainer/DiffusionSimpleTSD.yaml`](../configs/trainer/DiffusionSimpleTSD.yaml) |
| Retain-free TOFU experiment wiring | [`configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml`](../configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml) |
| Hydra training entry point | [`src/train.py`](../src/train.py) |
| Trainer registry | [`src/trainer/__init__.py`](../src/trainer/__init__.py) |
| TOFU DLM evaluation configuration | [`configs/eval/tofu_dlm.yaml`](../configs/eval/tofu_dlm.yaml) |
| Related detailed SimpleTSD documentation | [`docs/simple_tsd_algorithms.md`](simple_tsd_algorithms.md) |
| Evidence/commentary for the earlier SBRU anchor sweep | [`scripts/dlm/tofu/run_sbru_anc_sweep.sh`](../scripts/dlm/tofu/run_sbru_anc_sweep.sh) |
