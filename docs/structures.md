# `dlm_mu` Structure

## 1. Purpose

`dlm_mu` is a machine-unlearning research repository for autoregressive language models and masked discrete diffusion language models (DLLMs). It extends the OpenUnlearning-style training and evaluation framework with DLLM-specific models, trainers, unlearning objectives, and experiment scripts.

The repository is organized around four layers:

1. **Hydra configuration** in [`configs/`](../configs/) declares models, data, trainers, evaluations, and experiments.
2. **Python implementation** in [`src/`](../src/) loads those configurations and performs training or evaluation.
3. **DLLM primitives** in the [`dllm/`](../dllm/) submodule supply diffusion schedulers and model-specific functionality.
4. **Experiment scripts** in [`scripts/dlm/`](../scripts/dlm/) compose command-line overrides, launch distributed jobs, and aggregate results.

Generated checkpoints and metrics are normally written under `saves/`. This directory is created by runs or data-download commands and is not present in a fresh checkout.

---

## 2. Top-level map

```text
dlm_mu/
├── configs/            Hydra configuration tree
├── dllm/               Git submodule with general DLLM infrastructure
├── docs/               Algorithm and repository documentation
├── scripts/dlm/        DLLM experiment, evaluation, and analysis scripts
├── src/                First-party data, model, trainer, and evaluation code
├── outputs/            Legacy or ad hoc Hydra outputs
├── setup.py            Python package metadata
├── setup_data.py       Dataset and evaluation-artifact download utility
└── .gitmodules         DLLM submodule declaration

Generated when needed:
├── data/               Downloaded datasets
└── saves/              Checkpoints and evaluation results
```

### Directory roles

| Path | Role |
|---|---|
| [`configs/`](../configs/) | Defines the runtime composition of models, data, trainers, evaluation, generation, paths, and distributed launch settings. |
| [`dllm/`](../dllm/) | External Git submodule containing reusable discrete diffusion language-model components. |
| [`docs/`](./) | Method and architecture documentation. |
| [`scripts/dlm/`](../scripts/dlm/) | Launchers, sweeps, smoke tests, probes, and result utilities for DLLM experiments. |
| [`src/`](../src/) | Main Python package used by training and evaluation entry points. |
| `saves/` *(generated)* | Configured destination for checkpoints, logs, and evaluation summaries. |
| [`outputs/`](../outputs/) | Output from legacy or manually configured Hydra runs; not the configured primary save root. |

---

## 3. Runtime architecture

A typical training or unlearning run follows this path:

```text
shell script or direct CLI command
              │
              ▼
Hydra root config: train.yaml or unlearn.yaml
              │
              ├── model config
              ├── data and dataset configs
              ├── collator config
              ├── trainer config
              ├── evaluation config
              └── experiment overlay
              │
              ▼
src/train.py
              │
              ├── model.get_model(...)
              ├── data.get_data(...)
              ├── data.get_collators(...)
              ├── evals.get_evaluators(...)
              └── trainer.load_trainer(...)
              │
              ▼
Hugging Face Trainer / DLLM trainer
              │
              ▼
saves/<mode>/<task_name>/
```

Evaluation-only runs use [`src/eval.py`](../src/eval.py) and [`configs/eval.yaml`](../configs/eval.yaml).

---

## 4. Configuration tree

The project uses [Hydra](https://hydra.cc/) to compose configurations. Most behavior should be changed through YAML or command-line overrides instead of hard-coded edits.

```text
configs/
├── train.yaml
├── unlearn.yaml
├── eval.yaml
├── accelerate/
├── collator/
├── data/
│   └── datasets/
├── eval/
├── experiment/
├── generation/
├── hydra/
├── model/
├── paths/
└── trainer/
```

### 4.1 Root configurations

- [`configs/train.yaml`](../configs/train.yaml) is the default configuration for fine-tuning through [`src/train.py`](../src/train.py).
- [`configs/unlearn.yaml`](../configs/unlearn.yaml) configures the same Python entry point for machine unlearning.
- [`configs/eval.yaml`](../configs/eval.yaml) configures standalone evaluation through [`src/eval.py`](../src/eval.py).

An unlearning command normally selects the root configuration explicitly:

```bash
python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/dlm_simple_tsd \
    trainer=DiffusionSimpleTSD \
    task_name=my_simple_tsd_run
```

Production scripts generally invoke this through `accelerate launch` rather than plain Python.

### 4.2 Model configurations

[`configs/model/`](../configs/model/) contains model-specific settings, including:

- LLaDA;
- Dream;
- Llama;
- Qwen;
- Phi;
- Gemma; and
- other autoregressive baselines.

A model config specifies the model class or loading path, tokenizer settings, model arguments, and prompt-template arguments consumed by [`get_model`](../src/model/__init__.py).

### 4.3 Trainer configurations

[`configs/trainer/`](../configs/trainer/) contains one YAML file per training or unlearning method. Each file typically defines:

```yaml
handler: DiffusionSimpleTSD
method_args:
  alpha: 0.5
  # ...
```

`handler` must match a class registered in [`src/trainer/__init__.py`](../src/trainer/__init__.py). `method_args` are passed directly to that trainer's constructor.

The trainer configs cover both:

- **autoregressive methods**, such as gradient ascent, gradient difference, NPO, RMU, WGA, and others; and
- **DLLM methods**, such as DiffusionGradDiff, MDU, GAP, AURA, TSD, SimpleTSD, Semantic_Unlearn, UTSD, CSD, SBRU, and SBRU-TSD.

### 4.4 Data configurations

[`configs/data/`](../configs/data/) defines fine-tuning and unlearning data composition. Individual dataset definitions live in [`configs/data/datasets/`](../configs/data/datasets/).

The repository includes configurations for benchmarks such as:

- TOFU;
- MUSE;
- RWKU; and
- WMDP.

Unlearning configurations can combine forget and retain datasets or remove the retain dataset entirely for forget-only methods.

### 4.5 Experiment overlays

[`configs/experiment/`](../configs/experiment/) contains complete experiment recipes. They override several Hydra groups together, such as the model, trainer, dataset, and evaluator.

The main organization is:

```text
configs/experiment/
├── finetune/
├── unlearn/
│   ├── tofu/
│   ├── muse/
│   ├── rwku/
│   └── wmdp/
├── eval/
└── examples/
```

For example, [`configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml`](../configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml) selects:

- LLaDA as the default model;
- `DiffusionSimpleTSD` as the trainer;
- the TOFU forget dataset;
- no retain dataset; and
- the DLLM TOFU evaluator.

### 4.6 Evaluation and generation configurations

- [`configs/eval/`](../configs/eval/) defines benchmark evaluators and metric sets.
- [`configs/generation/`](../configs/generation/) defines decoding settings used by generation-based DLLM evaluation.

DLLM-specific evaluator configurations coexist with their autoregressive counterparts, for example TOFU versus TOFU-DLM.

### 4.7 Distributed training

[`configs/accelerate/`](../configs/accelerate/) contains Accelerate and DeepSpeed configurations, including ZeRO-3 and CPU-offload variants.

These files are selected by the shell launchers in [`scripts/dlm/`](../scripts/dlm/) and passed to `accelerate launch`.

### 4.8 Paths and Hydra output

[`configs/paths/default.yaml`](../configs/paths/default.yaml) defines the primary output location as

```text
saves/<mode>/<task_name>/
```

Hydra run behavior and log destinations are configured under [`configs/hydra/`](../configs/hydra/).

---

## 5. Source tree

```text
src/
├── train.py
├── eval.py
├── data/
├── model/
├── trainer/
│   ├── unlearn/
│   └── dlm/
│       └── unlearn/
└── evals/
    └── metrics/
```

### 5.1 Entry points

#### Training and unlearning

[`src/train.py`](../src/train.py) is the shared entry point for fine-tuning and unlearning. It:

1. seeds the run;
2. loads the model and tokenizer;
3. loads and combines datasets;
4. creates the data collator;
5. creates configured evaluators;
6. resolves the trainer class through the registry;
7. optionally resumes a checkpoint; and
8. runs training and configured evaluation.

The selected root YAML determines whether the run is fine-tuning or unlearning.

#### Evaluation

[`src/eval.py`](../src/eval.py) loads a model and each configured evaluator, then executes the evaluator independently.

### 5.2 Model layer

[`src/model/`](../src/model/) is responsible for model and tokenizer loading.

| File | Purpose |
|---|---|
| [`src/model/__init__.py`](../src/model/__init__.py) | Main `get_model` factory and model-selection logic. |
| [`src/model/dlm.py`](../src/model/dlm.py) | DLLM-specific loading and model handling. |
| [`src/model/probe.py`](../src/model/probe.py) | Model probing support. |

### 5.3 Data layer

[`src/data/`](../src/data/) contains dataset factories, preprocessing, and collators.

| File | Purpose |
|---|---|
| [`src/data/__init__.py`](../src/data/__init__.py) | `get_data`, dataset loading, and collator factories. |
| [`src/data/unlearn.py`](../src/data/unlearn.py) | Forget/retain dataset construction for unlearning. |
| [`src/data/qa.py`](../src/data/qa.py) | Question-answer dataset handling. |
| [`src/data/pretraining.py`](../src/data/pretraining.py) | Pretraining-style data handling. |
| [`src/data/rwku.py`](../src/data/rwku.py) | RWKU-specific data support. |
| [`src/data/collators.py`](../src/data/collators.py) | Padding, label masks, and batch collation. |
| [`src/data/utils.py`](../src/data/utils.py) | Shared data utilities. |

A forget/retain trainer generally receives a nested batch shaped conceptually as:

```python
{
    "forget": {...},
    "retain": {...},  # omitted in retain-free experiments
}
```

### 5.4 Trainer registry

[`src/trainer/__init__.py`](../src/trainer/__init__.py) is the central trainer factory.

It:

1. imports available trainer classes;
2. stores them in `TRAINER_REGISTRY` under each class name;
3. reads `trainer.handler` from Hydra; and
4. instantiates the matching class with `trainer.method_args`.

Adding a new trainer normally requires:

1. implementing its class;
2. exporting/importing it into the trainer package;
3. registering it in [`src/trainer/__init__.py`](../src/trainer/__init__.py); and
4. adding a corresponding YAML under [`configs/trainer/`](../configs/trainer/).

### 5.5 Autoregressive trainers

[`src/trainer/unlearn/`](../src/trainer/unlearn/) contains unlearning methods for conventional autoregressive language models. It includes shared unlearning behavior and methods such as:

- gradient ascent and gradient difference;
- NPO and SimNPO;
- DPO;
- RMU;
- CEU;
- SatImp;
- UNDIAL;
- WGA; and
- PDU.

Shared fine-tuning infrastructure is in [`src/trainer/base.py`](../src/trainer/base.py).

### 5.6 DLLM trainers

[`src/trainer/dlm/`](../src/trainer/dlm/) is the masked-diffusion training stack.

#### Shared DLLM training

[`src/trainer/dlm/base.py`](../src/trainer/dlm/base.py) defines `DiffusionFinetuneTrainer`. Its responsibilities include:

- constructing the diffusion alpha schedule;
- sampling timesteps and noising answer tokens;
- computing DLLM loss weights;
- handling LLaDA/Dream output conventions;
- diffusion NLL computation; and
- DLLM-specific logging callbacks.

[`src/trainer/dlm/callbacks.py`](../src/trainer/dlm/callbacks.py) contains progress and console-printing callbacks.

#### Shared DLLM unlearning

[`src/trainer/dlm/unlearn/base.py`](../src/trainer/dlm/unlearn/base.py) provides common unlearning functionality, including:

- frozen reference-model preparation;
- forget and retain diffusion NLL;
- optional retain loss for forget-only runs;
- vocabulary-normalized log probabilities; and
- common per-token reductions and rollout utilities.

#### DLLM unlearning methods

[`src/trainer/dlm/unlearn/`](../src/trainer/dlm/unlearn/) contains the method implementations:

| File | Main method or role |
|---|---|
| [`grad_ascent.py`](../src/trainer/dlm/unlearn/grad_ascent.py) | DLLM gradient-ascent baseline. |
| [`grad_diff.py`](../src/trainer/dlm/unlearn/grad_diff.py) | Forget ascent plus retain NLL. |
| [`npo.py`](../src/trainer/dlm/unlearn/npo.py) | DLLM NPO. |
| [`simnpo.py`](../src/trainer/dlm/unlearn/simnpo.py) | DLLM SimNPO. |
| [`wga.py`](../src/trainer/dlm/unlearn/wga.py) | DLLM WGA. |
| [`mdu.py`](../src/trainer/dlm/unlearn/mdu.py) | Masked diffusion unlearning objective. |
| [`gap.py`](../src/trainer/dlm/unlearn/gap.py) | GAP method. |
| [`aura.py`](../src/trainer/dlm/unlearn/aura.py) | AURA method. |
| [`tsd.py`](../src/trainer/dlm/unlearn/tsd.py) | Trajectory-selective distillation. |
| [`simple_tsd.py`](../src/trainer/dlm/unlearn/simple_tsd.py) | Retain-free SimpleTSD. |
| [`semantic_unlearn.py`](../src/trainer/dlm/unlearn/semantic_unlearn.py) | Forget-only semantic-target unlearning with RBF and shifted-cosine targets. |
| [`auto_unlearning.py`](../src/trainer/dlm/unlearn/auto_unlearning.py) | Closed-form anti-conditional unlearning. |
| [`simple_tsd_qmask.py`](../src/trainer/dlm/unlearn/simple_tsd_qmask.py) | Promptless-target SimpleTSD variant. |
| [`uniform_tsd.py`](../src/trainer/dlm/unlearn/uniform_tsd.py) | Uniform TSD variant. |
| [`concept_subspace.py`](../src/trainer/dlm/unlearn/concept_subspace.py) | Concept-subspace distillation/unlearning. |
| [`sbru.py`](../src/trainer/dlm/unlearn/sbru.py) | Semantic-ball repulsion unlearning. |
| [`sbru_tsd.py`](../src/trainer/dlm/unlearn/sbru_tsd.py) | SBRU composed with the SimpleTSD pipeline. |
| [`rollout.py`](../src/trainer/dlm/unlearn/rollout.py) | Shared reverse-trajectory and completion utilities. |

For details about SimpleTSD, see [`docs/simple_tsd_algorithms.md`](simple_tsd_algorithms.md). For Semantic_Unlearn, begin with the target-distribution table and theory in [`docs/semantic_unlearn.md`](semantic_unlearn.md), then continue to its separate implementation section.

### 5.7 Evaluation layer

[`src/evals/`](../src/evals/) contains evaluator factories and benchmark implementations.

| Path | Purpose |
|---|---|
| [`src/evals/__init__.py`](../src/evals/__init__.py) | Evaluator factory. |
| [`src/evals/base.py`](../src/evals/base.py) | Shared evaluator interfaces and behavior. |
| [`src/evals/tofu.py`](../src/evals/tofu.py) | TOFU evaluation. |
| [`src/evals/muse.py`](../src/evals/muse.py) | MUSE evaluation. |
| [`src/evals/rwku.py`](../src/evals/rwku.py) | RWKU evaluation. |
| [`src/evals/lm_eval.py`](../src/evals/lm_eval.py) | Integration with the LM Evaluation Harness. |
| [`src/evals/metrics/`](../src/evals/metrics/) | Utility, memorization, privacy, judge, and membership-inference metrics. |

The metrics directory includes both autoregressive and DLLM-specific implementations.

---

## 6. The `dllm` submodule

[`dllm/`](../dllm/) is declared as a Git submodule in [`.gitmodules`](../.gitmodules). It points to a separate diffusion-language-model repository and is not ordinary first-party source duplicated into `src/`.

It supplies reusable components imported by the local DLLM trainer stack, including:

- diffusion schedulers;
- LLaDA and Dream pipelines;
- loss weighting utilities;
- training tools; and
- model examples.

Its internal layout includes:

```text
dllm/
├── dllm/
│   ├── core/
│   ├── pipelines/
│   ├── tools/
│   └── utils/
├── examples/
├── scripts/
├── docs/
└── README.md
```

The boundary is:

- [`dllm/`](../dllm/) provides general DLLM machinery;
- [`src/trainer/dlm/`](../src/trainer/dlm/) adapts it to this repository's fine-tuning, unlearning, and evaluation framework.

After cloning the main repository, initialize this dependency with:

```bash
git submodule update --init --recursive
```

---

## 7. Experiment scripts

[`scripts/dlm/`](../scripts/dlm/) is the operational layer used for reproducible DLLM experiments.

```text
scripts/dlm/
├── lib.sh
├── aggregate_results.py
├── show_results.py
├── run_fluency.py
├── run_judge.py
├── tofu/
├── muse/
└── rwku/
```

### 7.1 Shared launch library

[`scripts/dlm/lib.sh`](../scripts/dlm/lib.sh) provides reusable shell functions for:

- GPU detection and validation;
- LLaDA/Dream presets;
- benchmark split presets;
- mapping method names to Hydra experiment configs;
- fine-tuning, unlearning, and evaluation launch commands; and
- evaluation-plan orchestration.

Most benchmark scripts source this file rather than duplicating the full `accelerate launch` command.

### 7.2 TOFU scripts

[`scripts/dlm/tofu/`](../scripts/dlm/tofu/) contains:

- generic training and unlearning drivers;
- method-specific launch wrappers;
- hyperparameter sweeps;
- evaluator and judge scripts;
- smoke tests; and
- analysis and visualization probes.

Important SimpleTSD-related files include:

| File | Purpose |
|---|---|
| [`unlearn.sh`](../scripts/dlm/tofu/unlearn.sh) | Generic TOFU unlearning driver. |
| [`simple_tsd.sh`](../scripts/dlm/tofu/simple_tsd.sh) | Primary SimpleTSD launcher. |
| [`simpletsd_qmask.sh`](../scripts/dlm/tofu/simpletsd_qmask.sh) | Promptless-target variant launcher. |
| [`simple_tsd_smoke_test.py`](../scripts/dlm/tofu/simple_tsd_smoke_test.py) | Fast method-level sanity checks. |
| [`run_semantic_unlearn.sh`](../scripts/dlm/tofu/run_semantic_unlearn.sh) | Semantic_Unlearn launcher. |
| [`run_auto_unlearning.sh`](../scripts/dlm/tofu/run_auto_unlearning.sh) | Closed-form Auto Unlearning launcher. |
| [`simple_tsd_probe_forget.py`](../scripts/dlm/tofu/simple_tsd_probe_forget.py) | Inspect selected forget positions. |
| [`simple_tsd_probe_to_html.py`](../scripts/dlm/tofu/simple_tsd_probe_to_html.py) | Render probe output for inspection. |
| [`simple_tsd_viz_target.py`](../scripts/dlm/tofu/simple_tsd_viz_target.py) | Visualize edited target behavior. |

The same directory contains scripts for TSD, UTSD, CSD, GAP, MDU, SBRU, and other methods.

### 7.3 Other benchmarks

- [`scripts/dlm/muse/`](../scripts/dlm/muse/) contains MUSE fine-tuning, forgetting, and TSD launchers.
- [`scripts/dlm/rwku/`](../scripts/dlm/rwku/) contains RWKU method and baseline launchers.

### 7.4 Result utilities

- [`scripts/dlm/aggregate_results.py`](../scripts/dlm/aggregate_results.py) aggregates result artifacts.
- [`scripts/dlm/show_results.py`](../scripts/dlm/show_results.py) renders comparison tables.
- `run_fluency` and `run_judge` scripts execute auxiliary scoring passes.

---

## 8. Outputs and generated artifacts

### 8.1 Primary output tree

The configured output root is `saves/` (created when a run or download writes to it):

```text
saves/
├── finetune/
│   └── <task_name>/
│       └── checkpoint-*/
├── unlearn/
│   └── <task_name>/
│       └── checkpoint-*/
└── eval/
    └── <task_name>/
        └── benchmark summaries and metrics
```

A checkpoint can contain:

- model shards and index files;
- tokenizer files;
- optimizer and scheduler state;
- DeepSpeed ZeRO state;
- RNG state;
- trainer state; and
- Hydra configuration snapshots.

These generated artifacts should not be treated as source code.

### 8.2 `outputs/`

[`outputs/`](../outputs/) contains legacy or ad hoc run products. The current path configuration points normal runs to `saves/`, so new code should not assume `outputs/` is authoritative.

### 8.3 Data downloads

[`setup_data.py`](../setup_data.py) downloads optional datasets and evaluation artifacts. Depending on its options, it populates locations such as:

- `data/`;
- `data/wmdp/`; and
- `saves/eval/`.

---

## 9. Common workflows

### 9.1 Fine-tune a DLLM

1. Select a model under [`configs/model/`](../configs/model/).
2. Select or create a fine-tuning experiment under [`configs/experiment/finetune/`](../configs/experiment/finetune/).
3. Launch [`src/train.py`](../src/train.py) through the appropriate Accelerate configuration.
4. Use the resulting checkpoint as the initial model for unlearning.

### 9.2 Run an unlearning method

1. Choose a trainer from [`configs/trainer/`](../configs/trainer/).
2. Choose an experiment overlay from [`configs/experiment/unlearn/`](../configs/experiment/unlearn/).
3. Set a unique `task_name`.
4. Launch with [`scripts/dlm/tofu/unlearn.sh`](../scripts/dlm/tofu/unlearn.sh), another benchmark script, or a direct `accelerate launch` command.
5. Inspect checkpoints and logs under `saves/unlearn/<task_name>/`.

### 9.3 Evaluate a checkpoint

1. Select the correct evaluator under [`configs/eval/`](../configs/eval/).
2. Point the model configuration to the saved checkpoint.
3. Run [`src/eval.py`](../src/eval.py), normally through a benchmark script.
4. Aggregate or display results with the utilities in [`scripts/dlm/`](../scripts/dlm/).

### 9.4 Add a DLLM unlearning method

A typical addition touches these locations:

```text
src/trainer/dlm/unlearn/<method>.py
src/trainer/dlm/unlearn/__init__.py
src/trainer/__init__.py
configs/trainer/<TrainerClass>.yaml
configs/experiment/unlearn/<benchmark>/<experiment>.yaml
scripts/dlm/<benchmark>/<launcher>.sh       # optional
scripts/dlm/<benchmark>/<smoke_test>.py     # recommended
```

The implementation should inherit from the narrowest suitable existing trainer, reuse common noising and reduction utilities, and match existing logging namespaces.

---

## 10. Where to start reading

### To understand the whole application

1. [`configs/unlearn.yaml`](../configs/unlearn.yaml)
2. [`src/train.py`](../src/train.py)
3. [`src/trainer/__init__.py`](../src/trainer/__init__.py)
4. [`src/data/__init__.py`](../src/data/__init__.py)
5. [`src/model/__init__.py`](../src/model/__init__.py)

### To understand DLLM training

1. [`src/trainer/dlm/base.py`](../src/trainer/dlm/base.py)
2. [`src/trainer/dlm/unlearn/base.py`](../src/trainer/dlm/unlearn/base.py)
3. [`dllm/README.md`](../dllm/README.md)

### To understand SimpleTSD

1. [`docs/simple_tsd_algorithms.md`](simple_tsd_algorithms.md)
2. [`src/trainer/dlm/unlearn/simple_tsd.py`](../src/trainer/dlm/unlearn/simple_tsd.py)
3. [`configs/trainer/DiffusionSimpleTSD.yaml`](../configs/trainer/DiffusionSimpleTSD.yaml)
4. [`configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml`](../configs/experiment/unlearn/tofu/dlm_simple_tsd.yaml)
5. [`scripts/dlm/tofu/simple_tsd.sh`](../scripts/dlm/tofu/simple_tsd.sh)

### To understand Semantic_Unlearn

1. [`docs/semantic_unlearn.md`](semantic_unlearn.md)
2. [`src/trainer/dlm/unlearn/semantic_unlearn.py`](../src/trainer/dlm/unlearn/semantic_unlearn.py)
3. [`configs/trainer/DiffusionSemanticUnlearn.yaml`](../configs/trainer/DiffusionSemanticUnlearn.yaml)
4. [`configs/experiment/unlearn/tofu/dlm_semantic_unlearn.yaml`](../configs/experiment/unlearn/tofu/dlm_semantic_unlearn.yaml)
5. [`scripts/dlm/tofu/run_semantic_unlearn.sh`](../scripts/dlm/tofu/run_semantic_unlearn.sh)

### To understand Auto Unlearning

1. [`docs/auto_unlearning.md`](auto_unlearning.md)
2. [`src/trainer/dlm/unlearn/auto_unlearning.py`](../src/trainer/dlm/unlearn/auto_unlearning.py)
3. [`configs/trainer/DiffusionAutoUnlearning.yaml`](../configs/trainer/DiffusionAutoUnlearning.yaml)
4. [`configs/experiment/unlearn/tofu/dlm_auto_unlearning.yaml`](../configs/experiment/unlearn/tofu/dlm_auto_unlearning.yaml)
5. [`scripts/dlm/tofu/run_auto_unlearning.sh`](../scripts/dlm/tofu/run_auto_unlearning.sh)

### To understand SBRU and SBRU-TSD

1. [`src/trainer/dlm/unlearn/sbru.py`](../src/trainer/dlm/unlearn/sbru.py)
2. [`src/trainer/dlm/unlearn/sbru_tsd.py`](../src/trainer/dlm/unlearn/sbru_tsd.py)
3. [`configs/trainer/DiffusionSBRU.yaml`](../configs/trainer/DiffusionSBRU.yaml)
4. [`configs/trainer/DiffusionSBRUTSD.yaml`](../configs/trainer/DiffusionSBRUTSD.yaml)
