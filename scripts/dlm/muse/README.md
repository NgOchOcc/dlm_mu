# MUSE — step-by-step commands

Every command below trains **and** evaluates: `run.sh` runs the MUSE evaluation
right after each checkpoint it produces, so there is no separate eval step.

Run everything from the repository root, with the environment active:

```bash
cd /prj/corp/airesearch/lasvegas/vol20-scratch/nluu/working/open-unlearning
conda activate unlearning
```

## Pipeline

| Phase | Trains on | Produces |
|---|---|---|
| 1. Corpus SFT | `MUSE_train` (raw text) | `saves/finetune/muse_<MODEL>_<split>_full` |
| 2. Knowmem SFT | `knowmem/forget_qa + retain_qa` | `saves/finetune/muse_<MODEL>_<split>_knowmem` |
| 3. Unlearn | `knowmem/forget_qa` | `saves/unlearn/muse_<MODEL>_<split>_<Method>` |

Evaluation always covers all three probes: `forget_knowmem`, `retain_knowmem`,
`forget_verbmem`.

Phase 1 is assumed to exist already. Check before starting:

```bash
ls -d saves/finetune/muse_*_full
```

Phase 2 exists because unlearning and evaluation both use the knowmem QA
format, while phase 1 only ever sees raw text — without it the model is scored
in a format it was never trained on.

## Step 1 — Knowmem SFT (phase 2), both models

Trains Dream first, then LLaDA-1.5, and evaluates each checkpoint.

```bash
KNOWMEM_SFT_EPOCHS=5 NO_UNLEARN=1 \
  bash scripts/dlm/muse/run.sh "dream llada1.5" 0,1,2,3 News
```

`NO_UNLEARN=1` stops after phase 2. Drop it to continue straight into
unlearning with the default method list.

To run one model at a time:

```bash
KNOWMEM_SFT_EPOCHS=5 NO_UNLEARN=1 bash scripts/dlm/muse/run.sh dream     0,1,2,3 News
KNOWMEM_SFT_EPOCHS=5 NO_UNLEARN=1 bash scripts/dlm/muse/run.sh llada1.5  0,1,2,3 News
```

Verify before moving on:

```bash
ls -d saves/finetune/muse_*_knowmem
python3 scripts/dlm/show_results.py --dataset muse
```

## Step 2 — Unlearn the baselines (WGA, NPO, BST)

Methods run in the order given, and the models run in sequence. Phase 2 is
skipped automatically once its checkpoint exists.

```bash
METHODS=DiffusionWGA,DiffusionNPO,DiffusionBST \
  bash scripts/dlm/muse/run.sh "dream llada1.5" 0,1,2,3 News
```

One model at a time:

```bash
METHODS=DiffusionWGA,DiffusionNPO,DiffusionBST \
  bash scripts/dlm/muse/run.sh dream 0,1,2,3 News

METHODS=DiffusionWGA,DiffusionNPO,DiffusionBST \
  bash scripts/dlm/muse/run.sh llada1.5 0,1,2,3 News
```

## Step 3 — Unlearn with SelectiveUnlearning

Trains for `SDU_EPOCHS` (default 10) rather than the 2 the baselines use.

```bash
METHODS=DiffusionSelectiveUnlearning \
  bash scripts/dlm/muse/run.sh "dream llada1.5" 0,1,2,3 News
```

One model at a time:

```bash
METHODS=DiffusionSelectiveUnlearning \
  bash scripts/dlm/muse/run.sh dream 0,1,2,3 News

METHODS=DiffusionSelectiveUnlearning \
  bash scripts/dlm/muse/run.sh llada1.5 0,1,2,3 News
```

To sweep the beta / preserve-weight grid instead of a single configuration:

```bash
BETAS="2 5 10 20 50" PRESERVE_WEIGHTS="0.1" \
  bash scripts/dlm/muse/selective_sweep.sh "dream llada1.5" 0,1,2,3 News
```

## Results

```bash
python3 scripts/dlm/show_results.py --dataset muse
python3 scripts/dlm/show_results.py --dataset muse --detail
python3 scripts/dlm/show_results.py --dataset muse --data-split News
```

`EQ` and `MU` only become non-empty once the LLM judge has run — they combine
ROUGE with the judge's Relevance and Correctness:

```bash
JUDGE_BASE_URL=http://127.0.0.1:8000/v1 \
JUDGE_MODEL=Qwen/Qwen3.8-27B \
JUDGE_WORKERS=8 \
nohup .venv-vllm/bin/python -u \
    scripts/dlm/evaluate_all_llm_judge.py \
    --benchmark muse --eval-root saves/eval \
    --poll-seconds 60 --batch-size 16 \
    > logs/judge_muse.log 2>&1 &
```

## Options

| Variable | Default | Effect |
|---|---|---|
| `KNOWMEM_SFT` | `1` | `0` skips phase 2 entirely |
| `KNOWMEM_SFT_EPOCHS` | `5` | Epochs for phase 2 (200 QA pairs) |
| `UNLEARN_EPOCHS` | `2` | Epochs for the baselines |
| `SDU_EPOCHS` | `10` | Epochs for SDU / SelectiveUnlearning |
| `DATA_SPLITS` | `News Books` | Third positional argument |
| `NO_EVAL` | off | Train without evaluating |
| `NO_EVAL_SFT` | off | Evaluate unlearning only, not phase 2 |
| `OVERWRITE` | off | Redo work that already has results |
| `RUN_TAG` | empty | Suffix for checkpoint/eval names (sweeps) |

Everything resumes: a run whose evaluation summary already exists is skipped,
so re-running the same command after an interruption continues where it left
off. `OVERWRITE=1` forces a redo.

## Notes

Quote a multi-word model list — `"dream llada1.5"`. Without the quotes
`llada1.5` lands in the GPU argument.

Phase 2 trains on `forget_qa` as well as `retain_qa`, so the model memorises
the forget answers immediately before unlearning. That is MUSE's intent — there
has to be something to forget — but it makes the task harder than a pipeline
without phase 2, and the scores are not comparable with earlier runs.
