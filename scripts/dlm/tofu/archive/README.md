# Archived TOFU scripts

Superseded by `unlearn.sh`, `retain.sh` and `evaluate.sh` in the parent
directory. Kept for the run history they document, **not** maintained: they were
written against the old `lib.sh` (`EPOCHS`, `TOFU_TRAINERS`, `RETAIN90_CKPT`,
`batch_plan`) and will not run as-is.

| archived | replaced by |
| --- | --- |
| `unlearn90.sh`, `unlearn95.sh`, `unlearn99.sh` | `unlearn.sh <model> [gpus] unlearn90\|unlearn95\|unlearn99` |
| `unlearn90_wsched.sh` | `unlearn.sh` — the 1/t ELBO weight it forced is now the default for both backbones |
| `finetune.sh` | `retain.sh <model> [gpus] retain100` |
| `retrain90.sh`, `retrain95.sh`, `retrain99.sh` | `retain.sh <model> [gpus] retain90\|retain95\|retain99` |
| `eval_base.sh`, `eval_retain.sh`, `eval_all.sh` | `evaluate.sh <model> [gpus] [split] base\|retain\|methods\|all` (`PARALLEL=1` is the old `eval_all.sh` behaviour), or `evaluate_retain.sh <model> [gpus]` for every reference model across its settings in one call |
| `all_llada.sh`, `all_llada_4gpu.sh`, `alldream.sh`, `runall.sh` | driver lists — write the two or three `unlearn.sh`/`retain.sh` calls directly |

Kept because they record measurements or one-off procedures that are not
reproducible from the new scripts:

- `traj_rmdu_forget01.sh` — measured step time / peak memory per accelerate
  config (2 GPU offload 77 s/it, 4 GPU no-offload 12.8-16 s/it, LLaDA OOM at 2
  GPUs without offload). Note its `GRAD_ACCUM=2` never took effect: the old
  `batch_plan` computed a plan the launchers then ignored, hardcoding 4x4. In the
  current `lib.sh`, `GRAD_ACCUM` and `TRAIN_BS` do apply.
- `swap_arm2.sh` — supervisor that killed and replaced arm 2 of a running
  `traj_iproj` sweep in place.
- `verify_oldbase.sh` — checked whether the old high Dream retain numbers were
  model health or pre-truncation eval inflation.
