import logging
import os

import hydra
from omegaconf import DictConfig
from transformers.trainer_utils import get_last_checkpoint
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything

logger = logging.getLogger(__name__)

# Values (case-insensitive) that mean "find and resume the latest checkpoint".
_AUTO_RESUME = {"auto", "last", "latest", "true"}
# Values that mean "no resume, train from scratch".
_NO_RESUME = {"", "none", "null", "false"}


def resolve_resume_checkpoint(resume, output_dir):
    """Turn the ``resume_from_checkpoint`` config into the arg ``Trainer.train`` wants.

    Accepts:
      * None / False / "none" / "false"  -> ``None`` (fresh run).
      * True / "auto" / "last"           -> newest ``checkpoint-*`` in ``output_dir``,
        or ``None`` with a warning if there is none. This makes it safe to leave
        resume on for every launch: the first run starts fresh, a relaunch after a
        crash picks up where it stopped.
      * an explicit path                 -> that path (must exist).
    """
    if resume is None or resume is False:
        return None
    if isinstance(resume, str) and resume.strip().lower() in _NO_RESUME:
        return None
    if resume is True or (isinstance(resume, str) and resume.strip().lower() in _AUTO_RESUME):
        last = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        if last is None:
            logger.warning(
                "resume_from_checkpoint=%r but no checkpoint found under %s; "
                "starting from scratch.",
                resume,
                output_dir,
            )
            return None
        logger.info("Resuming training from latest checkpoint: %s", last)
        return last
    # Explicit checkpoint path.
    if not os.path.isdir(resume):
        raise FileNotFoundError(f"resume_from_checkpoint path does not exist: {resume}")
    logger.info("Resuming training from checkpoint: %s", resume)
    return resume


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        resume = resolve_resume_checkpoint(
            cfg.get("resume_from_checkpoint", None), trainer_args.output_dir
        )
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
