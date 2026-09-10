import torch
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainingArguments

from trainer.base import FinetuneTrainer
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.npo import NPO
from trainer.unlearn.dpo import DPO
from trainer.unlearn.simnpo import SimNPO
from trainer.unlearn.rmu import RMU
from trainer.unlearn.undial import UNDIAL
from trainer.unlearn.ceu import CEU
from trainer.unlearn.satimp import SatImp
from trainer.unlearn.wga import WGA
from trainer.unlearn.pdu import PDU

# Diffusion Language Model (DLM) trainers — kept in the trainer.dlm subpackage,
# separate from the autoregressive LLM trainers above.
from trainer.dlm import DiffusionFinetuneTrainer
from trainer.dlm.unlearn import (
    DiffusionGradAscent,
    DiffusionGradDiff,
    DiffusionNPO,
    DiffusionSimNPO,
    DiffusionWGA,
    DiffusionMDU,
    DiffusionGAP,
    DiffusionAURA,
    DiffusionTSD,
    DiffusionSimpleTSD,
    DiffusionSemanticUnlearn,
    DiffusionAutoUnlearning,
    DiffusionSelectiveUnlearning,
    DiffusionSimpleTSDQMask,
    DiffusionUTSD,
    DiffusionCSD,
    DiffusionSBRU,
    DiffusionSBRUTSD,
)


import logging

logger = logging.getLogger(__name__)

TRAINER_REGISTRY: Dict[str, Any] = {}


def _register_trainer(trainer_class):
    TRAINER_REGISTRY[trainer_class.__name__] = trainer_class


def load_trainer_args(trainer_args: DictConfig, dataset):
    # Fully convert to plain Python containers: nested OmegaConf nodes (e.g.
    # gradient_checkpointing_kwargs) would otherwise reach TrainingArguments as
    # DictConfig and break its JSON serialization.
    if isinstance(trainer_args, DictConfig):
        trainer_args = OmegaConf.to_container(trainer_args, resolve=True)
    else:
        trainer_args = dict(trainer_args)
    warmup_epochs = trainer_args.pop("warmup_epochs", None)
    if warmup_epochs:
        batch_size = trainer_args["per_device_train_batch_size"]
        grad_accum_steps = trainer_args["gradient_accumulation_steps"]
        num_devices = torch.cuda.device_count()
        dataset_len = len(dataset)
        trainer_args["warmup_steps"] = int(
            (warmup_epochs * dataset_len)
            // (batch_size * grad_accum_steps * num_devices)
        )

    trainer_args = TrainingArguments(**trainer_args)
    return trainer_args


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset=None,
    eval_dataset=None,
    processing_class=None,
    data_collator=None,
    evaluators=None,
    template_args=None,
):
    trainer_args = trainer_cfg.args
    method_args = trainer_cfg.get("method_args", {})
    trainer_args = load_trainer_args(trainer_args, train_dataset)
    trainer_handler_name = trainer_cfg.get("handler")
    assert trainer_handler_name is not None, ValueError(
        f"{trainer_handler_name} handler not set"
    )
    trainer_cls = TRAINER_REGISTRY.get(trainer_handler_name, None)
    assert trainer_cls is not None, NotImplementedError(
        f"{trainer_handler_name} not implemented or not registered"
    )
    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        data_collator=data_collator,
        args=trainer_args,
        evaluators=evaluators,
        template_args=template_args,
        **method_args,
    )
    logger.info(
        f"{trainer_handler_name} Trainer loaded, output_dir: {trainer_args.output_dir}"
    )
    return trainer, trainer_args


# Register Finetuning Trainer
_register_trainer(Trainer)
_register_trainer(FinetuneTrainer)

# Register Unlearning Trainer
_register_trainer(GradAscent)
_register_trainer(GradDiff)
_register_trainer(NPO)
_register_trainer(DPO)
_register_trainer(SimNPO)
_register_trainer(RMU)
_register_trainer(UNDIAL)
_register_trainer(CEU)
_register_trainer(SatImp)
_register_trainer(WGA)
_register_trainer(PDU)

# Register DLM Trainers
_register_trainer(DiffusionFinetuneTrainer)
_register_trainer(DiffusionGradAscent)
_register_trainer(DiffusionGradDiff)
_register_trainer(DiffusionNPO)
_register_trainer(DiffusionSimNPO)
_register_trainer(DiffusionWGA)
_register_trainer(DiffusionMDU)
_register_trainer(DiffusionGAP)
_register_trainer(DiffusionAURA)
_register_trainer(DiffusionTSD)
_register_trainer(DiffusionSimpleTSD)
_register_trainer(DiffusionSemanticUnlearn)
_register_trainer(DiffusionAutoUnlearning)
_register_trainer(DiffusionSelectiveUnlearning)
_register_trainer(DiffusionSimpleTSDQMask)
_register_trainer(DiffusionUTSD)
_register_trainer(DiffusionCSD)
_register_trainer(DiffusionSBRU)
_register_trainer(DiffusionSBRUTSD)
