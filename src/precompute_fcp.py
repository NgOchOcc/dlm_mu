"""Hydra entry point for FCP-MDLM offline cap construction (Algorithm 1).

Loads the finetuned model theta0 (from the standard model config, typically
overridden to the finetuned checkpoint) and a pre-exposure model theta_b (from
``fcp.theta_b_path``), builds the per-fact atom bank + caps over the forget
split, and saves the cache to ``fcp.output_path``.

Example (see scripts/dlm/*/fcp.sh):
  CUDA_VISIBLE_DEVICES=0 python src/precompute_fcp.py \
      experiment=unlearn/tofu/dlm_fcp model=LLaDA-8B-Instruct \
      model.model_args.pretrained_model_name_or_path=<theta0_ckpt> \
      +fcp.theta_b_path=GSAI-ML/LLaDA-8B-Instruct \
      +fcp.output_path=saves/unlearn/fcp_cache/<task>.pt \
      +fcp.kappa=0.9 +fcp.right_shift_logits=false
"""

import os
import copy

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from data import get_data
from model import get_model
from trainer.dlm.unlearn.fcp_precompute import build_fact_cache


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    fcp_cfg = cfg.get("fcp", None)
    assert fcp_cfg is not None, "Pass +fcp.theta_b_path=... +fcp.output_path=..."
    theta_b_path = fcp_cfg.theta_b_path
    output_path = fcp_cfg.output_path
    kappa = float(fcp_cfg.get("kappa", 0.9))
    strict_epsilon = fcp_cfg.get("strict_epsilon", None)
    right_shift_logits = bool(fcp_cfg.get("right_shift_logits", False))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = cfg.model
    template_args = model_cfg.template_args

    # theta0 (finetuned) + shared tokenizer.
    theta0, tokenizer = get_model(model_cfg)
    theta0 = theta0.to(device)

    # theta_b (pre-exposure): same config, swapped checkpoint path.
    theta_b_cfg = copy.deepcopy(OmegaConf.to_container(model_cfg, resolve=True))
    theta_b_cfg["model_args"]["pretrained_model_name_or_path"] = theta_b_path
    theta_b_cfg["tokenizer_args"]["pretrained_model_name_or_path"] = theta_b_path
    theta_b, _ = get_model(OmegaConf.create(theta_b_cfg))
    theta_b = theta_b.to(device)

    # Forget dataset (the ForgetRetainDataset's .forget QADataset).
    data = get_data(
        cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=template_args
    )
    forget_dataset = data["train"].forget

    cache = build_fact_cache(
        theta0=theta0,
        theta_b=theta_b,
        forget_dataset=forget_dataset,
        tokenizer=tokenizer,
        right_shift_logits=right_shift_logits,
        kappa=kappa,
        strict_epsilon=strict_epsilon,
        device=device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(cache, output_path)
    print(f"[fcp-precompute] saved cache -> {output_path}")
    print(f"[fcp-precompute] meta: {cache['meta']}")


if __name__ == "__main__":
    main()
