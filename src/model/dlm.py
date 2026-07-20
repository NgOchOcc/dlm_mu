"""Loader for masked Diffusion Language Models (Dream, LLaDA).

Diffusion LMs differ from autoregressive LLMs in how the checkpoint is
instantiated (``AutoModelForMaskedLM`` / ``AutoModel`` with a diffusion
``[MASK]`` token) and in tokenizer setup (mask token, right padding). Those
details are delegated to the ``dllm`` package so this framework does not have
to duplicate model-specific customization.

This module is intentionally kept separate from the autoregressive loader in
``model/__init__.py``; the only coupling point is a single dispatch branch in
``get_model`` that routes ``model_type: dlm`` configs here.
"""

import logging

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def get_dlm_model(model_cfg: DictConfig):
    """Load a diffusion LM and its tokenizer via the ``dllm`` package.

    The model config reuses the same schema as autoregressive models
    (``model_args.pretrained_model_name_or_path``, ``torch_dtype``, ...) so
    that both kinds of models are configured uniformly; the fields are mapped
    onto ``dllm``'s ``ModelArguments`` here.
    """
    try:
        from dllm.utils import get_model as dllm_get_model
        from dllm.utils import get_tokenizer as dllm_get_tokenizer
        from dllm.utils.configs import ModelArguments as DLMModelArguments
    except ImportError as e:
        raise ImportError(
            "The `dllm` package is required to load diffusion language models. "
            "Install it (editable) from the Information-Gain-Sampler repo, e.g.\n"
            "  pip install -e /path/to/Information-Gain-Sampler/dllm"
        ) from e

    # Importing the pipeline model packages registers dllm's bundled LLaDA/Dream
    # classes with the transformers Auto* factories (model_type "llada"/"Dream").
    # This must happen *before* dllm_get_model so AutoModel.from_pretrained
    # resolves to dllm's class instead of the checkpoint's remote code — the two
    # are distinct class objects, and get_tokenizer's issubclass checks (which
    # set the diffusion [MASK] token) only match dllm's class.
    for _pipeline in ("llada", "dream"):
        try:
            __import__(f"dllm.pipelines.{_pipeline}.models")
        except ImportError:
            pass

    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = OmegaConf.to_container(model_cfg.model_args, resolve=True)
    model_name_or_path = model_args.get("pretrained_model_name_or_path", None)
    assert model_name_or_path is not None, ValueError(
        "pretrained_model_name_or_path missing from diffusion model_args."
    )

    dllm_args = DLMModelArguments(
        model_name_or_path=model_name_or_path,
        dtype=model_args.get("torch_dtype", "bfloat16"),
        load_in_4bit=model_args.get("load_in_4bit", False),
        attn_implementation=model_args.get("attn_implementation", None),
        lora=model_args.get("lora", False),
    )

    try:
        model = dllm_get_model(model_args=dllm_args)
        tokenizer = dllm_get_tokenizer(model_args=dllm_args)
    except Exception as e:
        logger.warning(
            f"Diffusion model {model_name_or_path} requested with {model_args}"
        )
        raise ValueError(
            f"Error {e} while loading diffusion model via dllm.utils.get_model()."
        )
    return model, tokenizer
