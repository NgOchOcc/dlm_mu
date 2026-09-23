import os

import torch
import datasets
import numpy as np
import logging
from typing import List, Dict, Any, Union

IGNORE_INDEX = -100  # TODO put in common constants

logger = logging.getLogger("data")


def _local_dataset_dir(path, name, split):
    """Where scripts/dlm/muse/download_data.py saves (repo, config, split)."""
    root = os.environ.get("DLM_DATASET_DIR")
    if not root:
        return None
    return os.path.join(root, path.replace("/", "__"), name or "default", split or "")


def _load_dataset_maybe_local(path, **kwargs):
    """Load from a pre-downloaded directory when one exists, else from the Hub.

    Datasets saved with `save_to_disk` are Arrow directories that
    `load_dataset` cannot read, so they are loaded with `load_from_disk`.
    Setting DLM_DATASET_DIR makes a run use them and skip the Hub entirely;
    anything not found there still falls back to a normal Hub load.
    """
    name, split = kwargs.get("name"), kwargs.get("split")

    # "a+b" concatenates splits; each part was saved separately, so load them
    # individually and join. Without this the combined name matches no
    # directory and the whole load silently falls back to the Hub.
    if split and "+" in split:
        parts = [p.strip() for p in split.split("+") if p.strip()]
        dirs = [_local_dataset_dir(path, name, p) for p in parts]
        if all(d and os.path.isdir(d) for d in dirs):
            logger.info(f"loading {path} split {split} from {os.path.dirname(dirs[0])}")
            return datasets.concatenate_datasets(
                [datasets.load_from_disk(d) for d in dirs]
            )
    else:
        local = _local_dataset_dir(path, name, split)
        if local and os.path.isdir(local):
            logger.info(f"loading {path} from {local}")
            return datasets.load_from_disk(local)

    return datasets.load_dataset(path, **kwargs)


def load_hf_dataset(path, filter_column=None, filter_value=None, **kwargs):
    """Load a HF dataset, optionally keeping only rows matching a column value.

    `filter_column`/`filter_value` support per-entity benchmarks such as RWKU,
    where every split carries a `subject` column and each unlearning run targets
    a single entity. `filter_value` may be a scalar or a list of accepted
    values. Raises if the filter matches nothing, so a typo in an entity name
    fails loudly instead of silently training on an empty set.
    """
    dataset = _load_dataset_maybe_local(path, **kwargs)
    # A null `filter_value` means "no filtering": configs wire `filter_column`
    # in statically and leave the value unset (e.g. eval.rwku_dlm.target_subject
    # = null) to evaluate the full test set.
    if filter_column is None or filter_value is None:
        return dataset
    accepted = set(filter_value if isinstance(filter_value, (list, tuple)) else [filter_value])
    if filter_column not in dataset.column_names:
        raise ValueError(
            f"filter_column '{filter_column}' not in {path} columns {dataset.column_names}"
        )
    dataset = dataset.filter(lambda row: row[filter_column] in accepted)
    if len(dataset) == 0:
        raise ValueError(
            f"filter {filter_column}={sorted(accepted)} matched 0 rows in {path}"
        )
    logger.info(f"{path}: filtered to {len(dataset)} rows where {filter_column} in {sorted(accepted)}")
    return dataset


def preprocess_chat_instance(
    tokenizer,
    template_config: Dict[str, Any],
    prompt_msgs: Union[List[str], str],
    response_msgs: Union[List[str], str],
    max_length: int,
    predict_with_generate: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocesses a chat instance for training or generation.
    When in training, both the returned `input_ids` and `labels` cover the entire conversation.
    `input_ids` has no padding, and `labels` assign `IGNORE_INDEX` to tokens where loss is not computed (i.e. all tokens except the final response message).
    When in generation, `input_ids` are returned only up to the last user prompt, excluding the assistant's response. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full `input_ids` token sequence.

    `prompt_msgs` and `response_msgs` are lists where, except for the last pair, all
    corresponding pairs are in-context examples. When they are a string and not
    a list, there are no in-context examples.

    Args:
        tokenizer: Tokenizer to apply on text
        template_config (Dict[str, Any]): Configuration for the chat template (comes from model-specific config).
        prompt_msgs (Union[List[str], str]): List of prompt messages or a single prompt message string.
        response_msgs (Union[List[str], str]): List of response messages or a single response message string.
        max_length (int): Maximum sequence length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    assert len(prompt_msgs) == len(response_msgs)
    if isinstance(prompt_msgs, str):
        assert isinstance(response_msgs, str)
        prompt_msgs, response_msgs = [prompt_msgs], [response_msgs]

    if template_config["apply_chat_template"]:
        chat = []
        system_prompt = template_config.get("system_prompt", None)
        if system_prompt:
            chat += [{"role": "system", "content": system_prompt}]
        for prompt, response in zip(prompt_msgs, response_msgs):
            chat += [{"role": "user", "content": prompt}]
            chat += [{"role": "assistant", "content": response}]
        date_str = template_config.get("date_string", None)
        date_info = {"date_string": date_str} if date_str is not None else {}
        chat_ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=False, **date_info
        )
        # all except last response are in-context examples
        wrapped_prompt = tokenizer.apply_chat_template(
            chat[:-1], tokenize=False, add_generation_prompt=True, **date_info
        )
        prompt_ids = tokenizer.apply_chat_template(
            chat[:-1], tokenize=True, add_generation_prompt=True, **date_info
        )
    else:
        wrapped_prompt = ""
        system_prompt_with_special_tokens = template_config.get(
            "system_prompt_with_special_tokens", None
        )
        if system_prompt_with_special_tokens:
            wrapped_prompt += system_prompt_with_special_tokens
        # add in-context examples
        n_few_shot = len(prompt_msgs) - 1
        for i in range(n_few_shot):
            fs_prompt, fs_response = prompt_msgs[i], response_msgs[i]
            wrapped_prompt += (
                template_config["user_start_tag"]
                + fs_prompt
                + template_config["user_end_tag"]
                + template_config["asst_start_tag"]
                + fs_response
                + template_config["asst_end_tag"]
            )

        # add actual example
        final_prompt, final_response = prompt_msgs[-1], response_msgs[-1]
        wrapped_prompt += (
            template_config["user_start_tag"]
            + final_prompt
            + template_config["user_end_tag"]
            + template_config["asst_start_tag"]
        )
        chat_ids = tokenizer(
            wrapped_prompt + final_response,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

        prompt_ids = tokenizer(
            wrapped_prompt,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]

    len_matched = len(prompt_ids)

    item = {}
    if predict_with_generate:
        item["input_ids"] = prompt_ids
        labels = chat_ids  # contains the entire conversation
    else:
        item["input_ids"] = chat_ids
        labels = [IGNORE_INDEX] * len_matched + chat_ids[len_matched:]
        if len(prompt_ids) == len(chat_ids):
            # Rarely, tokenization can result in this condition being entered.
            # Say a input prompt is ABC and target output is D, tokenizer(ABCD)
            # can be [AB, CD] and tokenizer(ABC) can be [AB, C]. In this case,
            # we ignore loss on all indices in the labels. So, there is no way
            # to use this for next token prediction. Be careful while
            # interpreting results of such instances.
            logger.warning(
                "Tokenization mismatch: no valid target tokens for loss computation"
            )

    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item


def preprocess_pretraining_instance(
    tokenizer,
    prefix: str,
    text_content: str,
    max_length: int,
    predict_with_generate: bool = False,
    insert_space: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocesses a pretraining instance for training or generation.
    When in training, both the returned `input_ids` and `labels` are over the entire token sequence. `input_ids` has no padding, `labels` assigns `IGNORE_INDEX` to ignore all tokens that we don't compute loss over (i.e. the the 0th index token, all prefix tokens)
    When in generation, `input_ids` are returned only until the prefix portion. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full input token sequence.
    Args:
        tokenizer: Tokenizer to apply on text
        prefix (str): The prefix string to prepend to the content.
        text_content (str): The main text content (following the prefix) to be tokenized.
        max_length (int): Maximum text content length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.
        insert_space (bool, optional): Whether to insert a space between prefix and content.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    full_seq_ids = tokenizer(
        prefix + (" " if insert_space else "") + text_content, add_special_tokens=True
    )["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    prefix_len = len(prefix_ids)
    full_seq_ids = full_seq_ids[: prefix_len + max_length]  # manual truncation

    len_matched = prefix_len
    if len_matched == 0:  # never give loss on index 0, when prefix is empty
        len_matched = 1
    labels = [IGNORE_INDEX] * len_matched + full_seq_ids[len_matched:]
    item = {}
    if predict_with_generate:
        item["input_ids"] = prefix_ids
    else:
        item["input_ids"] = full_seq_ids
    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item


def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column("index", indexing)
    return dataset
