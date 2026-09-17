from data.qa import QADataset
from data.utils import load_hf_dataset, add_dataset_index

_LETTERS = ["A", "B", "C", "D"]


class WMDPMCQDataset(QADataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        max_length=512,
        predict_with_generate=False,
    ):
        raw = load_hf_dataset(**hf_args)
        raw = add_dataset_index(raw)

        def _build(example):
            question = example["question"].strip()
            choices = example["choices"]
            lettered = "\n".join(
                f"{letter}. {choice}" for letter, choice in zip(_LETTERS, choices)
            )
            example["mcq_prompt"] = f"{question}\n{lettered}\nAnswer:"
            example["mcq_options"] = list(_LETTERS)
            return example

        self.data = raw.map(_build)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_args = template_args
        self.question_key = "mcq_prompt"
        self.answer_key = "mcq_options"
        self.predict_with_generate = predict_with_generate
        self.fs_data = None

    def correct_index(self, idx):
        return int(self.data[idx]["answer"])
