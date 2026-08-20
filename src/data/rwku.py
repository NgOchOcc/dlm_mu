from torch.utils.data import Dataset

from data.utils import load_hf_dataset, add_dataset_index


class RWKURawDataset(Dataset):
    """Passes RWKU rows through untokenized for the DLM utility handlers.

    The MMLU / TruthfulQA / TriviaQA metrics need the raw fields (few-shot
    ``examples``, ``mc1_targets``, ``answers`` lists) rather than a
    prompt/answer tokenization, so each item is the raw HF row plus an
    ``index``. Prompt construction and tokenization happen inside the metric
    handler (see ``evals/metrics/dlm_rwku.py``), which owns the chat template.
    """

    def __init__(self, hf_args, template_args=None, tokenizer=None, **kwargs):
        super().__init__()
        self.data = add_dataset_index(load_hf_dataset(**hf_args))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data[idx])
