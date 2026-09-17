from evals.base import Evaluator


class RWKUEvaluator(Evaluator):
    """RWKU knowledge-unlearning benchmark (jinzhuoran/RWKU).

    Forget (level1/2/3) and neighbor (level1/2) probes are RougeL-recall on
    generations (reuses ``rouge_diffusion``); utility is MMLU accuracy,
    TruthfulQA MC1, and TriviaQA EM via the DLM utility handlers.
    """

    def __init__(self, eval_cfg, **kwargs):
        super().__init__("RWKU", eval_cfg, **kwargs)
