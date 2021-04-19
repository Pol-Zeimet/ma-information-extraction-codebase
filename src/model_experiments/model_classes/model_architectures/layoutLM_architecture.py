from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    LayoutLMTokenizer,
    PreTrainedTokenizerBase,
    LayoutLMForTokenClassification
)
from model_experiments.experiments.experiment_class.experiment_class import Config


class LayoutLM:
    @staticmethod
    def create(config: Config) \
            -> (PreTrainedTokenizerBase,AutoModelForTokenClassification):
        tokenizer = LayoutLMTokenizer.from_pretrained(config.model_name_or_path)
        model = LayoutLMForTokenClassification.from_pretrained(config.model_name_or_path, num_labels=config.num_classes)
        return tokenizer, model
