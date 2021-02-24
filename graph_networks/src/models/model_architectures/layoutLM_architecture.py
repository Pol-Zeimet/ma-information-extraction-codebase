from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    LayoutLMTokenizer,
    PreTrainedTokenizerBase,
    LayoutLMForTokenClassification
)
from src.models.layoutlm_model import LayoutLMConfig

class LayoutLM:
    @staticmethod
    def create(config: LayoutLMConfig) \
            -> (PreTrainedTokenizerBase,AutoModelForTokenClassification):
        tokenizer = LayoutLMTokenizer.from_pretrained(config.model_name_or_path)
        model = LayoutLMForTokenClassification.from_pretrained(config.model_name_or_path, num_labels=config.num_classes)
        return tokenizer, model
