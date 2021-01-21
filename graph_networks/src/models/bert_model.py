import os
from typing import Dict, Any
from src.experiments.config import Config
from src.models.model_architectures.bert_architecture import Bert
import numpy as np
from transformers import Trainer, \
    DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, \
    PreTrainedTokenizerBase

from transformers.integrations import MLflowCallback
from src.util.metrics.baseline_metrics import accuracy_score, recall_score, precision_score, f1_score
import mlflow


class BertConfig(Config):
    def __init__(self,
                 model_dir: str,
                 model_id: str,
                 model_type: str,
                 label_list: [str],
                 num_classes: int,
                 train_f: str,
                 test_f: str,
                 validate_f: str,
                 overwrite_output_dir: bool = False,
                 logging: bool = True):
        super().__init__(logging=logging,
                         num_classes=num_classes)
        self.model_dir = model_dir
        self.model_id = model_id
        self.model_type = model_type
        self.label_list = label_list
        self.train_f = train_f
        self.test_f = test_f
        self.validate_f = validate_f
        self.overwrite_output_dir = overwrite_output_dir


class BertModel:
    def __init__(self, config: BertConfig, training_args, data_args, model_args):
        self.config = config
        self.logging = config.logging

        self.label_list = config.label_list
        self.model_id = config.model_id
        self.model_type = config.model_type

        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args

        self.model: AutoModelForTokenClassification = None
        self.tokenizer: PreTrainedTokenizerBase = None
        self.trainer: Trainer = None
        self.trainer_state = None
        self._create_model_architecture()

    def get_details(self) -> Dict[str, Any]:
        return {
            "model": "Bert",
            'number of classes': self.config.num_classes,
            "model_base": self.model_args.model_name_or_path,
            "config name": self.model_args.config_name,
            "tokenizer name": self.model_args.tokenizer_name
        }

    def _create_model_architecture(self) -> None:
        self.tokenizer, self.model = Bert.create(self.data_args, self.model_args, self.config.num_classes)

    def train(self, data):
        self.trainer_state = 'train'
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=data['train'],
            eval_dataset=data['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )
        self.trainer.remove_callback(MLflowCallback)

        self.trainer.train(self.model_args.model_name_or_path
                           if os.path.isdir(self.model_args.model_name_or_path)
                           else None)

    def evaluate(self):
        self.trainer_state = 'evaluate'
        return self.trainer.evaluate()

    def predict(self, datasets):
        self.trainer_state = 'test'
        test_dataset = datasets["test"]
        predictions, labels, metrics = self.trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)
        return predictions, labels, metrics

    def save_weights(self, path: str) -> None:
        self.trainer.save_model(path)
        print("Model saved in " + path)

    def _compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        #todo confusion matrix for baseline pls
        acc = accuracy_score(true_labels, true_predictions)
        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)

        mlflow.log_metric(self.trainer_state + "_accuracy", acc)
        mlflow.log_metric(self.trainer_state + "_f1", f1)
        mlflow.log_metric(self.trainer_state + "_recall", recall)
        mlflow.log_metric(self.trainer_state + "_precision", precision)

        return {
            "accuracy_score": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def tokenize_and_align_labels(self, examples, **kwargs):
        text_column_name = kwargs['text_column_name']
        label_column_name = kwargs['label_column_name']
        label_to_id = kwargs['label_to_id']

        padding = "max_length" if self.data_args.pad_to_max_length else False
        tokenized_inputs = self.tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            return_offsets_mapping=True,
        )

        offset_mappings = tokenized_inputs.pop("offset_mapping")
        labels = []
        for label, offset_mapping in zip(examples[label_column_name], offset_mappings):
            label_index = 0
            current_label = -100
            label_ids = []
            for offset in offset_mapping:
                # We set the label for the first token of each word. Special characters will have an offset of (0, 0)
                # so the test ignores them.
                if offset[0] == 0 and offset[1] != 0:
                    current_label = label_to_id[label[label_index]]
                    label_index += 1
                    label_ids.append(current_label)
                # For special tokens, we set the label to -100 so it's automatically ignored in the loss function.
                elif offset[0] == 0 and offset[1] == 0:
                    label_ids.append(-100)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(current_label if self.data_args.label_all_tokens else -100)
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
