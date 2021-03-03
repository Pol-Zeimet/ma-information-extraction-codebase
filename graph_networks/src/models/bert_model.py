import os
from typing import Dict, Any
from src.experiments.config import Config
from src.models.model_architectures.bert_architecture import Bert
from src.util.metrics.baseline_metrics import accuracy_score, recall_score, precision_score, f1_score
from src.util.metrics.levenshtein import postprocess_tokens
from src.util.plot import create_confusion_matrix
import numpy as np
from transformers import Trainer, \
    DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, \
    PreTrainedTokenizerBase
from transformers.integrations import MLflowCallback
import mlflow
from statistics import mean
from Levenshtein import distance as levenshtein_distance


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
                         num_classes=num_classes,
                         model_id=model_id)
        self.model_dir = model_dir
        self.model_type = model_type
        self.label_list = label_list
        self.train_f = train_f
        self.test_f = test_f
        self.validate_f = validate_f
        self.overwrite_output_dir = overwrite_output_dir


class BertModel:
    def __init__(self, config: BertConfig, training_args, data_args, model_args, working_dir: str = None):
        self.config = config
        self.logging = config.logging
        self.working_dir = working_dir

        self.label_list = self._get_label_list(config.label_list)
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

    @staticmethod
    def _get_label_list(given_labels):
        unique_labels = set()
        for label in given_labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

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

    def evaluate(self, data):
        self.trainer_state = 'eval'
        results, metrics = self.predict(data['validation'])
        return self.trainer.evaluate(), self._compute_levenshtein(results, data['validation']['ner_tags'],
                                                                  data['validation']['tokens'])

    def predict(self, dataset):
        padding = "max_length" if self.data_args.pad_to_max_length else False
        predictions, labels, metrics = self.trainer.predict(dataset)
        predictions = np.argmax(predictions, axis=2)
        cleaned_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = []
        for sequence, prediction in zip(dataset['tokens'], cleaned_predictions):
            tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(sequence,
                                                                                         padding=padding,
                                                                                         truncation=True,
                                                                                         is_split_into_words=True)))
            results.append(
                [(token, prediction) for token, prediction in zip(tokens, prediction)])
        return results, metrics

    def save_weights(self, path: str) -> None:
        self.trainer.save_model(path)
        print("Model saved in " + path)

    def set_working_dir(self, path):
        self.working_dir = path

    def _compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        # Remove ignored index (special tokens)
        predictions_cleaned = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        labels_cleaned = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        if (self.trainer.state.epoch is not None) & (self.trainer.state.global_step is not None):
            epoch = self.trainer.state.epoch
            step = self.trainer.state.global_step
        else:
            epoch = 'final'
            step = None
        create_confusion_matrix(self.working_dir,
                                self.label_list,
                                epoch=epoch,
                                step=step,
                                y_pred=np.asarray([item for sublist in predictions_cleaned for item in sublist]),
                                y_true=np.asarray([item for sublist in labels_cleaned for item in sublist]))
        acc = accuracy_score(labels_cleaned, predictions_cleaned)
        precision = precision_score(labels_cleaned, predictions_cleaned)
        recall = recall_score(labels_cleaned, predictions_cleaned)
        f1 = f1_score(labels_cleaned, predictions_cleaned)
        macro_acc = accuracy_score(labels_cleaned, predictions_cleaned)
        macro_precision = precision_score(labels_cleaned, predictions_cleaned, average='macro')
        macro_recall = recall_score(labels_cleaned, predictions_cleaned, average='macro')
        macro_f1 = f1_score(labels_cleaned, predictions_cleaned, average='macro')

        mlflow.log_metric(self.trainer_state + "_accuracy", acc)
        mlflow.log_metric(self.trainer_state + "_f1", f1)
        mlflow.log_metric(self.trainer_state + "_recall", recall)
        mlflow.log_metric(self.trainer_state + "_precision", precision)
        mlflow.log_metric(self.trainer_state + "macro_accuracy", macro_acc)
        mlflow.log_metric(self.trainer_state + "macro_f1", macro_f1)
        mlflow.log_metric(self.trainer_state + "macro_recall", macro_recall)
        mlflow.log_metric(self.trainer_state + "macro_precision", macro_precision)

        return {
            "accuracy_score": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_accuracy_score": macro_acc,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1
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

    def _compute_levenshtein(self, y_pred, y_true, tokens):
        addr_distances = []
        org_distances = []
        total_distances = []
        date_distances = []
        addr_coverages = []
        org_coverages = []
        total_coverages = []
        date_coverages = []

        for pred, true, token in zip(y_pred, y_true, tokens):
            org, total, addr, date = [], [], [], []
            for result in pred:
                if result[-1] != 'O':
                    tag = result[-1].split('-')[-1]
                    if tag == 'ORG':
                        org.append(result[0])
                    elif tag == 'GPE':
                        addr.append(result[0])
                    elif tag == 'DATE':
                        date.append(result[0])
                    elif tag == 'MONEY':
                        total.append(result[0])

            true_org, true_total, true_addr, true_date = [], [], [], []

            for pair in zip(token, true):
                text, label = pair[0], pair[1]
                if label != 'O':
                    tag = label.split('-')[-1]
                    if tag == 'ORG':
                        true_org.append(text)
                    elif tag == 'GPE':
                        true_addr.append(text)
                    elif tag == 'DATE':
                        true_date.append(text)
                    elif tag == 'MONEY':
                        true_total.append(text)


            true_addr_text = postprocess_tokens(true_addr, 'addr')
            true_org_text = postprocess_tokens(true_org, 'org')
            true_total_text = postprocess_tokens(true_total, 'total')
            true_date_text = postprocess_tokens(true_date, 'date')

            addr_text = postprocess_tokens(addr, 'addr')
            org_text = postprocess_tokens(org, 'org')
            total_text = postprocess_tokens(total, 'total')
            date_text = postprocess_tokens(date, 'date')

            if len(true_org_text) > 0:
                org_text_distance = levenshtein_distance(org_text, true_org_text)
                org_distances.append(org_text_distance)
                if len(true_org_text) > org_text_distance:
                    org_coverages.append((len(true_org_text) - org_text_distance) / len(true_org_text) * 100)
                else:
                    org_coverages.append((org_text_distance - len(true_org_text)) / len(true_org_text) * 100)

            if len(true_addr_text) > 0:
                addr_text_distance = levenshtein_distance(addr_text, true_addr_text)
                addr_distances.append(addr_text_distance)
                if len(true_addr_text) > addr_text_distance:
                    addr_coverages.append((len(true_addr_text) - addr_text_distance) / len(true_addr_text) * 100)
                else:
                    addr_coverages.append((addr_text_distance - len(true_addr_text)) / len(true_addr_text) * 100)
            if len(true_date_text) > 0:
                date_text_distance = levenshtein_distance(date_text, true_date_text)
                date_distances.append(date_text_distance)
                if len(true_date_text) > date_text_distance:
                    date_coverages.append((len(true_date_text) - date_text_distance) / len(true_date_text) * 100)
                else:
                    date_coverages.append((date_text_distance - len(true_date_text)) / len(true_date_text) * 100)
            if len(true_total_text) > 0:
                total_text_distance = levenshtein_distance(total_text, true_total_text)
                total_distances.append(total_text_distance)
                if len(true_total_text) > total_text_distance:
                    total_coverages.append((len(true_total_text) - total_text_distance) / len(true_total_text) * 100)
                else:
                    total_coverages.append((total_text_distance - len(true_total_text)) / len(true_total_text) * 100)

        all_distance_means = []
        all_coverage_means = []

        if len(addr_distances) > 0:
            mean_addr_distances = mean(addr_distances)
            mean_addr_coverages = mean(addr_coverages)
            all_distance_means.append(mean_addr_distances)
            all_coverage_means.append(mean_addr_coverages)

            mlflow.log_metric(f'eval_mean_addr_distances', mean_addr_distances)
            mlflow.log_metric(f'eval_mean_addr_coverage', mean_addr_coverages)
        else:
            mean_addr_distances = -1000
            mean_addr_coverages = -1000

        if len(org_distances) > 0:
            mean_org_distances = mean(org_distances)
            mean_org_coverages = mean(org_coverages)
            all_distance_means.append(mean_org_distances)
            all_coverage_means.append(mean_org_coverages)

            mlflow.log_metric(f'eval_mean_org_distances', mean_org_distances)
            mlflow.log_metric(f'eval_mean_org_coverage', mean_org_coverages)
        else:
            mean_org_distances = -1000
            mean_org_coverages = -1000

        if len(total_distances) > 0:
            mean_total_distances = mean(total_distances)
            mean_total_coverages = mean(total_coverages)
            all_distance_means.append(mean_total_distances)
            all_coverage_means.append(mean_total_coverages)

            mlflow.log_metric(f'eval_mean_total_distances', mean_total_distances)
            mlflow.log_metric(f'eval_mean_total_coverage', mean_total_coverages)
        else:
            mean_total_distances = -1000
            mean_total_coverages = -1000

        if len(date_distances) > 0:
            mean_date_distances = mean(date_distances)
            mean_date_coverages = mean(date_coverages)
            all_distance_means.append(mean_date_distances)
            all_coverage_means.append(mean_date_coverages)

            mlflow.log_metric(f'eval_mean_date_distances', mean_date_distances)
            mlflow.log_metric(f'eval_mean_date_coverage', mean_date_coverages)
        else:
            mean_date_distances = -1000
            mean_date_coverages = -1000

        if len(all_distance_means) > 0:
            total_distance_mean = mean(all_distance_means)
            mean_coverage = mean(all_coverage_means)
            mlflow.log_metric(f'eval_total_mean', total_distance_mean)
            mlflow.log_metric(f'eval_mean_coverage', mean_coverage)
        else:
            total_distance_mean = -1000
            mean_coverage = -1000

        return {
            'eval_mean_addr_distances': mean_addr_distances,
            'eval_mean_org_distances': mean_org_distances,
            'eval_mean_total_distances': mean_total_distances,
            'eval_mean_date_distances': mean_date_distances,
            'eval_total_mean': total_distance_mean,
            'eval_mean_addr_coverage': mean_addr_coverages,
            'eval_mean_org_coverage': mean_org_coverages,
            'eval_mean_total_coverage': mean_total_coverages,
            'eval_mean_date_coverage': mean_date_coverages,
            'eval_mean_coverage': mean_coverage
        }
