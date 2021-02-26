import os
from typing import Dict, Any
from src.experiments.config import Config
from src.models.model_architectures.layoutLM_architecture import LayoutLM
from src.util.metrics.levenshtein import postprocess_tokens
from src.util.plot import create_confusion_matrix
from transformers import AutoModelForTokenClassification, \
    PreTrainedTokenizerBase
import mlflow
from statistics import mean
from Levenshtein import distance as levenshtein_distance
from torch.nn import CrossEntropyLoss
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import numpy as np
import torch


class LayoutLMConfig(Config):
    def __init__(self,
                 model_id: str,
                 model_type: str,
                 num_classes: int,
                 n_train_epochs: int,
                 data_dir: str,
                 learning_rate: float = 5e-5,
                 max_seq_length: int=512,
                 model_name_or_path: str = "microsoft/layoutlm-base-uncased"):
        super().__init__(num_classes=num_classes,
                         model_id=model_id)
        self.n_train_epochs = n_train_epochs

        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.learning_rate = learning_rate


class LayoutLMModel:
    def __init__(self, config: LayoutLMConfig, working_dir: str = None):
        self.config = config
        self.working_dir = working_dir

        self.label_list = self._get_label_list(os.path.join(config.data_dir, "labels.txt"))
        self.label_map = {i: label for i, label in enumerate(self.label_list)}
        self.model_id = config.model_id
        self.model_type = config.model_type

        self.model: AutoModelForTokenClassification = None
        self.tokenizer: PreTrainedTokenizerBase = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._create_model_architecture()
        self.global_step = 0
        self.global_epoch = 0
        self.state = None

    @staticmethod
    def _get_label_list(path):
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels

    def get_details(self) -> Dict[str, Any]:
        return {
            "model": "LayouLM",
            'number of classes': self.config.num_classes,
            "model_base": self.config.model_name_or_path,
        }

    def _create_model_architecture(self) -> None:
        self.tokenizer, self.model = LayoutLM.create(self.config)
        self.model.to(self.device)

    def train(self, data):
        self.model.train()
        return self.predict(data)

    def predict(self, data):
        return self.model(input_ids=data['input_ids'],
                   bbox=data['bbox'],
                   attention_mask=data['attention_mask'],
                   token_type_ids=data['token_type_ids'],
                   labels=data['labels'])

    def evaluate(self, data):
        self.model.eval()
        return self.predict(data)

    def get_model_parameters(self):
        return self.model.parameters()

    def set_working_dir(self, path):
        self.working_dir = path

    def compute_metrics(self, p):
        predictions, labels = p
        epoch = str(self.global_step)
        step = str(self.global_epoch)
        if epoch == 'final':
            step = None
        create_confusion_matrix(self.working_dir,
                                self.label_list,
                                epoch=epoch,
                                step=step,
                                y_pred=np.asarray([item for sublist in predictions for item in sublist]),
                                y_true=np.asarray([item for sublist in labels for item in sublist]))
        acc = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        mlflow.log_metric(self.state + "_accuracy", acc)
        mlflow.log_metric(self.state + "_f1", f1)
        mlflow.log_metric(self.state + "_recall", recall)
        mlflow.log_metric(self.state + "_precision", precision)

        return {
            "accuracy_score": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def compute_levenshtein(self, y_pred, y_true, tokens):
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
            for text, result in zip(token, pred):
                if result != 'O':
                    tag = result.split('-')[-1]
                    if tag == 'ORG':
                        org.append(text)
                    elif tag == 'GPE':
                        addr.append(text)
                    elif tag == 'DATE':
                        date.append(text)
                    elif tag == 'MONEY':
                        total.append(text)

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
