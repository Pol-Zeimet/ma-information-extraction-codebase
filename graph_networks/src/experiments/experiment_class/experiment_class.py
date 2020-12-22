import os
import tempfile
import time
from abc import abstractmethod

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from experiments.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.main.data.batch import BatchGenerator
from src.main.data.datasets import Dataset, DatasetWithHoldoutSet
from src.main.experiments.config import Config, BaseConfig
from src.main.mlflow_logging import Repository
from src.main.plot import create_confusion_matrix


class BaseExperiment:
    def __init__(self, name: str, config: BaseConfig, dataset: Dataset):
        self.name = name
        self.config = config
        self.dataset = dataset
        self.working_dir: str = None
        self.mlflow_run_id: str = None

    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _run_holdout(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.working_dir = tmp_dir + "/"

            self._initial_log()
            self._run()
            self._final_log()
            mlflow.end_run()

        if isinstance(self.dataset, DatasetWithHoldoutSet):
            with tempfile.TemporaryDirectory() as tmp_dir:
                self.working_dir = tmp_dir + "/"

                self._initial_log()
                self._run_holdout()
                self._final_log()
                mlflow.end_run()

    def _initial_log(self) -> None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        mlflow.log_params(Repository.get_details())
        mlflow.log_params(self.config.to_dict())
        self.mlflow_run_id = self._get_mlflow_run_id()
        if isinstance(self.dataset, DatasetWithHoldoutSet):
            mlflow.set_tag("holdout_set", True)

    def _get_mlflow_run_id(self) -> str:
        return mlflow.active_run().info.run_uuid

    @abstractmethod
    def _final_log(self) -> None:
        raise NotImplementedError


    def cleanup(self) -> None:
        self.config = None
        self.dataset = None
        self.working_dir = None


class Experiment(BaseExperiment):
    def __init__(self, name: str, model, config: Config, dataset: Dataset):
        super().__init__(name, config, dataset)
        self.model = model
        self.train_set: pd.DataFrame = None
        self.val_set: pd.DataFrame = None
        self.batch_generator_val: BatchGenerator = None

    def _initial_log(self) -> None:
        super()._initial_log()
        mlflow.log_params(self.dataset.get_details([self.config.col_label]))
        mlflow.log_params(self.model.get_details())
        mlflow.set_tags({
            "type": "experiment",
            "evaluation": self.config.eval_type
                         })

    def setup(self):
        self.train_set, _, self.val_set = self.dataset.load()
        self.batch_generator_val = BatchGenerator(None, self.val_set, self.config.col_label, None, self.config.col_text_val)

        if isinstance(self.dataset, DatasetWithHoldoutSet):
            self.holdout_set = self.dataset.get_holdout_set()
            self.batch_generator_holdout = BatchGenerator(None, self.holdout_set, self.config.col_label, None, self.config.col_text_val)

    def _run(self) -> None:
        pass

    def _final_log(self) -> None:
        pass

    def cleanup(self) -> None:
        super().cleanup()
        self.model = None
        self.train_set = None
        self.val_set = None
        self.batch_generator_val = None

    @abstractmethod
    def predict(self) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, session, batch_generator: BatchGenerator) -> None:
        print("Start evaluation")
        X = batch_generator.validation
        max_n = len(X.keys())

        with tqdm(total=self.config.n_iter_eval*sum(len(v) for v in X.values())) as pbar:
            for i in range(self.config.n_iter_eval):
                label_support = list(X)
                start = time.time()
                test_y_true = []
                y_pred = []

                for label in label_support:
                    for sen in X[label]:
                        test_y_true.append(label)

                        if self.config.eval_type == "oneshot":
                            pairs, targets, target_labels = batch_generator.get_oneshot_pairs(X, max_n, label, sen)
                        if self.config.eval_type == "all":
                            pairs, targets, target_labels = batch_generator.get_one_against_all_pairs(X, label, sen)
                        if self.config.eval_type == "fewshot":
                            pairs, targets, target_labels = batch_generator.get_kshot_pairs(X, self.config.k_per_class, label, sen)

                        y_pred.append(self.predict(pairs, targets, target_labels, session))
                        pbar.update(1)

                end = time.time()
                print("TIME: Finished evaluation of test set in " + str(round(end - start, 3)) + "s")

                test_y_true = np.asarray(test_y_true)

                prec, rec, f1, sup = precision_recall_fscore_support(test_y_true, np.asarray(y_pred), average='micro')
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("precision", prec)

                create_confusion_matrix(self.working_dir, label_support, i, test_y_true, y_pred)

        print("Latest f1: {}\nprecision: {}\nrecall: {}".format(f1, prec, rec))
