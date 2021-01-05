import os
import tempfile
import time
from abc import abstractmethod

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from src.experiments.experiments_impls.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.data.data_generators import DataGenerator, DataGeneratorReducedLabels
from src.data.batch_generator import BatchGenerator
from src.data.datasets import Dataset, DatasetWithHoldoutSet
from src.experiments.config import Config
from src.mlflow_logging import Repository
#from src.plot import create_confusion_matrix


class BaseExperiment:
    def __init__(self, name: str, config: Config, dataset: Dataset):
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
    def __init__(self, name: str, model, config: Config, dataset: Dataset, generator: bool,
                 data_src: str,
                 label_src: str,
                 labels: list,
                 ):
        super().__init__(name, config, dataset)
        self.model = model
        self.train_set: pd.DataFrame = None
        self.val_set: pd.DataFrame = None
        self.batch_generator_val: BatchGenerator = None
        self.data_generator: DataGenerator = None
        self.data_src = data_src
        self.label_src = label_src
        self.labels = labels

    def _initial_log(self) -> None:
        super()._initial_log()
        mlflow.log_params(self.dataset.get_details([self.config.col_label]))
        mlflow.log_params(self.model.get_details())
        mlflow.set_tags({
            "type": "experiment",
            "evaluation": self.config.eval_type
                         })


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

    def evaluate(self, data_generator: DataGenerator) -> None:
        print("Start evaluation")
        start = time.time()
        y_pred, y_true = [], []
        for iteration in range(self.config.n_iter_eval):
            x, y = data_generator.__getitem__(iteration)
            y_true += y
            y_pred += self.predict(x)
        end = time.time()
        np.append()

        print("TIME: Finished evaluation of test set in " + str(round(end - start, 3)) + "s")
        precision, rec, f1, sup = precision_recall_fscore_support(np.asarray(y_true),
                                                                  np.asarray(y_pred),
                                                                  average='micro')
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("precision", precision)

        #create_confusion_matrix(self.working_dir, label_support, i, test_y_true, y_pred)

        print("Latest f1: {}\nprecision: {}\nrecall: {}".format(f1, prec, rec))
