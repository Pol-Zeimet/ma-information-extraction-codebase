import tempfile
import time
from abc import abstractmethod
from tqdm import tqdm
import mlflow
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from src.experiment_scripts.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.data.data_generators import DataGenerator
from src.experiments.config import Config
from src.util.plot import create_confusion_matrix, create_distance_plots, create_embeddings_plot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from tensorflow.python.ops import math_ops


class BaseExperiment:
    def __init__(self, name: str, config: Config):
        self.name = name
        self.config = config
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
            mlflow.start_run()
            self.working_dir = tmp_dir + "/"
            self._initial_log()
            self._run()
            self._final_log()
            mlflow.end_run()

    def _initial_log(self) -> None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        # mlflow.log_params(Repository.get_details())

    def _get_mlflow_run_id(self) -> str:
        return mlflow.active_run().info.run_uuid

    @abstractmethod
    def _final_log(self) -> None:
        raise NotImplementedError

    def cleanup(self) -> None:
        self.config = None
        self.working_dir = None


class Experiment(BaseExperiment):
    def __init__(self, name: str, config: Config
                 ):
        super().__init__(name, config)
        self.labels = None
        self.train_set: pd.DataFrame = None
        self.val_set: pd.DataFrame = None
        self.model = None

    def _initial_log(self) -> None:
        super()._initial_log()

    def _run(self) -> None:
        pass

    def _final_log(self) -> None:
        mlflow.log_params(self.config.to_dict())
        mlflow.log_params(self.model.get_details())
        mlflow.set_tags({
            "type": "experiment",
        })
        self.mlflow_run_id = self._get_mlflow_run_id()
        mlflow.log_artifacts(self.working_dir)

    def cleanup(self) -> None:
        super().cleanup()
        self.model = None
        self.train_set = None
        self.val_set = None

    @abstractmethod
    def _predict(self, x) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _batches_to_list(batch) -> np.ndarray():
        return np.asarray([[label
                            for prediction in batch
                            for label in prediction]])

    def _evaluate_batch(self, x, y_true_batch, batch_index=None):
        y_pred_batch, embeddings = self._predict(x)
        y_true = self._batches_to_list(y_true_batch)
        y_pred = self._batches_to_list(y_pred_batch)
        self._compute_metrics(y_true, y_pred, batch_index)

    def _evaluate(self, data_generator: DataGenerator) -> None:
        print("Start evaluation")
        start = time.time()
        x, y_true = data_generator.__getitem__(0)
        y_pred = self._predict(x)
        for iteration in tqdm(range(1, self.config.n_iter_eval)):
            x, y = data_generator.__getitem__(iteration)
            y_true = np.append(y_true, self._batches_to_list(y), axis=0)
            y_pred = np.append(y_pred, self._batches_to_list(self._predict(x)), axis=0)
        end = time.time()
        print("TIME: Finished evaluation of test set in " + str(round(end - start, 3)) + "s")
        self._compute_metrics(y_true, y_pred)

    def _compute_metrics(self, y_true, y_pred, batch_index=None):

        if self.config.one_hot:
            y_true_masked = np.ma.masked_array(y_true, mask=np.where(np.sum(y_true, axis=1) == 0, True, False))
            y_pred_masked = np.ma.masked_array(y_pred, mask=np.where(np.sum(y_true, axis=1) == 0, True, False))
            y_true = y_true_masked[~y_true_masked.mask].data
            y_pred = y_pred_masked[~y_pred_masked.mask].data
        else:
            y_true_masked = np.ma.masked_array(y_true, mask=np.where(y_true == -1, True, False))
            y_pred_masked = np.ma.masked_array(y_pred, mask=np.where(y_true == -1 == 0, True, False))
            y_true = y_true_masked[~y_true_masked.mask].data
            y_pred = y_pred_masked[~y_pred_masked.mask].data

        precision, rec, f1, sup = precision_recall_fscore_support(np.asarray(y_true),
                                                                  np.asarray(y_pred),
                                                                  average='micro')
        if self.config.num_classes == 5:
            labels = ['O', 'I-MONEY', 'I-ORG', 'I-DATE', 'I-GPE']
        else:
            labels = self.labels
        acc = accuracy_score(np.asarray(y_true), np.asarray(y_pred))

        if batch_index is None:
            mlflow.log_metric("eval_accuracy", acc)
            mlflow.log_metric("eval_f1", f1)
            mlflow.log_metric("eval_recall", rec)
            mlflow.log_metric("eval_precision", precision)
        else:
            mlflow.log_metric("final_accuracy", acc)
            mlflow.log_metric("final_f1", f1)
            mlflow.log_metric("final_recall", rec)
            mlflow.log_metric("final_precision", precision)

        y_pred = [self.labels[idx] for idx in y_pred]
        y_true = [self.labels[idx] for idx in y_true]
        create_confusion_matrix(self.working_dir, labels, batch_index, np.asarray(y_true), np.asarray(y_pred))
        print("Latest f1: {}\nprecision: {}\nrecall: {}".format(f1, precision, rec))

    def _evaluate_embeddings(self, inputs_batch,
                             true_labels_batch,
                             tokens_batch,
                             positions_batch,
                             iteration):
        if iteration != 0:
            predictions_batch, embeddings_batch, masks_batch = self.model.predict(inputs_batch)
        else:
            embeddings_batch = inputs_batch[0]
            masks_batch = np.any(math_ops.not_equal(inputs_batch[0], 0), axis=-1)

        if self.config.num_classes == 5:
            labels = ['O', 'I-MONEY', 'I-ORG', 'I-DATE', 'I-GPE']
        else:
            # wir gruppieren die B, I und L Label zusammen
            labels = ['O',
                      'I-MONEY', 'I-MONEY', 'I-MONEY',
                      'I-ORG', 'I-ORG', 'I-ORG',
                      'I-DATE', 'I-DATE', 'I-DATE',
                      'I-GPE', 'I-GPE', 'I-GPE']

        embeddings_list = []
        label_list = []
        for e, l, m in zip(embeddings_batch, true_labels_batch, masks_batch):
            cutoff = np.argwhere(m == False)[0][0]
            embeddings_list.append(e[:cutoff])
            l = l[:cutoff]
            l = [labels[np.where(one_hot == 1)[0][0]] for one_hot in l]
            label_list.append(l)

        embeddings = np.concatenate(embeddings_list)
        labels = np.concatenate(label_list)
        tokens = np.concatenate(tokens_batch)
        # positions = np.concatenate(positions_batch)

        print("Run PCA 70 ...")
        pca_70 = PCA(n_components=70)
        pca_result_70 = pca_70.fit_transform(embeddings)

        print("Run T-SNE...")
        transformed = TSNE(n_components=2, verbose=1, perplexity=50).fit_transform(pca_result_70)

        df = pd.DataFrame()
        df['tsne-2d-one'] = transformed[:, 0]
        df['tsne-2d-two'] = transformed[:, 1]
        df['label'] = labels
        df['token'] = tokens

        df = df.sort_values(['label'])

        print(f"Create embeddings plot: ...")
        create_embeddings_plot(self.working_dir, iteration, df)

        print(f"Create distance plot: ...")
        create_distance_plots(self.working_dir, df, embeddings, iteration)
