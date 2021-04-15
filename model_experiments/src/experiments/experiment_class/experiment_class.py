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
from src.util.metrics.levenshtein import compute_levenshtein
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os, sys


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
            model_type = self.config.model_id.split('_')[2]
            if model_type == 'Bert' or model_type == 'LayoutLM':
                run_name = model_type + '_' + str(self.config.num_classes)
            else:
                run_name = model_type + '_' + str(self.config.n_folds) \
                                      + '_' + str(self.config.num_classes) \
                                      + '_' + str(self.config.learning_rate)
            mlflow.start_run(run_name=run_name)
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
        self.pca70 = None

    def _initial_log(self) -> None:
        super()._initial_log()

    def _run(self) -> None:
        pass

    def _final_log(self) -> None:
        print("starting final log")
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

    def _evaluate_batch(self, x, y_true_batch, tokens, positions, epoch, step):
        y_pred_batch, embeddings, masks = self._predict(x)
        predictions, truth = [], []
        for pred, true, mask_length in zip(y_pred_batch, y_true_batch, masks):
            predictions.append(pred[:mask_length])
            if self.config.one_hot:
                truth.append([np.argwhere(one_hot == 1)[0][0] for one_hot in true[:mask_length]])
            else:
                truth.append(true[:mask_length])
        y_true = np.concatenate(truth)
        y_pred = np.concatenate(predictions)
        positions = np.concatenate(positions)
        self._compute_metrics(y_true, y_pred, str(epoch), str(step))
        compute_levenshtein(predictions, truth, tokens, True)

    def _evaluate(self, data_generator: DataGenerator, n_eval_steps, epoch) -> None:
        print("Start evaluation")
        start = time.time()
        y_true, y_pred = [],[]
        y_true_batched, y_pred_batched, tokens_batched = [], [], []
        print(f"Evaluating Epoch {epoch}")
        for iteration in tqdm(range(0, n_eval_steps)):
            x, y = data_generator.__getitem__(iteration)
            tokens, positions = data_generator.get_tokens_and_positions(iteration)
            predictions, embeddings, masks = self._predict(x)
            predictions_list, truth = [], []
            for pred, true, mask_length in zip(predictions, y, masks):
                predictions_list.append(pred[:mask_length])
                if self.config.one_hot:
                    truth.append([np.argwhere(one_hot == 1)[0][0] for one_hot in true[:mask_length]])
                else:
                    truth.append(true[:mask_length])
            tokens_batched.append(tokens)
            y_true_batched.append(truth)
            y_pred_batched.append(predictions_list)
            y_true.append(np.concatenate(truth))
            y_pred.append(np.concatenate(predictions_list))
        end = time.time()

        tokens_batched = [token_list for batch in tokens_batched for token_list in batch]
        y_true_batched = [y_true_list for batch in y_true_batched for y_true_list in batch]
        y_pred_batched = [y_pred_list for batch in y_pred_batched for y_pred_list in batch]
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        print("TIME: Finished evaluation of test set in " + str(round(end - start, 3)) + "s")
        self._compute_metrics(y_true, y_pred, str(epoch), None)
        compute_levenshtein(y_pred_batched, y_true_batched, tokens_batched, False)

    def _compute_metrics(self, y_true, y_pred, epoch: str, step: str):

        precision, rec, f1, sup = precision_recall_fscore_support(np.asarray(y_true),
                                                                  np.asarray(y_pred),
                                                                  average='micro')

        macro_precision, macro_rec, macro_f1, macro_sup = precision_recall_fscore_support(np.asarray(y_true),
                                                                  np.asarray(y_pred),
                                                                  average='macro')
        if self.config.num_classes == 5:
            labels = ['O', 'I-MONEY', 'I-ORG', 'I-DATE', 'I-GPE']
        else:
            labels = self.labels
        acc = accuracy_score(np.asarray(y_true), np.asarray(y_pred))

        if step is None:
            mlflow.log_metric("train_accuracy", acc)
            mlflow.log_metric("train_f1", f1)
            mlflow.log_metric("train_recall", rec)
            mlflow.log_metric("train_precision", precision)
            mlflow.log_metric("train_macro_accuracy", acc)
            mlflow.log_metric("train_macro_f1", f1)
            mlflow.log_metric("train_macro_recall", rec)
            mlflow.log_metric("train_macro_precision", precision)
        else:
            mlflow.log_metric("eval_accuracy", acc)
            mlflow.log_metric("eval_f1", f1)
            mlflow.log_metric("eval_recall", rec)
            mlflow.log_metric("eval_precision", precision)
            mlflow.log_metric("eval_macro_accuracy", acc)
            mlflow.log_metric("eval_macro_f1", f1)
            mlflow.log_metric("eval_macro_recall", rec)
            mlflow.log_metric("eval_macro_precision", precision)

        y_pred = [labels[idx] for idx in y_pred]
        y_true = [labels[idx] for idx in y_true]
        create_confusion_matrix(self.working_dir, labels, epoch, step, np.asarray(y_true), np.asarray(y_pred))
        print("Latest f1: {}\nprecision: {}\nrecall: {}".format(f1, precision, rec))
        print("Latest macro_f1: {}\nmacro_precision: {}\nmacro_recall: {}".format(macro_f1, macro_precision, macro_rec))



    def _evaluate_embeddings(self, inputs_batch, true_labels_batch, tokens_batch, positions_batch, epoch, step):

        predictions_batch, embeddings_batch, mask_lengths = self.model.predict(inputs_batch)
        if epoch == 'init':
            embeddings_batch = inputs_batch[0]

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
        truth_list = []
        predictions_list = []
        sequence_idxs_list =[]
        for sequence_idx, (e, l, m, o) in enumerate(zip(embeddings_batch, true_labels_batch, mask_lengths, predictions_batch)):
            embeddings_list.append(e[:m])
            l = l[:m]
            o = o[:m]
            sequence_idxs_list.append(np.full(m,sequence_idx))
            if self.config.one_hot:
                tr = [np.argwhere(one_hot == 1)[0][0] for one_hot in l]
                l = [labels[idx] for idx in tr]
            else:
                tr = l
                l = [labels[idx] for idx in tr]

            truth_list.append(tr)
            label_list.append(l)
            predictions_list.append(o)

        sequence_idxs = np.concatenate(sequence_idxs_list)
        embeddings = np.concatenate(embeddings_list)
        labels = np.concatenate(label_list)
        truth = np.concatenate(truth_list)
        tokens = np.concatenate(tokens_batch)
        predictions = np.concatenate(predictions_list)
        truth_matching = np.where(truth == predictions, 'correct', 'incorrect')
        positions = np.concatenate(positions_batch)

        if epoch == 'init':
            self.pca70 = PCA(n_components=70)
            transformed = self.pca70.fit_transform(embeddings)
        else:
            transformed = self.pca70.transform(embeddings)
        df = pd.DataFrame()
        df['sequence'] = sequence_idxs
        df['token'] = tokens
        df['label'] = labels
        df[f"position"] = positions.tolist()
        df['truth_matching'] = truth_matching
        df['pca'] = transformed.tolist()
        if step is None:
            df.to_pickle(os.path.join(self.working_dir, f"umap_embeddings_{epoch}"))
        else:
            df.to_pickle(os.path.join(self.working_dir, f"umap_embeddings_{epoch}_{step}"))
