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
from Levenshtein import distance as levenshtein_distance
from statistics import mean

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

    def _evaluate_batch(self, x, y_true_batch, tokens, positions,epoch_index=None, batch_index=None):
        y_pred_batch, embeddings, masks = self._predict(x)
        predictions, truth = [], []
        for pred, true, mask_length in zip(y_pred_batch, y_true_batch, masks) :
            predictions.append(pred[:mask_length])
            if self.config.one_hot:
                truth.append([np.argwhere(one_hot == 1)[0][0] for one_hot in true[:mask_length]])
            else:
                truth.append(true[:mask_length])
        y_true = np.concatenate(truth)
        y_pred = np.concatenate(predictions)
        positions = np.concatenate(positions)
        self._compute_metrics(y_true, y_pred, batch_index)
        self._compute_levenshtein(truth, predictions, tokens)

    def _evaluate(self, data_generator: DataGenerator) -> None:
        print("Start evaluation")
        start = time.time()
        y_true, y_pred = [],[]
        y_true_batched, y_pred_batched, tokens_batched = [],[],[]
        for iteration in tqdm(range(0, self.config.n_eval_epochs)):
            x, y = self.data_generator_validation.__getitem__(iteration)
            tokens, positions = self.data_generator_validation.get_tokens_and_positions(iteration)
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

        tokens_batched = np.concatenate(tokens_batched)
        y_true_batched = np.concatenate(y_true_batched)
        y_pred_batched = np.concatenate(y_pred_batched)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        print("TIME: Finished evaluation of test set in " + str(round(end - start, 3)) + "s")
        self._compute_metrics(y_true, y_pred)
        self._compute_levenshtein(y_true_batched, y_pred_batched, tokens_batched)

    def _compute_metrics(self, y_true, y_pred, batch_index=None):

        precision, rec, f1, sup = precision_recall_fscore_support(np.asarray(y_true),
                                                                  np.asarray(y_pred),
                                                                  average='micro')
        if self.config.num_classes == 5:
            labels = ['O', 'I-MONEY', 'I-ORG', 'I-DATE', 'I-GPE']
        else:
            labels = self.labels
        acc = accuracy_score(np.asarray(y_true), np.asarray(y_pred))

        if batch_index is None:
            mlflow.log_metric("final_accuracy", acc)
            mlflow.log_metric("final_f1", f1)
            mlflow.log_metric("final_recall", rec)
            mlflow.log_metric("final_precision", precision)
        else:
            mlflow.log_metric("eval_accuracy", acc)
            mlflow.log_metric("eval_f1", f1)
            mlflow.log_metric("eval_recall", rec)
            mlflow.log_metric("eval_precision", precision)

        y_pred = [self.labels[idx] for idx in y_pred]
        y_true = [self.labels[idx] for idx in y_true]
        create_confusion_matrix(self.working_dir, labels, batch_index, np.asarray(y_true), np.asarray(y_pred))
        print("Latest f1: {}\nprecision: {}\nrecall: {}".format(f1, precision, rec))


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
        for e, l, m in zip(embeddings_batch, true_labels_batch, mask_lengths):
            
            embeddings_list.append(e[:m])
            l = l[:m]
            if self.config.one_hot:
                tr = [np.argwhere(one_hot == 1)[0][0] for one_hot in l]
                l = [labels[idx] for idx in tr]
            truth_list.append(tr)
            label_list.append(l)

        embeddings = np.concatenate(embeddings_list)
        labels = np.concatenate(label_list)
        truth = np.concatenate(truth_list)
        tokens = np.concatenate(tokens_batch)
        predictions = [labels[idx] for idx in np.concatenate(predictions_batch)]
        truth_matching = np.where(truth == predictions, 'correct predictions', 'incorrect predictions')
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
        df['truth_matching'] = truth_matching

        df = df.sort_values(['label'])

        print(f"Create embeddings plot: ...")
        create_embeddings_plot(self.working_dir, epoch, df)

        print(f"Create distance plot: ...")
        create_distance_plots(self.working_dir, df, embeddings, epoch)


    def _compute_levenshtein(self, y_pred, y_true, tokens):
        
        addr_distances = []
        org_distances = []
        total_distances = []
        date_distances = []

        for pred, true, token in zip(y_pred, y_true, tokens):   
            true_addr,true_org, true_total, true_date = [],[],[],[]
            addr,org,total,date = [],[],[],[]
            for idx, (pred_label, true_label) in enumerate(zip(pred,true)):
                if pred_label == 1:
                    total.append(token[idx])
                elif pred_label == 2:
                    org.append(token[idx])
                elif pred_label == 3:
                    date.append(token[idx])
                elif pred_label == 4:
                    addr.append(token[idx])
                
                if true_label == 1:
                    true_total.append(token[idx])
                elif true_label == 2:
                    true_org.append(token[idx])
                elif true_label == 3:
                    true_date.append(token[idx])
                elif true_label == 4:
                    true_addr.append(token[idx])

            true_addr_text = ' '.join(true_addr).strip()
            true_org_text = ' '.join(true_org).strip()
            true_total_text = ' '.join(true_total).strip()
            true_date_text = ' '.join(true_date).strip()

            addr_text = ' '.join(addr).replace(' ##', '').replace(' - ', '-').replace('( ', '(').replace(' )', ')').replace(' ,',',').replace(' .', '.').strip()
            org_text = ' '.join(org).replace(' ##', '').replace(' - ', '-').replace('( ', '(').replace(' )', ')').replace(' ,',',').replace(' .', '.').strip()
            total_text = ' '.join(total).replace(' ##', '').replace(' . ', '.').strip()
            date_text = ' '.join(date).replace(' ##', '').replace(' - ', '-').replace(' : ', ':').strip()


            addr_distances.append(levenshtein_distance(str.lower(org_text),str.lower(true_org_text)))
            org_distances.append(levenshtein_distance(str.lower(addr_text),str.lower(true_addr_text)))
            total_distances.append(levenshtein_distance(str.lower(date_text),str.lower(true_date_text)))
            date_distances.append(levenshtein_distance(str.lower(total_text),str.lower(true_total_text)))
        

        mean_addr_distances = mean(addr_distances)
        mean_org_distances = mean(org_distances)
        mean_total_distances = mean(total_distances)
        mean_date_distances = mean(date_distances)
        total_mean = mean_addr_distances + mean_org_distances + mean_total_distances + mean_date_distances

        if len(y_true) == self.config.batch_size:
            mlflow.log_metric('eval_mean_addr_distances', mean_addr_distances)
            mlflow.log_metric('eval_mean_org_distances', mean_org_distances)
            mlflow.log_metric('eval_mean_total_distances', mean_total_distances)
            mlflow.log_metric('eval_mean_date_distances', mean_date_distances)
            mlflow.log_metric('eval_total_mean', total_mean)
            print("Latest levenshtein:")
            print(f"mean addr distances: {mean_addr_distances}")
            print(f"mean org distances: {mean_org_distances}")
            print(f"mean total distances: {mean_total_distances}")
            print(f"mean date distances: {mean_date_distances}")
            print(f"total mean: {total_mean}")


        if len(y_true) != self.config.batch_size:
            mlflow.log_metric('final_mean_addr_distances', mean_addr_distances)
            mlflow.log_metric('final_mean_org_distances', mean_org_distances)
            mlflow.log_metric('final_mean_total_distances', mean_total_distances)
            mlflow.log_metric('final_mean_date_distances', mean_date_distances)
            mlflow.log_metric('final_total_mean', total_mean)
            print("Final levenshtein:")
            print(f"mean addr distances: {mean_addr_distances}")
            print(f"mean org distances: {mean_org_distances}")
            print(f"mean total distances: {mean_total_distances}")
            print(f"mean date distances: {mean_date_distances}")
            print(f"total mean: {total_mean}")