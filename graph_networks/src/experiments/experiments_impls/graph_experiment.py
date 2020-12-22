from collections import defaultdict
from typing import Dict, List, Optional

import mlflow
import numpy as np
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm

from src.main.data.batch import BatchGenerator
from src.main.data.datasets import Dataset
from src.main.experiments.config import Config
from src.main.experiments.experiment import Experiment
from src.main.models.graph_model import GraphNetConfig, GraphModel
from src.main.plot import create_similarity_plots


class GraphExperiment(Experiment):
    def __init__(self, config: GraphNetConfig, dataset: Dataset):
        super().__init__("grapg_net", None, config, dataset)
        self.batch_generator_train: BatchGenerator = None
        self.similarity_store: Dict = None

    def setup(self):
        super().setup()
        self.batch_generator_train = BatchGenerator(self.train_set, None, self.config.col_label, self.config.col_text_train, None)
        self.model = GraphModel(self.config)


    def _run(self) -> None:
        super()._run()
        self.train()
        print("Done with train")
        self.evaluate(None, self.batch_generator_val)
        print("Done evaluation on test set")

    def _run_holdout(self):
        mlflow.set_tag("related_to_run", self.mlflow_run_id)

        self.similarity_store = defaultdict(lambda: defaultdict(list))
        self.evaluate(None, self.batch_generator_holdout)
        print("Done evaluation on holdout set")

    def cleanup(self):
        super().cleanup()
        self.batch_generator_train = None
        self.similarity_store = None

    def _final_log(self):
        if self.config.plot_distances:
            create_similarity_plots(self.working_dir, self.similarity_store)
            mlflow.log_artifacts(self.working_dir)

    def train(self) -> None:
        print("Training with {} iterations, batch size={}".format(str(self.config.n_iter_train), str(self.config.batchsize)))

        with tf.Session() as session:
            K.set_session(session)
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

            for i in tqdm(range(1, self.config.n_iter_train)):
                (inputs, targets) = self.batch_generator_train.get_batch(self.config.batchsize)
                loss = self.model.train(inputs, targets)
                mlflow.log_metric("loss", loss[0])
                mlflow.log_metric("accuracy", loss[1])

            print("Training done.")

            session.close()
            print("Session closed")

    def evaluate(self, _, batch_generator: BatchGenerator) -> None:
        with tf.Session() as session:
            K.set_session(session)
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

            super().evaluate(session, batch_generator)

            session.close()

    def predict(self, pairs, targets, target_labels, session)  -> np.ndarray:
        prediction = self.model.predict(pairs, session, target_labels)

        if self.config.plot_distances:
            self._store_similarities(targets, target_labels)

        return prediction

    def _store_similarities(self, targets: np.ndarray, target_labels: List[str]) -> None:
        label = target_labels[list(targets).index(1.0)]
        for target_label, sim in zip(target_labels, self.model.similarity_store):
            self.similarity_store[label][target_label].append(sim[0])

    def embed_sample(self, sample_text: str, compare_to: List[str]) -> np.ndarray:
        pairs = np.vstack((np.asarray([sample_text]*len(compare_to)), np.asarray(compare_to)))

        with tf.Session() as session:
            K.set_session(session)
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

            results = self.model.embed_sample(pairs)

            session.close()

        return results
