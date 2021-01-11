from typing import Dict
import mlflow
import numpy as np
import math
from src import DataGenerator, DataGeneratorReducedLabels
from src import Experiment
from src import GraphNetConfig, GraphModel
from tqdm import tqdm


class GraphExperiment(Experiment):

    def __init__(self, config: GraphNetConfig, data_src: str, label_src: str, slug_src: str, labels: list):
        super().__init__("graph_net", None, config)
        self.data_generator_train: DataGenerator = None
        self.data_generator_validation: DataGenerator = None
        self.similarity_store: Dict = None
        self.data_src = data_src
        self.label_src = label_src
        self.all_slugs: list = None
        self.slug_src = slug_src
        self.labels = labels

    def setup(self):
        self.all_slugs = np.load(self.slug_src)
        #mit train test split lösen $Todo
        train_slugs = self.all_slugs[0:math.floor(len(self.all_slugs) * self.config.train_test_split)]
        validation_slugs = self.all_slugs[math.floor(len(self.all_slugs) * self.config.train_test_split):]

        if len(self.labels) == 5:
            self.data_generator_train = DataGeneratorReducedLabels(graph_src=self.data_src,
                                                                   label_src=self.label_src,
                                                                   labels=self.labels,
                                                                   slugs=train_slugs,
                                                                   one_hot=self.config.one_hot,
                                                                   shuffle=self.config.shuffle)

            self.data_generator_validation = DataGeneratorReducedLabels(graph_src=self.data_src,
                                                                        label_src=self.label_src,
                                                                        labels=self.labels,
                                                                        slugs=validation_slugs,
                                                                        one_hot=self.config.one_hot,
                                                                        shuffle=self.config.shuffle)
        else:
            self.data_generator_train = DataGenerator(graph_src=self.data_src,
                                                      label_src=self.label_src,
                                                      labels=self.labels,
                                                      slugs=train_slugs,
                                                      shuffle=self.config.shuffle,
                                                      one_hot=self.config.one_hot)

            self.data_generator_validation = DataGenerator(graph_src=self.data_src,
                                                           label_src=self.label_src,
                                                           labels=self.labels,
                                                           slugs=validation_slugs,
                                                           shuffle=self.config.shuffle,
                                                           one_hot=self.config.one_hot)

        self.model = GraphModel(self.config)

    def _run(self) -> None:
        super()._run()
        self.train()
        print("Done with train")
        self.evaluate()
        print("Done evaluation on test set")

    def cleanup(self):
        super().cleanup()
        self.data_generator = None
        self.similarity_store = None

    def train(self) -> None:
        print("Training with {} iterations, batch size={}".format(str(self.config.n_iter_train),
                                                                  str(self.config.batch_size)))
        for iter in tqdm(range(self.config.n_iter_train)):
            (inputs, targets) = self.data_generator_train.__getitem__(iter)
            loss = self.model.train_on_single_batch(inputs, targets)
            mlflow.log_metric("loss", loss[0])
            if iter % 5 == 0:
                super().evaluate_batch(inputs, targets)
        print("Training done.")

    def evaluate(self, _, data_generator: DataGenerator) -> None:
        super().evaluate(self.data_generator_validation)

    def predict(self, x) -> np.ndarray:
        prediction = self.model.predict(x)
        return prediction

    def _run_holdout(self) -> None:
        super()._run_holdout()
