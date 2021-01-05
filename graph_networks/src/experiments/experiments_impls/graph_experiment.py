from collections import defaultdict
from typing import Dict
import mlflow
import numpy as np
import math
from src.data.data_generators import DataGenerator, DataGeneratorReducedLabels
from src.experiments.experiment_class.experiment_class import Experiment
from src.models.graph_model import GraphNetConfig, GraphModel


class GraphExperiment(Experiment):
    def _run_holdout(self) -> None:
        pass

    def __init__(self, config: GraphNetConfig, data_generator, data_src:str, label_src: str, slug_src:str, labels: list):
        super().__init__("grapg_net", None, config, data_src, label_src, labels)
        self.data_generator_train: data_generator = None
        self.data_generator_validation: data_generator = None
        self.similarity_store: Dict = None
        self.data_src = data_src
        self.label_src = label_src
        self.all_slugs: list = None
        self.slug_src = slug_src
        self.labels = labels

    def setup(self):
        self.all_slugs = np.load(self.slug_src)
        train_slugs = self.all_slugs[0:math.floor(len(self.all_slugs)*self.config.train_test_split)]
        validation_slugs = self.all_slugs[math.floor(len(self.all_slugs)*self.config.train_test_split):]

        if len(self.labels) == 5:
            self.data_generator_train = DataGeneratorReducedLabels(graph_src=self.data_src,
                                                                   label_src=self.label_src,
                                                                   labels=self.labels,
                                                                   slugs=train_slugs)
        else:
            self.data_generator_validation = DataGenerator(graph_src=self.data_src,
                                                           label_src=self.label_src,
                                                           labels=self.train_labels,
                                                           slugs=validation_slugs,
                                                           shuffle=False)
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

    def _final_log(self):
        if self.config.plot_distances:
            mlflow.log_artifacts(self.working_dir)

    def train(self) -> None:
        print("Training with {} iterations, batch size={}".format(str(self.config.n_iter_train), str(self.config.batch_size)))
        loss = self.model.train(self.data_generator_train)
        mlflow.log_metric("loss", loss[0])
        mlflow.log_metric("accuracy", loss[1])
        print("Training done.")

    def evaluate(self, _, data_generator: DataGenerator) -> None:
        super().evaluate(self.data_generator_validation)


    def predict(self, x)  -> np.ndarray:
        prediction = self.model.predict(x)

        return prediction
