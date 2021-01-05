from typing import Dict, Any, Optional

import numpy as np
from keras import optimizers

#from keras.utils import plot_model

from src.experiments.config import Config
from src.models.model_architectures.graph_model_architecture import GraphModelSoftmax


class GraphNetConfig(Config):
    def __init__(self, base_path: str,
                 penalty: float,
                 n_iter_train: int,
                 n_iter_eval: int,
                 batch_size: int,
                 lr: float,
                 model_id: str,
                 graph_fold_count: int,
                 node_count: int,
                 edge_count: int,
                 node_vector_length: int,
                 edge_vector_length: int,
                 num_classes: int,
                 n_folds: int,
                 col_label: str,
                 logging: bool = False,
                 decay: float = 0.0):
        super().__init__(base_path, n_iter_eval = n_iter_eval, col_label = col_label, logging = logging,num_classes = num_classes)
        self.model_id = model_id
        self.batch_size = batch_size
        self.n_iter_train = n_iter_train
        self.learning_rate = lr
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        self.penalty = penalty
        self.decay = decay

        self.graph_fold_count = graph_fold_count
        self.node_count = node_count
        self.edge_count = edge_count
        self.node_vector_length = node_vector_length
        self.edge_vector_length = edge_vector_length
        self.n_folds = n_folds


class GraphModel:
    def __init__(self, config: GraphNetConfig):
        self.config = config
        self.logging = config.logging
        self.model_id = config.model_id
        self.create_model_architecture()
        self.similarity_store = None
        self.model = None

    def get_details(self) -> Dict[str, Any]:
        return {"model": "Graph_Net_Softmax",
                "padded node count": self.config.node_count,
                "padded edge count": self.config.edge_count,
                }

    def create_model_architecture(self) -> None:
        if self.model_id == "Gaph_Model_Softmax":
            self.model = GraphModelSoftmax.create(self.config.node_count,
                                                  self.config.edge_count,
                                                  self.config.node_vector_length,
                                                  self.config.edge_vector_length,
                                                  self.config.n_folds,
                                                  self.config.num_classes)

        print(self.model.summary)
        adam = optimizers.Adam(lr=self.config.learning_rate,
                               beta_1=self.config.adam_beta_1,
                               beta_2=self.config.adam_beta_2,
                               epsilon=None,
                               decay=self.config.decay,
                               amsgrad=False)

        self.model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    def train_on_generator(self, train_generator) -> None:
        return self.model.fit(verbose=1, x=train_generator, epochs=15, steps_per_epoch=20)

    def train_on_single_batch(self, inputs, targets) -> None:
        return self.model.train_on_batch(inputs, [targets])

    ##needs rework
    def predict(self) -> np.ndarray:

        predicts = self.model.predict()

        return predicts

    #def save_architecture_as_plot(self, path: str) -> None:
    #    plot_model(self.model, to_file=path + 'siamese_network.png')

    def save_weights(self, path: str) -> None:
        model_file = path + 'graph_model' + self.config.model_id + '.h5'

        self.model.save_weights(model_file)
        print("Model saved in " + model_file)


