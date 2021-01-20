from typing import Dict, Any

import numpy as np
from tensorflow import argmax as tf_argmax, one_hot as tf_one_hot
from tensorflow.keras import optimizers
from src.experiments.config import Config
from src.models.model_architectures.graph_model_architecture import GraphModelSoftmax, GraphModelCRF, GraphModelCRFv2
from tensorflow.keras import Model


class GraphNetConfig(Config):
    def __init__(self,
                 penalty: float,
                 n_iter_train: int,
                 n_iter_eval: int,
                 batch_size: int,
                 lr: float,
                 model_id: str,
                 node_count: int,
                 edge_count: int,
                 node_vector_length: int,
                 edge_vector_length: int,
                 num_classes: int,
                 n_folds: int,
                 reducer_type: str,
                 input_units: int,
                 intermediate_units: int,
                 bilstm_units: int,
                 one_hot: bool,
                 shuffle: bool,
                 logging: bool = False,
                 train_test_split: float = 0.25,
                 decay: float = 0.0):
        super().__init__(
            logging=logging,
            num_classes=num_classes)
        self.model_id = model_id
        self.batch_size = batch_size
        self.n_iter_train = n_iter_train
        self.n_iter_eval = n_iter_eval
        self.learning_rate = lr
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        self.penalty = penalty
        self.decay = decay
        self.one_hot = one_hot
        self.shuffle = shuffle
        self.node_count = node_count
        self.edge_count = edge_count
        self.node_vector_length = node_vector_length
        self.edge_vector_length = edge_vector_length
        self.n_folds = n_folds
        self.reducer_type = reducer_type
        self.input_units = input_units
        self.intermediate_units = intermediate_units
        self.bilstm_units = bilstm_units
        self.train_test_split = train_test_split


class GraphModel:
    def __init__(self, config: GraphNetConfig):
        self.config = config
        self.logging = config.logging
        self.model_id = config.model_id
        self.model_type = self.model_id.split('_')[2]
        self.model: Model = None
        self._create_model_architecture()

    def get_details(self) -> Dict[str, Any]:
        return {
            "model": self.model_id,
            "model type": self.model_type,
            "padded node count": self.config.node_count,
            "padded edge count": self.config.edge_count,
            "node_vector_length": self.config.node_vector_length,
            "edge_vector_length": self.config.edge_vector_length,
            "n_folds": self.config.n_folds,
            "reducer_type": self.config.reducer_type,
            "input_units": self.config.input_units,
            "intermediate_units": self.config.intermediate_units,
            "bilstm_units": self.config.bilstm_units
        }

    def _create_model_architecture(self) -> None:
        adam = optimizers.Adam(lr=self.config.learning_rate,
                               beta_1=self.config.adam_beta_1,
                               beta_2=self.config.adam_beta_2,
                               epsilon=None,
                               decay=self.config.decay,
                               amsgrad=False)

        if self.model_type == "Softmax":
            self.model = GraphModelSoftmax.create(self.config.node_count,
                                                  self.config.edge_count,
                                                  self.config.node_vector_length,
                                                  self.config.edge_vector_length,
                                                  self.config.n_folds,
                                                  self.config.num_classes,
                                                  self.config.reducer_type,
                                                  self.config.input_units,
                                                  self.config.intermediate_units,
                                                  self.config.bilstm_units,
                                                  adam
                                                  )

            self.model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

        elif self.model_type == "CRF":
            self.model = GraphModelCRF.create(self.config.node_count,
                                              self.config.edge_count,
                                              self.config.node_vector_length,
                                              self.config.edge_vector_length,
                                              self.config.n_folds,
                                              self.config.num_classes,
                                              self.config.reducer_type,
                                              self.config.input_units,
                                              self.config.intermediate_units,
                                              self.config.bilstm_units,
                                              adam
                                              )
            self.model.compile(optimizer=adam, metrics=["accuracy"])

        elif self.model_type == "CRFv2":
            self.model = GraphModelCRFv2.create(self.config.node_count,
                                                self.config.edge_count,
                                                self.config.node_vector_length,
                                                self.config.edge_vector_length,
                                                self.config.n_folds,
                                                self.config.num_classes,
                                                self.config.reducer_type,
                                                self.config.input_units,
                                                self.config.intermediate_units,
                                                self.config.bilstm_units,
                                                adam
                                                )

    def train_on_generator(self, train_generator):
        return self.model.fit(verbose=1, x=train_generator, epochs=15, steps_per_epoch=20)

    def train_on_single_batch(self, inputs, targets):
        return self.model.train_on_batch(inputs, [targets])

    def predict(self, x) -> (np.ndarray, np.ndarray):
        output = self.model.predict(x)
        if self.model_type == 'Softmax':
            predictions = output[0]
            graph_embeddings = output[1]
            predictions = tf_one_hot(tf_argmax(predictions, axis=2), depth=self.config.num_classes)
            predictions = [np.where(labels == 1)[0][0] for prediction in predictions for labels in prediction]
        else:
            predictions = output[0][0]
            graph_embeddings = output[1]

        return predictions, graph_embeddings

    def embed(self, x) -> np.ndarray:
        output = self.model.predict(x)
        graph_embeddings = output[1]
        return graph_embeddings

    def save_weights(self, path: str) -> None:
        model_file = path + 'graph_model' + self.config.model_id + '.h5'

        self.model.save_weights(model_file)
        print("Model saved in " + model_file)
