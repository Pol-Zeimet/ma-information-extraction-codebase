from typing import Dict, Any, Optional

import numpy as np
from keras import optimizers
from keras.utils import plot_model

from src.main.experiments.config import Config
from src.main.models.model_architectures.graph_model_architecturte import  Graph_Model_Softmax



class GraphNetConfig(Config):
    def __init__(self, base_path: str,
                 penalty: float,
                 eval_type: str,
                 n_iter_train: int,
                 n_iter_eval: int,
                 batchsize: int,
                 lr: float,
                 col_label: str,
                 model_id: str,
                 node_count=512,
                 edge_count=40000,
                 logging: bool = False,
                 k_per_class: Optional[int] = None,
                 decay: float = 0.0):
        super().__init__(base_path, eval_type, k_per_class, n_iter_eval, col_label, logging)
        self.penalty = penalty
        self.batchsize = batchsize
        self.learning_rate = lr
        self.n_iter_train = n_iter_train
        self.node_count = node_count
        self.edge_count = edge_count
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        self.decay = decay
        self.model_id = model_id


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
                "representation": self.config.pretrained_embeddings,
                }

    def create_model_architecture(self) -> None:
        if self.model_id == "Gaph_Model_Softmax":
            self.model = Graph_Model_Softmax.create(self.node_count, self.edge_count)

        print(self.model.summary)
        adam = optimizers.Adam(lr=self.config.learning_rate,
                               beta_1=self.config.adam_beta_1,
                               beta_2=self.config.adam_beta_2,
                               epsilon=None,
                               decay=self.config.decay,
                               amsgrad=False)

        self.model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, inputs, targets) -> None:
        return self.model.train_on_batch(inputs, [targets])


    ##needs rework
    def predict(self, x_test, session, target_labels, path: str = None) -> np.ndarray:
        similarities = self.model.predict([x_test[0], x_test[1]])
        self.similarity_store = similarities

        predicts = target_labels[np.argmax(similarities)]

        return predicts

    def save_architecture_as_plot(self, path: str) -> None:
        plot_model(self.model, to_file=path + 'siamese_network.png')

    def save_weights(self, path: str) -> None:
        model_file = path + 'graph_model.h5'

        self.model.save_weights(model_file)
        print("Model saved in " + model_file)


