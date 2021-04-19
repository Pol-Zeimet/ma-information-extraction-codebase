import mlflow
import numpy as np
import math
from model_experiments.data.data_generators import DataGenerator, DataGeneratorReducedLabels
from model_experiments.experiments.experiment_class.experiment_class import Experiment
from model_experiments.model_classes.graph_model import GraphNetConfig, GraphModel
from tqdm import tqdm


class GraphExperiment(Experiment):

    def __init__(self, config: GraphNetConfig, data_src: str, label_src: str, doc_name_src: str, additional_data_src: str, labels: list):
        super().__init__(config.model_id, config)
        self.data_generator_train: DataGenerator = None
        self.data_generator_validation: DataGenerator = None

        self.data_src = data_src
        self.label_src = label_src
        self.additional_data_src = additional_data_src
        self.all_doc_names: list = None
        self.doc_name_src = doc_name_src
        self.labels = labels
        self.model = None
        self.inputs_for_embeddings = None
        self.targets_for_embeddings = None
        self.tokens_for_embeddings = None
        self.positions_for_embeddings = None
        self.n_train_steps = None
        self.n_eval_steps = None

    def setup(self):
        self.all_doc_names = np.load(self.doc_name_src)
        train_doc_names = self.all_doc_names[math.floor(len(self.all_doc_names) * self.config.train_test_split):]
        validation_doc_names = self.all_doc_names[0:math.floor(len(self.all_doc_names) * self.config.train_test_split)]
        self.n_train_steps = len(train_doc_names) // self.config.batch_size
        self.n_eval_steps = len(validation_doc_names) // self.config.batch_size

        if self.config.num_classes == 5:
            self.data_generator_train = DataGeneratorReducedLabels(graph_src=self.data_src,
                                                                   label_src=self.label_src,
                                                                   labels=self.labels,
                                                                   doc_names=train_doc_names,
                                                                   additional_data_src=self.additional_data_src,
                                                                   node_shape=self.config.node_vector_length,
                                                                   node_count=self.config.node_count,
                                                                   edge_shape=self.config.edge_vector_length,
                                                                   edge_count=self.config.edge_count,
                                                                   batch_size=self.config.batch_size,
                                                                   one_hot=self.config.one_hot,
                                                                   shuffle=self.config.shuffle)

            self.data_generator_validation = DataGeneratorReducedLabels(graph_src=self.data_src,
                                                                        label_src=self.label_src,
                                                                        labels=self.labels,
                                                                        doc_names=validation_doc_names,
                                                                        additional_data_src=self.additional_data_src,
                                                                        node_shape=self.config.node_vector_length,
                                                                        node_count=self.config.node_count,
                                                                        edge_shape=self.config.edge_vector_length,
                                                                        edge_count=self.config.edge_count,
                                                                        batch_size=self.config.batch_size,
                                                                        one_hot=self.config.one_hot,
                                                                        shuffle=self.config.shuffle)
        else:
            self.data_generator_train = DataGenerator(graph_src=self.data_src,
                                                      label_src=self.label_src,
                                                      labels=self.labels,
                                                      doc_names=train_doc_names,
                                                      additional_data_src=self.additional_data_src,
                                                      node_shape=self.config.node_vector_length,
                                                      node_count=self.config.node_count,
                                                      edge_shape=self.config.edge_vector_length,
                                                      edge_count=self.config.edge_count,
                                                      batch_size=self.config.batch_size,
                                                      shuffle=self.config.shuffle,
                                                      one_hot=self.config.one_hot)

            self.data_generator_validation = DataGenerator(graph_src=self.data_src,
                                                           label_src=self.label_src,
                                                           labels=self.labels,
                                                           doc_names=validation_doc_names,
                                                           additional_data_src=self.additional_data_src,
                                                           node_shape=self.config.node_vector_length,
                                                           node_count=self.config.node_count,
                                                           edge_shape=self.config.edge_vector_length,
                                                           edge_count=self.config.edge_count,
                                                           batch_size=self.config.batch_size,
                                                           shuffle=self.config.shuffle,
                                                           one_hot=self.config.one_hot)

        self.model = GraphModel(self.config)

    def _run(self) -> None:
        super()._run()
        self._train()
        print("Done with train")
        self._evaluate(self.data_generator_validation, self.n_eval_steps, 'final')
        print("Done evaluation on test set")

    def cleanup(self):
        super().cleanup()
        self.data_generator_train = None
        self.data_generator_validation = None
        self.inputs_for_embeddings = None
        self.targets_for_embeddings = None
        self.tokens_for_embeddings = None
        self.positions_for_embeddings = None

    def _train(self) -> None:
        print("Training for {} epochs with {} steps and batch size={}".format(str(self.config.n_train_epochs),
                                                                              str(self.n_train_steps),
                                                                              str(self.config.batch_size)))

        self.inputs_for_embeddings, self.targets_for_embeddings = self.data_generator_validation.__getitem__(0)
        self.tokens_for_embeddings, self.positions_for_embeddings = self.data_generator_validation.get_tokens_and_positions(
            0)
        super()._evaluate_embeddings(self.inputs_for_embeddings, self.targets_for_embeddings,
                                     self.tokens_for_embeddings, self.positions_for_embeddings, epoch='init',
                                     step=None)

        for epoch in tqdm(range(self.config.n_train_epochs)):
            for train_step in tqdm(range(self.n_train_steps), ascii=True, desc=f"{self.config.model_id}, "
                                                                               f" {self.config.learning_rate},"
                                                                               f"  epoch {epoch}"):
                inputs_train, targets_train = self.data_generator_train.__getitem__(train_step)
                loss = self.model.train_on_single_batch(inputs_train, targets_train)
                mlflow.log_metric("loss", loss)
                super()._evaluate_embeddings(self.inputs_for_embeddings, self.targets_for_embeddings,
                                             self.tokens_for_embeddings, self.positions_for_embeddings,
                                             epoch, train_step)
                if train_step % 5 == 0:
                    tokens, positions = self.data_generator_train.get_tokens_and_positions(train_step)
                    super()._evaluate_batch(inputs_train, targets_train, tokens, positions, epoch, train_step)

            self._evaluate(self.data_generator_validation, self.n_eval_steps, epoch)
            self.data_generator_validation.on_epoch_end()
            self.data_generator_train.on_epoch_end()

        print("Training done.")

    def _evaluate(self, data_generator: DataGenerator, n_eval_steps, epoch) -> None:
        super()._evaluate(data_generator, n_eval_steps, epoch)

    def _predict(self, x) -> np.ndarray:
        prediction = self.model.predict(x)
        return prediction

    def _run_holdout(self) -> None:
        raise NotImplementedError

    def _final_log(self) -> None:
        mlflow.log_param('data', self.data_src.split('/')[-2]+self.data_src.split('/')[-1])
        super()._final_log()


