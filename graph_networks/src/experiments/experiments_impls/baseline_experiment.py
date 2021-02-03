import time
import os
from src.experiments.experiment_class.experiment_class import Experiment
from src.util.Bert_Arguments.arguments import ModelArguments, DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.utils import logging as transformer_logging
from transformers.trainer_utils import is_main_process
import logging
from datasets import ClassLabel, load_dataset
from src.models.bert_model import BertModel, BertConfig


class BertExperiment(Experiment):
    def __init__(self, config: BertConfig):
        super().__init__(config.model_id, config)

        self.features = None
        self.text_column_name = None
        self.label_column_name = None
        self.datasets = None
        self.model = None

        self.model_args = None
        self.data_args = None
        self.training_args = None

        self.parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

        self.args_dict = {
            "task_name": "ner",
            "data_dir": ".",
            "train_file": config.train_f,
            "test_file": config.test_f,
            "validation_file": config.validate_f,
            "output_dir": os.path.join(self.config.model_dir, self.config.model_id),
            "do_train": True,
            "do_eval": True,
            "do_predict": True,
            "no_cuda": False,
            "overwrite_output_dir": self.config.overwrite_output_dir
        }
        self.model_args, self.data_args, self.training_args = self.parser.parse_dict(self.args_dict)

    def setup(self) -> None:
        if (
            os.path.exists(self.training_args.output_dir)
            and os.listdir(self.training_args.output_dir)
            and self.training_args.do_train
            and not self.training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({self.training_args.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )

        if self.config.logging:
            self._setup_logging()

        set_seed(self.training_args.seed)
        untokenized_datasets, text_column_name, label_column_name, num_labels, label_to_id = self._setup_datasets()
        if num_labels != self.config.num_classes:
            raise ValueError(
                f"Number of classes given in config ({self.config.num_classes}) ist not identical to "
                f"number of labels present in training data({num_labels})"
            )

        self.model = BertModel(self.config, self.training_args, self.data_args, self.model_args)
        self.datasets = untokenized_datasets.map(
            self.model.tokenize_and_align_labels,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            fn_kwargs={'text_column_name': text_column_name,
                       'label_column_name': label_column_name,
                       'label_to_id': label_to_id}
        )

    def _setup_datasets(self):
        if self.data_args.dataset_name is not None:
            datasets = load_dataset(self.data_args.dataset_name, self.data_args.dataset_config_name)
        else:
            data_files = {}
            if self.data_args.train_file is not None:
                data_files["train"] = self.data_args.train_file
            if self.data_args.validation_file is not None:
                data_files["validation"] = self.data_args.validation_file
            if self.data_args.test_file is not None:
                data_files["test"] = self.data_args.test_file
            extension = self.data_args.train_file.split(".")[-1]
            datasets = load_dataset(extension, data_files=data_files, field='data')
        if self.training_args.do_train:
            column_names = datasets["train"].column_names
            features = datasets["train"].features
        else:
            column_names = datasets["validation"].column_names
            features = datasets["validation"].features
        text_column_name = "tokens" if "tokens" in column_names else column_names[0]
        label_column_name = (
            f"{self.data_args.task_name}_tags" if f"{self.data_args.task_name}_tags" in column_names else column_names[
                1]
        )
        if isinstance(features[label_column_name].feature, ClassLabel):
            label_list = features[label_column_name].feature.names
            # No need to convert the labels since they are already ints.
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            label_list = self._get_label_list( datasets["train"][label_column_name])
            label_to_id = {l: i for i, l in enumerate(label_list)}
        num_labels = len(label_list)

        return datasets, text_column_name, label_column_name, num_labels, label_to_id

    def _run(self) -> None:
        super()._run()
        if self.training_args.do_train:
            self._train()
            print("Done with train")
        if self.training_args.do_eval:
            self._evaluate()
            print("Done evaluation on evaluation set")
        if self.training_args.do_predict:
            self._predict()
            print('Done predicting test set')

    def _train(self) -> None:
        self.model.train(self.datasets)

    def _predict(self):
        if self.config.logging:
            self.logger.info("*** Predict ***")
        start = time.time()
        results, metrics = self.model.predict(self.datasets['test'])
        end = time.time()
        print("TIME: Finished prediction of test set in " + str(round(end - start, 3)) + "s")
        if self.config.logging:
            self.logger.info("***** Test results *****")
            for key, value in metrics.items():
                self.logger.info(f"  {key} = {value}")


    def _evaluate(self):
        start = time.time()
        results, distances = self.model.evaluate(self.datasets)
        end = time.time()
        print("TIME: Finished evaluation set in " + str(round(end - start, 3)) + "s")
        if self.config.logging:
            self.logger.info("***** Eval results *****")
            for key, value in results.items():
                self.logger.info(f"  {key} = {value}")
            for key, value in distances.items():
                self.logger.info(f"  {key} = {value}")

    @staticmethod
    def _get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    def _run_holdout(self) -> None:
        super()._run_holdout()

    def _final_log(self) -> None:
        mlflow.log_artifacts(self.working_dir)

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if is_main_process(self.training_args.local_rank) else logging.WARN,
        )
        # Log on each process the small summary:
        self.logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f"distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        if is_main_process(self.training_args.local_rank):
            transformer_logging.set_verbosity_info()
            transformer_logging.enable_default_handler()
            transformer_logging.enable_explicit_format()

        self.logger.info("Training/evaluation parameters %s", self.training_args)
        mlflow.log_param("Training/evaluation parameters", self.training_args)

    def cleanup(self):
        super().cleanup()
        self.parser = None
        self.model = None
        self.datasets = None