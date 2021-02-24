import time
from src.experiments.experiment_class.experiment_class import Experiment
from src.models.layoutlm_model import LayoutLMModel
import numpy as np
from src.models.layoutlm_model import LayoutLMConfig
import mlflow
from transformers import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from layoutlm.data.funsd import FunsdDataset, InputFeatures


class LayoutLMExperiment(Experiment):
    def __init__(self, config: LayoutLMConfig):
        super().__init__(config.model_id, config)

        self.text_column_name = None
        self.label_column_name = None
        self.config = config
        self.datasets = None
        self.model = None
        self.args = {'local_rank': -1,
                'overwrite_cache': True,
                'data_dir': self.config.data_dir,
                'model_name_or_path': self.config.model_name_or_path,
                'max_seq_length': self.config.max_seq_length ,
                'model_type': 'layoutlm', }

    def setup(self) -> None:
        self.model = LayoutLMModel(config=self.config)
        self.train_set, self.eval_set = self._setup_datasets()

    def _setup_datasets(self):
        train_set = FunsdDataset(self.args, self.model.tokenizer, self.config.label_list, self.model.pad_token_label_id,
                                 mode="train")
        eval_set = FunsdDataset(self.args, self.model.tokenizer, self.config.label_list, self.model.pad_token_label_id,
                                 mode="test")
        return train_set, eval_set

    def _run(self) -> None:
        super()._run()
        self.model.set_working_dir(self.working_dir)
        self._train()
        print("Done with train")
        self._evaluate()
        print("Done evaluation on evaluation set")

    def _train(self) -> None:
        train_sampler = RandomSampler(self.train_set)
        train_dataloader = DataLoader(self.train_set,
                                      sampler=train_sampler,
                                      batch_size=2)
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.model.state = 'train'
        self.model.global_step = 0
        self.model.global_epoch = 0
        for epoch in range(self.config.n_train_epochs):
            for batch in tqdm(train_dataloader, desc="Training"):
                input_ids = batch[0].to(self.model.device)
                bbox = batch[4].to(self.model.device)
                attention_mask = batch[1].to(self.model.device)
                token_type_ids = batch[2].to(self.model.device)
                labels = batch[3].to(self.model.device)
                # forward pass
                outputs = self.model.train({'input_ids':input_ids,
                                            'bbox':bbox,
                                            'attention_mask':attention_mask,
                                            'token_type_ids':token_type_ids,
                                            'labels':labels})
                if self.model.global_step % 20 == 0:
                    self._evaluate_batch(batch, outputs)
                loss = outputs.loss
                mlflow.log_metric("loss", loss)
                # print loss every 100 steps
                if self.model.global_step % 100 == 0:
                    print(f"Loss after {self.model.global_step} steps: {loss.item()}")

                # backward pass to get the gradients
                loss.backward()

                # update
                optimizer.step()
                optimizer.zero_grad()
                self.model.global_step += 1
            self._evaluate()
            self.model.global_epoch += 1

    def _evaluate_batch(self, batch, model_outputs):
        input_ids = batch[0].to(self.model.device)
        labels = batch[3].to(self.model.device)
        tokens = []
        world_level_predictions = []
        world_level_labels = []

        tmp_eval_loss = model_outputs.loss
        logits = model_outputs.logits
        eval_loss = tmp_eval_loss.item()
        preds = logits.detach().cpu().numpy()
        out_label_ids = labels.detach().cpu().numpy()
        for id_list, token_pred_list, token_label_list in zip(input_ids.squeeze().tolist(), preds, out_label_ids):
            world_level_predictions_list = []
            world_level_labels_list = []
            tokens_list = []
            for id, token_pred, token_label in zip(id_list, token_pred_list, token_label_list):
                token = self.model.tokenizer.decode([id])
                if id in [self.model.tokenizer.cls_token_id,
                          self.model.tokenizer.sep_token_id,
                          self.model.tokenizer.pad_token_id]:
                    continue
                elif token.startswith("##") and len(tokens) > 0:
                    tokens[-1] = tokens[-1] + token.replace('##', '')
                else:
                    tokens_list.append(token_pred)
                    world_level_predictions_list.append(token_pred)
                    world_level_labels_list.append(token_label)
            tokens.append(tokens_list)
            world_level_predictions.append(world_level_predictions_list)
            world_level_labels.append(world_level_labels_list)

        mlflow.log_metric('eval_loss', eval_loss)
        preds = np.argmax(preds, axis=2)

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.model.pad_token_label_id:
                    out_label_list[i].append(self.model.label_map[out_label_ids[i][j]])
                    preds_list[i].append(self.model.label_map[preds[i][j]])

        return self.model.compute_metrics((preds, out_label_ids)), \
               self.model.compute_levenshtein(world_level_predictions, world_level_labels, tokens)

    def _evaluate(self):
        start = time.time()
        eval_sampler = SequentialSampler(self.eval_set)
        eval_dataloader = DataLoader(self.eval_set,
                                     sampler=eval_sampler,
                                     batch_size=2)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # put model in evaluation mode
        self.model.state = 'eval'
        tokens = []
        world_level_predictions = []
        world_level_labels = []
        self.model.global_epoch = 'final'
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                input_ids = batch[0].to(self.model.device)
                bbox = batch[4].to(self.model.device)
                attention_mask = batch[1].to(self.model.device)
                token_type_ids = batch[2].to(self.model.device)
                labels = batch[3].to(self.model.device)

                # forward pass
                outputs = self.model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     labels=labels)
                # get the loss and logits
                tmp_eval_loss = outputs.loss
                logits = outputs.logits

                eval_loss += tmp_eval_loss.item()
                nb_eval_steps += 1

                # compute the predictions
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids, labels.detach().cpu().numpy(), axis=0
                    )
                    for id_list, token_pred_list, token_label_list in zip(input_ids.squeeze().tolist(), preds,
                                                                          labels.detach().cpu().numpy()):
                        world_level_predictions_list = []
                        world_level_labels_list = []
                        tokens_list = []
                        for id, token_pred, token_label in zip(id_list, token_pred_list, token_label_list):
                            token = self.model.tokenizer.decode([id])
                            if id in [self.model.tokenizer.cls_token_id,
                                      self.model.tokenizer.sep_token_id,
                                      self.model.tokenizer.pad_token_id]:
                                continue
                            elif token.startswith("##") and len(tokens) > 0:
                                tokens[-1] = tokens[-1] + token.replace('##', '')
                            else:
                                tokens_list.append(token_pred)
                                world_level_predictions_list.append(token_pred)
                                world_level_labels_list.append(token_label)
                        tokens.append(tokens_list)
                        world_level_predictions.append(world_level_predictions_list)
                        world_level_labels.append(world_level_labels_list)

        # compute average evaluation loss
        eval_loss = eval_loss / nb_eval_steps
        mlflow.log_metric('final_eval_loss', eval_loss)
        preds = np.argmax(preds, axis=2)

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.config.pad_token_label_id:
                    out_label_list[i].append(self.model.label_map[out_label_ids[i][j]])
                    preds_list[i].append(self.model.label_map[preds[i][j]])

        results =  self.model.compute_metrics((preds, out_label_ids))
        distances = self.model.compute_levenshtein(world_level_predictions,world_level_labels, tokens)
        end = time.time()
        print("TIME: Finished evaluation set in " + str(round(end - start, 3)) + "s")
        print("***** Eval results *****")
        for key, value in results.items():
            print(f"  {key} = {value}")
        for key, value in distances.items():
            print(f"  {key} = {value}")

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
        args = self.training_args.to_dict()
        for key in args.keys():
            mlflow.log_param(key, args[key])
        mlflow.set_tags({
            "type": "experiment",
        })
        self.mlflow_run_id = self._get_mlflow_run_id()
        mlflow.log_param('num_classes', self.config.num_classes)
        mlflow.log_param('model_id', self.config.model_id)
        mlflow.log_param('model_type', self.config.model_type)
        mlflow.log_artifacts(self.working_dir)

    def cleanup(self):
        super().cleanup()
        self.parser = None
        self.model = None
        self.datasets = None