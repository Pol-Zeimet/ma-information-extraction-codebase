import os
from src import BertExperiment
from src import BertConfig
from src import ExperimentPipeline

exp = []
model_ids = ['Baseline_Model_Bert_5_classes',
             'Baseline_Model_Bert_9_classes',
             'Baseline_Model_Bert_13_classes']

data_registry ={
    '5':
        {
            "train_f": os.path.abspath('../data/SROIE/NER/train_less_classes.json'),
            "test_f": os.path.abspath('../data/SROIE/NER/test_less_classes.json'),
            "validate_f": os.path.abspath('../data/SROIE/NER/validate_less_classes.json'),
            "labels" : ['O', 'I-MONEY', 'I-ORG', 'I-DATE', 'I-GPE']
        },
    '9':
        {
            "train_f": os.path.abspath('../data/SROIE/NER/train_medium_classes.json'),
            "test_f": os.path.abspath('../data/SROIE/NER/test_medium_classes.json'),
            "validate_f": os.path.abspath('../data/SROIE/NER/validate_medium_classes.json'),
            "labels": ['O', 'B-MONEY', 'I-MONEY', 'B-ORG', 'I-ORG', 'B-DATE', 'I-DATE', 'B-GPE', 'I-GPE']
        },
    '13':
        {
            "train_f": os.path.abspath('../data/SROIE/NER/train_classes.json'),
            "test_f": os.path.abspath('../data/SROIE/NER/test_classes.json'),
            "validate_f": os.path.abspath('../data/SROIE/NER/validate_classes.json'),
            "labels": ['O', 'B-MONEY', 'I-MONEY', 'L-MONEY', 'B-ORG', 'I-ORG', 'L-ORG', 'B-DATE', 'I-DATE', 'L-DATE',
                       'B-GPE', 'I-GPE', 'L-GPE']
        }
}

exp.extend(BertExperiment(BertConfig(model_dir="",
                                     model_id=model_id,
                                     model_type=model_id.split('_')[2],
                                     label_list=data_registry[model_id.split('_')[-2]]["labels"],
                                     num_classes=int(model_id.split('_')[-2]),
                                     train_f=data_registry[model_id.split('_')[-2]]["train_f"],
                                     test_f=data_registry[model_id.split('_')[-2]]["test_f"],
                                     validate_f=data_registry[model_id.split('_')[-2]]["validate_f"],
                                     logging=True
                                     )) for model_id in model_ids)

if __name__ == "__main__":
    experiments = exp
    pipeline = ExperimentPipeline()
    pipeline.run(experiments)