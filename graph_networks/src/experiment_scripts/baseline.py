import os
from src import BertExperiment, ExperimentPipeline, BertConfig

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

exp = []
model_ids = ['Baseline_Model_Bert_5_classes',
             'Baseline_Model_Bert_9_classes',
             'Baseline_Model_Bert_13_classes']

data_registry ={
    '5':
        {
            "train_f": os.path.abspath('graph_networks/data/SROIE/Baseline/train_less_classes.json'),
            "test_f": os.path.abspath('graph_networks/data/SROIE/Baseline/test_less_classes.json'),
            "validate_f": os.path.abspath('graph_networks/data/SROIE/Baseline/validate_less_classes.json'),
            "labels" : ['O', 'I-MONEY', 'I-ORG', 'I-DATE', 'I-GPE']
        },
    '9':
        {
            "train_f": os.path.abspath('graph_networks/data/SROIE/Baseline/train_medium_classes.json'),
            "test_f": os.path.abspath('graph_networks/data/SROIE/Baseline/test_medium_classes.json'),
            "validate_f": os.path.abspath('graph_networks/data/SROIE/Baseline/validate_medium_classes.json'),
            "labels": ['O', 'B-MONEY', 'I-MONEY', 'B-ORG', 'I-ORG', 'B-DATE', 'I-DATE', 'B-GPE', 'I-GPE']
        },
    '13':
        {
            "train_f": os.path.abspath('graph_networks/data/SROIE/Baseline/train_classes.json'),
            "test_f": os.path.abspath('graph_networks/data/SROIE/Baseline/test_classes.json'),
            "validate_f": os.path.abspath('graph_networks/data/SROIE/Baseline/validate_classes.json'),
            "labels": ['O', 'B-MONEY', 'I-MONEY', 'L-MONEY', 'B-ORG', 'I-ORG', 'L-ORG', 'B-DATE', 'I-DATE', 'L-DATE',
                       'B-GPE', 'I-GPE', 'L-GPE']
        }
}

exp.extend(BertExperiment(BertConfig(model_dir="graph_networks/Checkpoints/",
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