import os
from src import ExperimentPipeline
from src.experiments.experiments_impls.layoutlm_baseline_experiment import LayoutLMExperiment
from src.models.layoutlm_model import LayoutLMConfig

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

exp = []
model_ids = ['Baseline_Model_LayoutLM_5_classes',
             'Baseline_Model_LayoutLM_9_classes',
             'Baseline_Model_LayoutLM_13_classes']

data_registry = {
    '5': '/content/content/MyDrive/ma-information-extraction-codebase/graph_networks/data/SROIE/LayoutLM_SROIE_data/5_classes',
    '9': '/content/content/MyDrive/ma-information-extraction-codebase/graph_networks/data/SROIE/LayoutLM_SROIE_data/9_classes',
    '13': '/content/content/MyDrive/ma-information-extraction-codebase/graph_networks/data/SROIE/LayoutLM_SROIE_data/13_classes'
}

exp.extend(LayoutLMExperiment(LayoutLMConfig(model_id=model_id,
                                             data_dir=data_registry[model_id.split('_')[-2]],
                                             model_type=model_id.split('_')[2],
                                             num_classes=int(model_id.split('_')[-2]),
                                             n_train_epochs=5,
                                             )) for model_id in model_ids)

if __name__ == "__main__":
    experiments = exp
    pipeline = ExperimentPipeline()
    pipeline.run(experiments=experiments)
