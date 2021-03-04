from src import GraphExperiment
from src import GraphNetConfig
from src import ExperimentPipeline
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

n_train_epochs = 5
batch_size = 5

labels = ['O', 'B-MONEY', 'I-MONEY', 'L-MONEY', 'B-ORG', 'I-ORG', 'L-ORG', 'B-DATE', 'I-DATE', 'L-DATE', 'B-GPE',
          'I-GPE', 'L-GPE']
node_count = 256
edge_count = 12500
node_vector_length = 250
edge_vector_length = 5

penalty = 0.01
lr_s = [0.0001, 0.0005, 0.00075,  0.001, 0.005]
train_test_split = 0.25
exp = []
model_ids = ['Graph_Model_CRFv2Concat_0_fold_5_classes',
             'Graph_Model_CRFv2Concat_1_fold_5_classes',
             'Graph_Model_CRFv2Concat_2_fold_5_classes']

exp.extend(GraphExperiment(GraphNetConfig(batch_size=batch_size,
                                          n_train_epochs=n_train_epochs,
                                          n_folds=int(model_id.split('_')[3]),
                                          penalty=penalty,
                                          model_id=model_id,
                                          node_count=node_count,
                                          edge_count=edge_count,
                                          node_vector_length=node_vector_length,
                                          edge_vector_length=edge_vector_length,
                                          num_classes=int(model_id.split('_')[-2]),
                                          reducer_type='mean',
                                          input_units=250,
                                          intermediate_units=250,
                                          bilstm_units=64,
                                          train_test_split=train_test_split,
                                          lr=lr,
                                          shuffle=True,
                                          one_hot=True if model_id.split('_')[2].startswith('Softmax') else False),
                           data_src=os.path.abspath('ma-information-extraction-codebase/graph_networks/data/SROIE/graphs_w2v/'),
                           label_src=os.path.abspath('ma-information-extraction-codebase/graph_networks/data/SROIE/labels_w2v/'),
                           additional_data_src = os.path.abspath('ma-information-extraction-codebase/graph_networks/data/SROIE/results_df_v2.json'),
                           labels=labels,
                           slug_src=os.path.abspath('ma-information-extraction-codebase/graph_networks/data/SROIE/slugs.npy'))
           for model_id in model_ids for lr in lr_s)

batch_size = 2
model_ids = ['Graph_Model_CRFv2Concat_3_fold_5_classes']

exp.extend(GraphExperiment(GraphNetConfig(batch_size=batch_size,
                                          n_train_epochs=n_train_epochs,
                                          n_folds=int(model_id.split('_')[3]),
                                          penalty=penalty,
                                          model_id=model_id,
                                          node_count=node_count,
                                          edge_count=edge_count,
                                          node_vector_length=node_vector_length,
                                          edge_vector_length=edge_vector_length,
                                          num_classes=int(model_id.split('_')[-2]),
                                          reducer_type='mean',
                                          input_units=250,
                                          intermediate_units=250,
                                          bilstm_units=64,
                                          train_test_split=train_test_split,
                                          lr=lr,
                                          shuffle=True,
                                          one_hot=True if model_id.split('_')[2].startswith('Softmax') else False),
                           data_src=os.path.abspath('ma-information-extraction-codebase/graph_networks/data/SROIE/graphs_w2v/'),
                           label_src=os.path.abspath('ma-information-extraction-codebase/graph_networks/data/SROIE/labels_w2v/'),
                           additional_data_src = os.path.abspath('ma-information-extraction-codebase/graph_networks/data/SROIE/results_df_v2.json'),
                           labels=labels,
                           slug_src=os.path.abspath('ma-information-extraction-codebase/graph_networks/data/SROIE/slugs.npy'))
           for model_id in model_ids for lr in lr_s)

if __name__ == "__main__":
    experiments = exp
    pipeline = ExperimentPipeline()
    pipeline.run(experiments)
