import os
from model_experiments.experiments import GraphExperiment
from model_experiments.model_classes.graph_model import GraphNetConfig
from model_experiments.experiments import ExperimentPipeline

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

n_train_epochs = 5
batch_size = 5

labels = ['O', 'B-MONEY', 'I-MONEY', 'L-MONEY', 'B-ORG', 'I-ORG', 'L-ORG', 'B-DATE', 'I-DATE', 'L-DATE', 'B-GPE',
          'I-GPE', 'L-GPE']
node_count = 256
edge_count = 25000
node_vector_length = 768
edge_vector_length = 5

penalty = 0.01
lr_s = [0.0001, 0.0005, 0.00075, 0.001, 0.005]
train_test_split = 0.25
exp = []
model_ids = ['Graph_Model_Softmax_0_fold_5_classes',
             'Graph_Model_Softmax_1_fold_5_classes',
             'Graph_Model_Softmax_2_fold_5_classes']

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
                                          input_units=768,
                                          intermediate_units=768,
                                          bilstm_units=64,
                                          train_test_split=train_test_split,
                                          lr=lr,
                                          shuffle=True,
                                          one_hot=True if model_id.split('_')[2] == 'Softmax' else False),
                           data_src=os.path.join(os.path.dirname(__file__), '../../data/SROIE/graphs/'),
                           label_src=os.path.join(os.path.dirname(__file__), '../../data/SROIE/labels/'),
                           additional_data_src=os.path.join(os.path.dirname(__file__),
                                                            '../../data/SROIE/results_df.json'),
                           labels=labels,
                           doc_name_src=os.path.join(os.path.dirname(__file__), '../../data/SROIE/doc_names.npy'))
           for model_id in model_ids for lr in lr_s)

batch_size = 2
model_ids = ['Graph_Model_Softmax_3_fold_5_classes']
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
                                          input_units=768,
                                          intermediate_units=768,
                                          bilstm_units=64,
                                          train_test_split=train_test_split,
                                          lr=lr,
                                          shuffle=True,
                                          one_hot=True if model_id.split('_')[2] == 'Softmax' else False),
                           data_src=os.path.join(os.path.dirname(__file__), '../../data/SROIE/graphs/'),
                           label_src=os.path.join(os.path.dirname(__file__), '../../data/SROIE/labels/'),
                           additional_data_src=os.path.join(os.path.dirname(__file__),
                                                            '../../data/SROIE/results_df.json'),
                           labels=labels,
                           doc_name_src=os.path.join(os.path.dirname(__file__), '../../data/SROIE/doc_names.npy'))
           for model_id in model_ids for lr in lr_s)
if __name__ == "__main__":
    experiments = exp
    pipeline = ExperimentPipeline()
    pipeline.run(experiments)
