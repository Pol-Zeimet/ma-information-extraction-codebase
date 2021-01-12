import os
from src import GraphExperiment
from src import GraphNetConfig
from src import ExperimentPipeline

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

n_iter_eval = 10
n_iter_train = 40
batch_size = 10

labels = ['O', 'B-MONEY', 'I-MONEY', 'L-MONEY', 'B-ORG', 'I-ORG', 'L-ORG', 'B-DATE', 'I-DATE', 'L-DATE', 'B-GPE',
          'I-GPE', 'L-GPE']
less_labels = ['O', 'I-MONEY', 'I-ORG', 'I-DATE', 'I-GPE']
node_count = 512
edge_count = 40000
node_vector_length = 728
edge_vector_length = 5

penalty = 0.01
lr_s = [0.001, 0.005, 0.01]
text = "text"
train_est_split = 0.25
exp = []
model_ids = ['Graph_Model_Softmax_0_fold_5_classes',
             'Graph_Model_Softmax_1_fold_5_classes',
             'Graph_Model_Softmax_2_fold_5_classes',
             'Graph_Model_Softmax_3_fold_5_classes']

exp.extend(GraphExperiment(GraphNetConfig(batch_size=batch_size,
                                          n_iter_eval=n_iter_eval,
                                          n_iter_train=n_iter_train,
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
                                          lr=lr,
                                          shuffle=True,
                                          one_hot=True if model_id.split('_')[2] == 'Softmax' else False),
                           data_src=os.path.abspath('graph_networks/data/SROIE/graphs/'),
                           label_src=os.path.abspath('graph_networks/data/SROIE/labels/'),
                           labels=less_labels if model_id.split('_')[-2] == 5 else labels,
                           slug_src=os.path.abspath('graph_networks/data/SROIE/slugs.npy'))
           for model_id in model_ids for lr in lr_s)

if __name__ == "__main__":
    experiments = exp
    pipeline = ExperimentPipeline()
    pipeline.run(experiments)
