import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
import os
from util.preprocessing_utils import build_graph, Embedder, find_max_positions


def process_df_to_graph(subset):
    max_X_pos, max_Y_pos = find_max_positions(subset.position)
    aggregated_positions = [[1000 * float(subset.position[i][0]) / max_X_pos,
                             1000 * float(subset.position[i][1]) / max_Y_pos,
                             1000 * float(subset.position[i][4]) / max_X_pos,
                             1000 * float(subset.position[i][5]) / max_Y_pos]
                            for i, row in subset.iterrows() for token in subset.tokens[i]]
    aggregated_tokens = [token for tokens in subset.tokens for token in tokens]
    aggregated_labels = [label for labels in subset.labels for label in labels]
    return build_graph(aggregated_positions, aggregated_tokens, embedder, include_globals=True), aggregated_labels




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_type')
    parser.add_argument('--data_src', type=str, help="Path to Dataframe created in preprocessing. Specifically the results_df.json for graphs.")
    parser.add_argument('--embedding_model_src', help = "Only relevant when using BERT Embeddings. Defaul path  is set to trained BERT Baseline Model.")
    parser.add_argument('--labels_output_dir', type=str, help="Path to Output directory for labels",
                        default=os.path.join(os.path.dirname(__file__), "../model_experiments/data/SROIE/labels"))
    parser.add_argument('--graphs_output_dir', type=str, help="Path to Output directory for labels",
                        default=os.path.join(os.path.dirname(__file__), "../model_experiments/data/SROIE/graphs"))
    args = parser.parse_args()
    embedder = Embedder(embedding_type=args.embedding_type, model_src=args.model_src)

    data = pd.read_json(args.data_src)
    file_count = len(set(data.doc_name.values)) - 1
    for doc_name in tqdm(set(data.doc_name.values)):
        subset = data[data.doc_name == doc_name]
        graph, labels = process_df_to_graph(subset)
        graph_path = os.path.join(args.graphs_output_dir, doc_name)
        np.save(graph_path, graph)

        label_path = os.path.join(args.labels_output_dir, doc_name)
        np.save(label_path, labels)
