
import argparse
import numpy as np
from tqdm import tqdm

from src.utils.utils import build_graph, Embedder, find_max_positions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_type')
    parser.add_argument('--data_src')
    #todo
    parser.add_argument('--model_src', help = "Only relevant for BERT Embeddings. defaul is set to trained BERT Baseline Model. ")
    args = parser.parse_args()
    embedder = Embedder(embedding_type=args.embedding_type, model_src=args.model_src)

    #todo
    data = None
    file_count = len(set(data.slug.values)) - 1
    i = 0
    slugs = []
    for slug in tqdm(set(data.slug.values)):
        subset = data[data.slug == slug]
        max_X_pos, max_Y_pos = find_max_positions(data.position)
        aggregated_positions = [[1000 * float(subset.position[i][0]) / max_X_pos,
                                 1000 * float(subset.position[i][1]) / max_Y_pos,
                                 1000 * float(subset.position[i][4]) / max_X_pos,
                                 1000 * float(subset.position[i][5]) / max_Y_pos]
                                for i, row in subset.iterrows() for token in subset.tokens[i]]
        aggregated_tokens = [token for tokens in subset.tokens for token in tokens]
        aggregated_labels = [label for labels in subset.labels for label in labels]
        np.save(graph_dir + slug, build_graph(aggregated_positions, aggregated_tokens, embedder, include_globals=True))
        np.save(label_dir + slug, aggregated_labels)

    np.save(slug_dst, slugs)