from src.Predictor_class.predictor import Predictor
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract entities from a scanned receipt '
                                                 'using one of the possible models')
    parser.add_argument('--model_type', choices=['bert', 'layoutlm', 'graph'], nargs=1,
                        help="Model type. graph, layoutlm, bert. Default is LayoutLm", default='layoutlm')
    parser.add_argument('--model_path', type=str, nargs=1,
                        help="Path to the model or the model weights. Default is path to LayoutLm",
                        default=os.path.join(os.path.dirname(__file__), '../data/images/output_dir'))
    parser.add_argument('--image_path', type=str, nargs=1, help="Path to the scan of the receipt", required=True)
    parser.add_argument('--output_dir', type=str, nargs=1, help="Path to save the annotated image to")
    parser.add_argument('--embeddings', choices=["w2v", "bert"], nargs=1,
                        help="Embeddings type. Only for graph nets and rnn. w2v or bert", default='bert')
    parser.add_argument('--embedding_size', choices=[768, 250], nargs=1,
                        help="Length of the embeddings. Only needed fr graph nets and rnn. "
                             "Standard is 768 for bert, 250 for w2v",
                        default=768)
    parser.add_argument('--padded_node_count', type=int, nargs=1,
                        help="node count in graph will be padded to this amount to fit model. "
                             "Only needed fr graph nets and rnn. Standard is 256",
                        default=256)
    parser.add_argument('--padded_edge_count', type=int, nargs=1,
                        help="edge  count  in graph will be padded to this amount to fit model. "
                             "Only needed fr graph nets and rnn. Standard is 25000.",
                        default=25000)

    args = parser.parse_args()
    predictor = Predictor(model_name=args.model_type, model_path=args.model_path, output_dir=args.output_dir,
                          embeddings=args.embeddings, pad_to_nodes=args.padded_node_count,
                          pad_to_edges=args.padded_edge_count, embedding_size=args.embedding_size)
    predictor.create()
    predictor.predict_image(args.image_path)
