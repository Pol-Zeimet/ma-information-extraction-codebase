import statistics
import numpy as np
from tqdm import tqdm
from .embedder import Embedder
from typing import List
from PIL import Image
from transformers import LayoutLMTokenizer

def is_overlapping(box1: List, box2: List) -> bool:
    """Checks if two boxes are overlapping
    Args:
    :param box1 like [x1,y1,x2,y2]
    :param box2 like [x1,y1,x2,y2]
    Returns
    :return Boolean True if overlapping, else  False
    """
    # boxshape in our case:
    return (box1[2] >= box2[0] and box2[2] >= box1[0]) and (box1[3] >= box2[1] and box2[3] >= box1[1])


def define_neighborhood(box):
    """creates a neighbourhood box around a box by extending it"""
    # boxshape in our case: [x1,y1,x2,y2]
    return [box[0] - 100, box[1] - 80, box[2] + 20, box[3] + 20]


def find_max_positions(positions: List) -> (float, float):
    """Returns max x and y position of  bounding boxes in document
    :param positions List like [[x1,y1,..., x4,y4], ...,[x1,y1,..., x4,y4]] with clockwise corner positions starting top left
    :return tuple of max x and y positions
    """

    max_x = max([float(box[2]) for box in positions])
    max_y = max([float(box[3]) for box in positions])
    return max_x, max_y


def calculate_edge(sender_pos: List, receiver_pos: List, identical: bool) -> List:
    """Calculate edge between two nodes
    Args:
        :param sender_pos sender position like [x1,y1,x2,y2]
        :param receiver_pos: receiver position like [x1,y1,x2,y2]
        :param identical:  bool if positions are identical. This can happen depending on the preprocessing.
    Returns:
        :return List representing the edge.
        Contains:
            the x and y distance between nodes
            Sender Aspect ratio
            relative width of sender compared to receiver
            relative height of sender compared  to receiver
    """
    if identical:
        return [0.0,  # x distance
                0.0,  # y distance
                (sender_pos[2] - sender_pos[0]) / (sender_pos[3] - sender_pos[1]),  # sender aspect ratio
                1.0,  # relative width
                1.0  # relative height
                ]
    else:
        sender_middle = [statistics.mean([sender_pos[0], sender_pos[2]]),
                         statistics.mean([sender_pos[1], sender_pos[3]])]
        receiver_middle = [statistics.mean([receiver_pos[0], receiver_pos[2]]),
                           statistics.mean([receiver_pos[1], receiver_pos[3]])]

        return [sender_middle[0] - receiver_middle[0],  # x distance
                sender_middle[1] - receiver_middle[1],  # y distance
                (sender_pos[2] - sender_pos[0]) / (sender_pos[3] - sender_pos[1]),  # sender aspect ratio
                (sender_pos[2] - sender_pos[0]) / (receiver_pos[2] - receiver_pos[0]),  # relative width
                (sender_pos[3] - sender_pos[1]) / (receiver_pos[3] - receiver_pos[1])  # relative height
                ]


def iob_to_label(label):
    """returns label without prefix"""
    if label != 'O':
        return label[2:]
    else:
        return "other"


def build_graph(positions: List, tokens: List, embedder: Embedder, include_globals: bool = False) -> [np.ndarray,
                                                                                                      np.ndarray,
                                                                                                      np.ndarray,
                                                                                                      np.ndarray,
                                                                                                      np.ndarray]:
    """Build graph using Tokens, their Positions and a Embedder object to calculate token embeddings
        Args:
            :param positions List of positions like [[[x1,y1, x2,y2], ...,[x1,y1, x2,y2]]
            :param tokens List of token
            :param embedder Embedder Object to calculate word embeddings
            :param include_globals Indicate if a global representation vector for the graph is to be calculated.
            Vector will be None otherwise
        Return:
            :returns a List containing:
                a numpy array as lobal graph vector
                a numpy array of Nodes
                a numpy array of Edges
                a numpy array containing node ids representing senders
                a numpy array containing node ids representing receivers


    """
    if include_globals:
        globals = embedder.embed(''.join(tokens), is_split_into_words=False, truncation=True)
    else:
        globals = None

    nodes = [embedder.embed(token) for token in tokens]
    edges = []
    senders = []
    receivers = []
    for sender, sender_position in tqdm(enumerate(positions, start=0)):
        neighborhood = define_neighborhood(sender_position)
        for receiver, receiver_position in enumerate(positions, start=0):
            if sender == receiver:
                continue
            elif sender_position == receiver_position:
                edges.append(calculate_edge(sender_position, receiver_position, identical=True))
                senders.append(sender)
                receivers.append(receiver)

            elif is_overlapping(neighborhood, receiver_position):
                edges.append(calculate_edge(sender_position, receiver_position, identical=False))
                senders.append(sender)
                receivers.append(receiver)

    return [globals, nodes, edges, senders, receivers]


def convert_example_to_features(image: Image,
                                words: List,
                                boxes: List,
                                actual_boxes: List,
                                tokenizer: LayoutLMTokenizer,
                                max_seq_length: int = 512,
                                cls_token_box: List = [0, 0, 0, 0],
                                sep_token_box: List = [1000, 1000, 1000, 1000],
                                pad_token_box: List = [0, 0, 0, 0]) -> ():
    """
    converts lists of words, scaled and raw bboxes from a scanned receipt image to lists of token and boxes so you can
    use them as input for LayoutLM

    :param image: a PIL image of the receipt
    :param words: list of words from OCR data of the receipt
    :param boxes: list of normalized bounding boxes from OCR data of the receipt
    :param actual_boxes: list of raw bounding boxes from OCR data of the receipt
    :param tokenizer: instance of LayoutLM Tokenizer
    :param max_seq_length: maximum sequence length to trunctuate to
    :param cls_token_box: default bounding box to [CLS] token
    :param sep_token_box: default bounding box to [SEP] token
    :param pad_token_box: default bounding box to [PAD] token
    :return:
        Tuple containing:
            input_ids, a list of input ids for each token
            input_mask,  covering added [cls], [sep] and [pad] token
            segment_ids, same as input_ids (in theory for segments, but we dont have those, only token)
            token_boxes as new list of normalized boxes for tokenized words
            token_actual_boxes as new list of actual  boxes for tokenized words
    """
    width, height = image.size
    tokens = []
    token_boxes = []
    token_actual_boxes = []
    for word, box, actual_bbox in zip(words, boxes, actual_boxes):
        # splitting word in tokens if necessary
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
        token_actual_boxes.extend([actual_bbox] * len(word_tokens))

    # Truncation: account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
        token_actual_boxes = token_actual_boxes[: (max_seq_length - special_tokens_count)]

    # add [SEP] token, with corresponding token boxes and actual boxes
    tokens += [tokenizer.sep_token]
    token_boxes += [sep_token_box]
    token_actual_boxes += [[0, 0, width, height]]

    segment_ids = [0] * len(tokens)

    # next: [CLS] token
    tokens = [tokenizer.cls_token] + tokens
    token_boxes = [cls_token_box] + token_boxes
    token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
    segment_ids = [1] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    input_mask += [0] * padding_length
    segment_ids += [tokenizer.pad_token_id] * padding_length
    token_boxes += [pad_token_box] * padding_length
    token_actual_boxes += [pad_token_box] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(token_boxes) == max_seq_length
    assert len(token_actual_boxes) == max_seq_length

    return input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes


def convert_to_padded_graph(words: List,
                            boxes: List,
                            embedder: Embedder,
                            pad_to_nodes: int = 256,
                            pad_to_edges: int = 25000) -> List:
    """
    Construct a graph from given words and positions and pad it to required length for nodes and edges
    :param words: List of words
    :param boxes: List of bboxes for words like [[x1,y1,x2,y2], ... , [x1,y1,x2,y2]]
    :param embedder: instance of an embedder object to embedd tokens
    :param pad_to_nodes: desired number of nodes to pad to
    :param pad_to_edges: desired number of edges to pad to
    :return:
        a List of 4 numpy arrays consisting the graph. No globals are calculcated.
        the list contains:
            nodes
            edges
            senders
            receivers
    """
    graph = build_graph(boxes, words, embedder, include_globals=False)
    print(len(graph))
    print(len(graph[1]))
    print(len(graph[2]))
    print(len(graph[3]))
    print(len(graph[4]))
    x = [np.full((1, pad_to_nodes, embedder.get_embedding_size()), -1, dtype='float32'),
         np.full((1, pad_to_edges, 5), -1, dtype='float32'),
         np.full((1, pad_to_edges), -1, dtype='int32'),
         np.full((1, pad_to_edges), -1, dtype='int32')]
    x[0][0][0:len(graph[1])] = graph[1]  # creating batch of nodes
    x[1][0][0:len(graph[2])] = graph[2]  # creating batch of edges
    x[2][0][0:len(graph[3])] = graph[3]  # creating batch of senders
    x[3][0][0:len(graph[4])] = graph[4]  # creating batch of receivers

    return x


def normalize_box(box: List, width: int, height: int) -> List:
    """normalize a given box using given width and height"""
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def normalize_box_for_graphs(box: List, max_x_pos: int, max_y_pos: int) -> List:
    """normalize a given box using given max_x_pos and max_y_pos.
    Specifically for graphs because they work on float instead of int"""
    return [
        1000 * float(box[0]) / max_x_pos,
        1000 * float(box[1]) / max_y_pos,
        1000 * float(box[2]) / max_x_pos,
        1000 * float(box[3]) / max_y_pos
    ]
