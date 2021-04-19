import statistics
import numpy as np
from tqdm import tqdm
from .Embedder import Embedder


def is_overlapping(box1, box2):
    # boxshape in our case: [x1,y1,x2,y2]
    return (box1[2] >= box2[0] and box2[2] >= box1[0]) and (box1[3] >= box2[1] and box2[3] >= box1[1])


def define_neighborhood(box):
    # boxshape in our case: [x1,y1,x2,y2]
    return [box[0] - 100, box[1] - 80, box[2] + 20, box[3] + 20]


def find_max_positions(positions):
    # positions shape = [[x1,y1,..., x4,y4], ...,[x1,y1,..., x4,y4]] with clockwise corner positions starting top left
    max_x = max([float(box[2]) for box in positions])
    max_y = max([float(box[3]) for box in positions])
    return max_x, max_y


def calculate_edge(sender_pos, receiver_pos, identical):
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
    if label != 'O':
        return label[2:]
    else:
        return "other"


def build_graph(positions, tokens, embedder:Embedder, include_globals=False):
    if include_globals:
        globals = embedder.embed(''.join(tokens), is_split_into_words=False, truncation=True)

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
    if include_globals:
        return [globals, nodes, edges, senders, receivers]
    else:
        return [nodes, edges, senders, receivers]


def convert_example_to_features(image, words, boxes, actual_boxes, tokenizer, args,
                                cls_token_box=[0, 0, 0, 0],
                                sep_token_box=[1000, 1000, 1000, 1000],
                                pad_token_box=[0, 0, 0, 0]):
    width, height = image.size

    tokens = []
    token_boxes = []
    token_actual_boxes = []
    for word, box, actual_bbox in zip(words, boxes, actual_boxes):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
        token_actual_boxes.extend([actual_bbox] * len(word_tokens))

    # Truncation: account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    if len(tokens) > args.max_seq_length - special_tokens_count:
        tokens = tokens[: (args.max_seq_length - special_tokens_count)]
        token_boxes = token_boxes[: (args.max_seq_length - special_tokens_count)]
        token_actual_boxes = token_actual_boxes[: (args.max_seq_length - special_tokens_count)]

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
    padding_length = args.max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    input_mask += [0] * padding_length
    segment_ids += [tokenizer.pad_token_id] * padding_length
    token_boxes += [pad_token_box] * padding_length
    token_actual_boxes += [pad_token_box] * padding_length

    assert len(input_ids) == args.max_seq_length
    assert len(input_mask) == args.max_seq_length
    assert len(segment_ids) == args.max_seq_length
    # assert len(label_ids) == args.max_seq_length
    assert len(token_boxes) == args.max_seq_length
    assert len(token_actual_boxes) == args.max_seq_length

    return input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes


def convert_to_padded_graph(words, boxes, embedder:Embedder, pad_to_nodes=256, pad_to_edges=25000, embedding_size=768, ):
    graph = build_graph(boxes, words, embedder, include_globals=False)
    print(len(graph))
    print(len(graph[0]))
    print(len(graph[1]))
    print(len(graph[2]))
    print(len(graph[3]))
    x = [np.zeros(shape=(1, pad_to_nodes, embedding_size), ),
         np.zeros(shape=(1, pad_to_edges, 5)),
         np.zeros(shape=(1, pad_to_edges), dtype='int32'),
         np.zeros(shape=(1, pad_to_edges), dtype='int32')]

    x[0][0][0:len(graph[0])] = graph[0]  # creating batch of nodes
    x[1][0][0:len(graph[1])] = graph[1]  # creating batch of edges
    x[2][0][0:len(graph[2])] = graph[2]  # creating batch of senders
    x[2][0][len(graph[2]):] = -1  # setting senders for added edges to -1
    x[3][0][0:len(graph[3])] = graph[3]  # creating batch of receivers
    x[3][0][len(graph[3]):] = -1  # setting receivers for added edges to -1

    return x


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def normalize_box_for_graphs(box, max_x_pos, max_y_pos):
    return [
        1000 * float(box[0]) / max_x_pos,
        1000 * float(box[1]) / max_y_pos,
        1000 * float(box[2]) / max_x_pos,
        1000 * float(box[3]) / max_y_pos
    ]
