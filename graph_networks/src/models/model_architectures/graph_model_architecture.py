from graph_nets import blocks
from graph_nets import graphs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class graphConvV2(keras.layers.Layer):
    def __init__(self, input_units, intermediate_units, node_shape, edge_shape, **kwargs, ):
        super(graphConvV2, self).__init__(**kwargs)
        self.node_shape = node_shape
        self.edges_shape = edge_shape

        self.node_layer_in = layers.Dense(input_units, activation='relu', name='node_layer_in')
        self.intermediate_node_layer = layers.Dense(intermediate_units, activation='relu',
                                                    name='intermediate_node_layer')
        self.node_layer_out = layers.Dense(node_shape, activation='relu', name='node_layer_out')

        self.edge_layer = layers.Dense(edge_shape, activation='relu', name='edge_layer')

    def call(self, nodes, edges, senders, receivers):
        # building_graph
        nodes_shape = nodes.shape
        edges_shape = edges.shape
        batch_size = tf.shape(nodes)[0]

        num_nodes = tf.constant(nodes_shape[1])
        num_edges = tf.constant(edges_shape[1])
        assert self.node_shape == nodes.shape.as_list()[-1]
        assert self.edges_shape == edges.shape.as_list()[-1]

        combined_graphs_tuple = graphs.GraphsTuple(n_node=tf.fill([batch_size], num_nodes),
                                                   n_edge=tf.fill([batch_size], num_edges),
                                                   nodes=tf.reshape(nodes, [batch_size * num_nodes, self.node_shape]),
                                                   edges=tf.reshape(edges, [batch_size * num_edges, self.edges_shape]),
                                                   senders=tf.reshape(senders, [batch_size * num_edges]),
                                                   receivers=tf.reshape(receivers, [batch_size * num_edges]),
                                                   globals=None,
                                                   )

        # Schritt 1
        # Tripel auf Kanten erstellen
        a = blocks.broadcast_receiver_nodes_to_edges(combined_graphs_tuple)
        b = combined_graphs_tuple.edges
        c = blocks.broadcast_sender_nodes_to_edges(combined_graphs_tuple)
        d = tf.concat([a, b, c], axis=1)

        combined_graphs_tuple = combined_graphs_tuple.replace(
            edges=d
        )

        # Schritt 2
        # Tripel in MLP verarbeiten
        embeddings = self.node_layer_in(combined_graphs_tuple.edges)
        combined_graphs_tuple = combined_graphs_tuple.replace(
            edges=self.node_layer_out(embeddings))

        # schritt 3
        # hidden states auf knoten zu neuen Embeddings aggregieren
        # try different reducers
        reducer = tf.math.unsorted_segment_mean

        combined_graphs_tuple = combined_graphs_tuple.replace(
            nodes=blocks.ReceivedEdgesToNodesAggregator(reducer=reducer)(combined_graphs_tuple))

        # Schritt4
        # neuen wert für kante berechnen
        combined_graphs_tuple = combined_graphs_tuple.replace(
            edges=self.edge_layer(combined_graphs_tuple.edges))

        # Schritt5
        # Graphen wieder in Tensoren aufbrechen

        nodes = tf.reshape(combined_graphs_tuple.nodes, [batch_size, num_nodes, self.node_shape])
        edges = tf.reshape(combined_graphs_tuple.edges, [batch_size, num_edges, self.edges_shape])
        senders = tf.reshape(combined_graphs_tuple.senders, [batch_size, num_edges])
        receivers = tf.reshape(combined_graphs_tuple.receivers, [batch_size, num_edges])

        return nodes, edges, senders, receivers

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_units': self.input_units,
            'intermediate_units': self.intermediate_units,
            'node_shape': self.node_shape,
            'edge_shape': self.edge_shape
        })
        return config


class GraphModelSoftmax:
    @staticmethod
    def create(node_count, edge_count, node_vector_length=768, edge_vector_length=5, n_folds=2, num_classes=5):

        input_units = 768
        intermediate_units = 768

        # input
        nodes_input = keras.Input(shape=(node_count, node_vector_length), name='nodes_input')
        edges_input = keras.Input(shape=(edge_count, edge_vector_length), name='edges_input')
        senders_input = keras.Input(shape=edge_count, dtype='int32', name='senders_input')
        receivers_input = keras.Input(shape=edge_count, dtype='int32', name='receivers_input')

        # conv_layer
        graph_conv_layer = graphConvV2(input_units,
                                       intermediate_units,
                                       node_vector_length,
                                       edge_vector_length,
                                       name='graph_conv')

        masking = keras.layers.Masking(name='masking')
        bilstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name='bilstm', )
        activation = layers.Activation('relu', name='relu_activation')
        pre_output_layer = layers.TimeDistributed(layers.Dense(64, activation='relu'))
        output_layer = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))

        # graph convolution
        nodes = nodes_input
        edges = edges_input
        senders = senders_input
        receivers = receivers_input

        for i in range(0, n_folds):
            nodes, edges, new_senders, receivers = graph_conv_layer(nodes, edges, senders, receivers)

        # sequence labeling
        mask = masking.compute_mask(nodes)
        sequence = bilstm(nodes, mask=mask)
        activated_sequence = activation(sequence)
        pre_output = pre_output_layer(activated_sequence)
        output = output_layer(pre_output)

        return keras.Model([nodes_input, edges_input, senders_input, receivers_input], output)