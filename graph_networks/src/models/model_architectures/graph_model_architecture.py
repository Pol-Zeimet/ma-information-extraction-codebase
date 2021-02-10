from graph_nets import blocks
from graph_nets import graphs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .customized_crf_model import ModelWithCRFLoss
from tf2crf import CRF


class GraphConvV2(keras.layers.Layer):
    def __init__(self, input_units, intermediate_units, node_shape, edge_shape, reducer_type, **kwargs, ):
        super(GraphConvV2, self).__init__(**kwargs)
        self.node_shape = node_shape
        self.edges_shape = edge_shape
        self.reducer_type = reducer_type
        self.supports_masking = True

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

        num_previous_accumulated_nodes_per_graph = tf.range(0, batch_size) * num_nodes
        offsets = tf.tile(num_previous_accumulated_nodes_per_graph[:, None], [1, num_edges])
        offsets = tf.reshape(offsets, [-1])

        senders = tf.reshape(senders, [batch_size * num_edges])
        receivers = tf.reshape(receivers, [batch_size * num_edges])

        senders_offset = tf.where(senders != -1, senders + offsets, senders)
        receivers_offset = tf.where(receivers != -1, receivers + offsets, receivers)

        combined_graphs_tuple = graphs.GraphsTuple(n_node=tf.fill([batch_size], num_nodes),
                                                   n_edge=tf.fill([batch_size], num_edges),
                                                   nodes=tf.reshape(nodes, [batch_size * num_nodes, self.node_shape]),
                                                   edges=tf.reshape(edges, [batch_size * num_edges, self.edges_shape]),
                                                   senders=senders_offset,
                                                   receivers=receivers_offset,
                                                   globals=None,
                                                   )

        combined_graphs_tuple = combined_graphs_tuple.replace(
            edges=self.node_layer_out(
                self.node_layer_in(tf.concat([blocks.broadcast_receiver_nodes_to_edges(combined_graphs_tuple),
                                              combined_graphs_tuple.edges,
                                              blocks.broadcast_sender_nodes_to_edges(combined_graphs_tuple)], axis=1))))

        if self.reducer_type == "mean":
            reducer = tf.math.unsorted_segment_mean

        combined_graphs_tuple = combined_graphs_tuple.replace(
            nodes=blocks.ReceivedEdgesToNodesAggregator(reducer=reducer)(combined_graphs_tuple))

        combined_graphs_tuple = combined_graphs_tuple.replace(
            edges=self.edge_layer(combined_graphs_tuple.edges))

        return tf.reshape(combined_graphs_tuple.nodes, [batch_size, num_nodes, self.node_shape]), \
               tf.reshape(combined_graphs_tuple.edges, [batch_size, num_edges, self.edges_shape]), \
               senders, \
               receivers

    def compute_output_shape(self, input_shape):
        return input_shape


class GraphModelSoftmax:
    @staticmethod
    def create(node_count, edge_count, node_vector_length=768, edge_vector_length=5, n_folds=2, num_classes=5,
               reducer_type="mean", input_units=768, intermediate_units=768, bilstm_units=64, optimizer=None):
        # input
        nodes_input = keras.Input(shape=(node_count, node_vector_length), name='nodes_input')
        edges_input = keras.Input(shape=(edge_count, edge_vector_length), name='edges_input')
        senders_input = keras.Input(shape=edge_count, dtype='int32', name='senders_input')
        receivers_input = keras.Input(shape=edge_count, dtype='int32', name='receivers_input')

        # conv_layer
        graph_conv_layer = GraphConvV2(input_units,
                                       intermediate_units,
                                       node_vector_length,
                                       edge_vector_length,
                                       reducer_type=reducer_type,
                                       name='graph_conv')

        masking = keras.layers.Masking(name='masking')
        bilstm = layers.Bidirectional(layers.LSTM(bilstm_units, return_sequences=True), name='bilstm', )
        # activation =  layers.Activation('relu', name = 'relu_activation')
        pre_output_layer = layers.TimeDistributed(layers.Dense(64, activation='relu'))
        output_layer = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'), name='outputlayer')

        # graph convolution
        nodes = nodes_input
        edges = edges_input
        senders = senders_input
        receivers = receivers_input
        mask = masking.compute_mask(nodes)

        for fold in range(n_folds):
            nodes, edges, new_senders, receivers = graph_conv_layer(nodes, edges, senders, receivers)

        # sequence labeling
        graph_embeddings = nodes

        sequence = bilstm(nodes, mask=mask)
        # activated_sequence = activation(sequence)
        pre_output = pre_output_layer(sequence)

        output = output_layer(pre_output)

        model = keras.Model(inputs=[nodes_input, edges_input, senders_input, receivers_input],
                            outputs=[output, graph_embeddings, mask])
        model.compile(optimizer=optimizer, loss={'outputlayer': 'categorical_crossentropy'},
                      metrics={'outputlayer': "accuracy"})
        model.summary()
        return model


class GraphModelCRF:
    @staticmethod
    def create(node_count, edge_count, node_vector_length=768, edge_vector_length=5, n_folds=2, num_classes=5,
               reducer_type="mean", input_units=768, intermediate_units=768, bilstm_units=64, optimizer=None):
        # input
        nodes_input = keras.Input(shape=(node_count, node_vector_length), name='nodes_input')
        edges_input = keras.Input(shape=(edge_count, edge_vector_length), name='edges_input')
        senders_input = keras.Input(shape=edge_count, dtype='int32', name='senders_input')
        receivers_input = keras.Input(shape=edge_count, dtype='int32', name='receivers_input')

        # conv_layer
        graph_conv_layer = GraphConvV2(input_units,
                                       intermediate_units,
                                       node_vector_length,
                                       edge_vector_length,
                                       reducer_type=reducer_type,
                                       name='graph_conv')

        masking = keras.layers.Masking(name='masking')
        bilstm = layers.Bidirectional(layers.LSTM(bilstm_units, return_sequences=True), name='bilstm')
        # activation = layers.TimeDistributed(layers.Activation('relu'), name='relu_activation')
        pre_crf_layer_1 = layers.TimeDistributed(layers.Dense(64, activation='relu'), name='pre_crf_layer_1')
        pre_crf_layer_2 = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'),
                                                 name='pre_crf_layer_2')
        crf_layer = CRF(name='crf_layer')

        # graph convolution
        nodes = nodes_input
        edges = edges_input
        senders = senders_input
        receivers = receivers_input
        mask = masking.compute_mask(nodes)

        for i in range(0, n_folds):
            nodes, edges, senders, receivers = graph_conv_layer(nodes, edges, senders, receivers)

        graph_embeddings = nodes
        # sequence labeling
        masked_nodes = masking(nodes)
        sequence = bilstm(masked_nodes, mask=mask)

        # activated_sequence = activation(sequence)
        latent_activated_sequence = pre_crf_layer_1(sequence)
        latent_activated_sequence = pre_crf_layer_2(latent_activated_sequence)

        masked_sequence = masking(latent_activated_sequence)
        mask = masking.compute_mask(masked_sequence)
        output = crf_layer(masked_sequence, mask=mask)

        model = keras.Model([nodes_input, edges_input, senders_input, receivers_input], [output, graph_embeddings])
        model.summary()
        model = ModelWithCRFLoss(model)
        model.compile(optimizer=optimizer, metrics=["accuracy"])
        return model


class GraphModelCRFv2:
    @staticmethod
    def create(node_count, edge_count, node_vector_length=768, edge_vector_length=5, n_folds=2, num_classes=5,
               reducer_type="mean", input_units=768, intermediate_units=768, bilstm_units=64, optimizer=None):
        # input
        nodes_input = keras.Input(shape=(node_count, node_vector_length), name='nodes_input')
        edges_input = keras.Input(shape=(edge_count, edge_vector_length), name='edges_input')
        senders_input = keras.Input(shape=edge_count, dtype='int32', name='senders_input')
        receivers_input = keras.Input(shape=edge_count, dtype='int32', name='receivers_input')

        # conv_layer
        graph_conv_layer = GraphConvV2(input_units,
                                       intermediate_units,
                                       node_vector_length,
                                       edge_vector_length,
                                       reducer_type=reducer_type,
                                       name='graph_conv')

        masking = keras.layers.Masking(name='masking')
        bilstm = layers.Bidirectional(layers.LSTM(bilstm_units, return_sequences=True), name='bilstm')
        # activation = layers.TimeDistributed(layers.Activation('relu'), name='relu_activation')
        pre_crf_layer_1 = layers.TimeDistributed(layers.Dense(64, activation='relu'), name='pre_crf_layer_1')
        pre_crf_layer_2 = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'),
                                                 name='pre_crf_layer_2')

        crf_layer = CRF(name='crf_layer')

        # graph convolution
        nodes = nodes_input
        edges = edges_input
        senders = senders_input
        receivers = receivers_input

        mask = masking.compute_mask(nodes)

        for i in range(0, n_folds):
            nodes, edges, new_senders, receivers = graph_conv_layer(nodes, edges, senders, receivers)

        graph_embeddings = nodes

        # sequence labeling
        sequence = bilstm(nodes, mask=mask)
        # activated_sequence = activation(sequence)
        latent_activated_sequence = pre_crf_layer_1(sequence)
        latent_activated_sequence = pre_crf_layer_2(latent_activated_sequence)
        output = crf_layer(latent_activated_sequence, mask=mask)
        model = keras.Model([nodes_input, edges_input, senders_input, receivers_input], [output, graph_embeddings])
        model.summary()
        model = ModelWithCRFLoss(model)
        model.compile(optimizer=optimizer, metrics=["accuracy"], run_eagerly=True)
        return model
