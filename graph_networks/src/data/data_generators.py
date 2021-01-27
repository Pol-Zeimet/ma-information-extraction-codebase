from tensorflow import keras
import numpy as np
import os
import pandas as pd


class DataGenerator(keras.utils.Sequence):
    def __init__(self, graph_src, label_src, labels, slugs,additional_data_src, batch_size=32, node_count=512, node_shape=768,
                 edge_count=40000, edge_shape=5, shuffle=True, one_hot=False):
        # Initialization
        self.graph_src = graph_src
        self.label_src = label_src
        self.batch_size = batch_size
        self.list_slugs = slugs
        self.shuffle = shuffle
        self.node_count = node_count
        self.node_shape = node_shape
        self.edge_count = edge_count
        self.edge_shape = edge_shape
        self.labels = np.asarray(labels, dtype="object")
        self.failed_loads = []
        self.index_list = []
        self.on_epoch_end()
        self.one_hot = one_hot
        self.additional_data_df = pd.read_json(additional_data_src)
        self.max_X_pos, self.max_Y_pos = self.get_max_positions()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_slugs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.index_list[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of slugs
        list_slugs_temp = [self.list_slugs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_slugs_temp)

        return X, y


    def get_max_positions(self):
        # positions shape = [[x1,y1,..., x4,y4], ...,[x1,y1,..., x4,y4]]
        # with clockwise corner positions starting top left
        positions = self.additional_data_df.position
        max_x = max([float(box[5]) for box in positions])
        max_y = max([float(box[6]) for box in positions])
        return max_x, max_y

    def get_tokens_and_positions(self, index):
        indexes = self.index_list[index * self.batch_size:(index + 1) * self.batch_size]
        aggregated_tokens = []
        aggregated_positions = []
        for slug in [self.list_slugs[k] for k in indexes]:
            subset = self.additional_data_df[self.additional_data_df.slug == slug]
            aggregated_positions.append([[1000 * float(subset.position[i][0]) / self.max_X_pos,
                                          1000 * float(subset.position[i][1]) / self.max_Y_pos,
                                          1000 * float(subset.position[i][4]) / self.max_X_pos,
                                          1000 * float(subset.position[i][5]) / self.max_Y_pos]
                                         for i, row in subset.iterrows() for subset_token_i in subset.tokens[i]])
            aggregated_tokens.append([token for subset_tokens in subset.tokens for token in subset_tokens])
        return aggregated_tokens, aggregated_positions



    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.index_list = np.arange(len(self.list_slugs))
        if self.shuffle is True:
            np.random.shuffle(self.index_list)

    def __data_generation(self, list_slugs_temp):
        # Generates data containing batch_size samples
        x = [np.zeros(shape=(self.batch_size, self.node_count, self.node_shape)),
             np.zeros(shape=(self.batch_size, self.edge_count, self.edge_shape)),
             np.zeros(shape=(self.batch_size, self.edge_count), dtype='int32'),
             np.zeros(shape=(self.batch_size, self.edge_count), dtype='int32')]

        if self.one_hot:
            y = np.full((self.batch_size, self.node_count, len(self.labels)), 0, dtype='int32')

        else:
            y = np.full((self.batch_size, self.node_count), -1., dtype='int32')

        # Generate data
        for i, slug in enumerate(list_slugs_temp):
            try:
                # laden Graph
                graph = np.load(os.path.join(self.graph_src, slug + '.npy'), allow_pickle=True)

                i  # i is batch index
                x[0][i][0:len(graph[1])] = graph[1]  # creating batch of nodes
                x[1][i][0:len(graph[2])] = graph[2]  # creating batch of edges

                x[2][i][0:len(graph[3])] = graph[3]  # creating batch of senders
                x[2][i][len(graph[3]):] = -1  # setting senders for added edges to -1

                x[3][i][0:len(graph[4])] = graph[4]  # creating batch of receivers
                x[3][i][len(graph[4]):] = -1  # setting receivers for added edges to -1

                # load Labels
                i_labels = np.load(os.path.join(self.label_src, slug + '.npy'), allow_pickle=True)
                y_i = [np.argwhere(self.labels == label)[0][0] for label in i_labels]

                if self.one_hot:
                    y_i = np.asarray([keras.utils.to_categorical(y_i, num_classes=len(self.labels), dtype='float')
                                      for i in range(self.batch_size)], dtype="object")

                y[i][0:len(i_labels)] = y_i

            except RuntimeError as error:
                print('error with file: ' + slug)
                print(error)
                self.failed_loads.append(slug)

        return x, y


class DataGeneratorReducedLabels(keras.utils.Sequence):
    def __init__(self, graph_src, label_src, labels, slugs,additional_data_src, batch_size=32, node_count=512, node_shape=768,
                 edge_count=40000, edge_shape=5, shuffle=True, one_hot=False
                 ):
        # Initialization
        self.graph_src = graph_src
        self.label_src = label_src
        self.batch_size = batch_size
        self.list_slugs = slugs
        self.shuffle = shuffle
        self.node_count = node_count
        self.node_shape = node_shape
        self.edge_count = edge_count
        self.edge_shape = edge_shape
        self.labels = np.asarray(labels, dtype="object")
        self.failed_loads = []
        self.index_list = []
        self.on_epoch_end()
        self.one_hot = one_hot
        self.additional_data_df = pd.read_json(additional_data_src)
        self.max_X_pos, self.max_Y_pos = self.get_Max_positions()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_slugs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.index_list[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of slugs
        list_slugs_temp = [self.list_slugs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_slugs_temp)

        return X, y

    def get_Max_positions(self):
        # positions shape = [[x1,y1,..., x4,y4], ...,[x1,y1,..., x4,y4]]
        # with clockwise corner positions starting top left
        positions = self.additional_data_df.position
        max_X = max([float(box[5]) for box in positions])
        max_Y = max([float(box[6]) for box in positions])
        return max_X, max_Y

    def get_tokens_and_positions(self, index):
        indexes = self.index_list[index * self.batch_size:(index + 1) * self.batch_size]
        aggregated_tokens = []
        aggregated_positions = []
        for slug in [self.list_slugs[k] for k in indexes]:
            subset = self.additional_data_df[self.additional_data_df.slug == slug]
            aggregated_positions.append([[1000 * float(subset.position[i][0]) / self.max_X_pos,
                                          1000 * float(subset.position[i][1]) / self.max_Y_pos,
                                          1000 * float(subset.position[i][4]) / self.max_X_pos,
                                          1000 * float(subset.position[i][5]) / self.max_Y_pos]
                                         for i, row in subset.iterrows() for subset_token_i in subset.tokens[i]])
            aggregated_tokens.append([token for subset_tokens in subset.tokens for token in subset_tokens])
        return aggregated_tokens, aggregated_positions

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.index_list = np.arange(len(self.list_slugs))
        if self.shuffle is True:
            np.random.shuffle(self.index_list)

    def __data_generation(self, list_slugs_temp):
        # Generates data containing batch_size samples

        x = [np.zeros(shape=(self.batch_size, self.node_count, self.node_shape), ),
             np.zeros(shape=(self.batch_size, self.edge_count, self.edge_shape)),
             np.zeros(shape=(self.batch_size, self.edge_count), dtype='int32'),
             np.zeros(shape=(self.batch_size, self.edge_count), dtype='int32')]

        if self.one_hot:
            y = np.full((self.batch_size, self.node_count, 5), 0, dtype='int32')

        else:
            y = np.full((self.batch_size, self.node_count), -1., dtype='int32')

        # Generate data
        for i, slug in enumerate(list_slugs_temp):
            try:
                # laden Graph
                graph = np.load(os.path.join(self.graph_src, slug + '.npy'), allow_pickle=True)

                i  # i is batch index
                x[0][i][0:len(graph[1])] = graph[1]  # creating batch of nodes
                x[1][i][0:len(graph[2])] = graph[2]  # creating batch of edges

                x[2][i][0:len(graph[3])] = graph[3]  # creating batch of senders
                x[2][i][len(graph[3]):] = -1  # setting senders for added edges to -1

                x[3][i][0:len(graph[4])] = graph[4]  # creating batch of receivers
                x[3][i][len(graph[4]):] = -1  # setting receivers for added edges to -1

                # load Labels
                i_labels = np.load(os.path.join(self.label_src, slug + '.npy'), allow_pickle=True)
                y_i = np.asarray([np.argwhere(self.labels == label)[0][0] for label in i_labels], dtype="object")
                y_i = np.where((y_i == 1) | (y_i == 2) | (y_i == 3), 1, y_i)
                y_i = np.where((y_i == 4) | (y_i == 5) | (y_i == 6), 2, y_i)
                y_i = np.where((y_i == 7) | (y_i == 8) | (y_i == 9), 3, y_i)
                y_i = np.where((y_i == 10) | (y_i == 11) | (y_i == 12), 4, y_i)

                if self.one_hot:
                    y_i = keras.utils.to_categorical(y_i, num_classes=5, dtype='float')

                y[i][0:len(y_i)] = y_i

            except RuntimeError as error:
                print('error with file: ' + slug)
                print(error)
                self.failed_loads.append(slug)

        return x, y