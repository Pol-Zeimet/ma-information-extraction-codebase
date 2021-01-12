from tensorflow import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    def __init__(self, graph_src, label_src, labels, slugs, batch_size=32, node_count=512, node_shape=768,
                 edge_count=60000, edge_shape=5, shuffle=True, one_hot=False):
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
        self.labels = np.asarray(labels)
        self.failed_loads = []
        self.on_epoch_end()

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

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.index_list = np.arange(len(self.list_slugs))
        if self.shuffle == True:
            np.random.shuffle(self.index_list)

    def __data_generation(self, list_slugs_temp):
        # Generates data containing batch_size samples
        x = [np.zeros(shape=(self.batch_size, self.node_count, self.node_shape)),
             np.zeros(shape=(self.batch_size, self.edge_count, self.edge_shape)),
             np.zeros(shape=(self.batch_size, self.edge_count), dtype='int32'),
             np.zeros(shape=(self.batch_size, self.edge_count), dtype='int32')]

        y = np.zeros(shape=(self.batch_size, self.node_count), dtype='float')

        # Generate data
        for i, slug in enumerate(list_slugs_temp):
            try:
                # laden Graph
                graph = np.load(self.graph_src + slug + '.npy', allow_pickle=True)

                i  # i is batch index
                x[0][i][0:len(graph[1])] = graph[1]  # creating batch of nodes
                x[1][i][0:len(graph[2])] = graph[2]  # creating batch of edges

                x[2][i][0:len(graph[3])] = graph[3]  # creating batch of senders
                x[2][i][len(graph[3]):] = -1  # setting senders for added edges to -1

                x[3][i][0:len(graph[4])] = graph[4]  # creating batch of receivers
                x[3][i][len(graph[4]):] = -1  # setting receivers for added edges to -1

                # load Labels
                i_labels = np.load(self.label_src + slug + '.npy', allow_pickle=True)
                y[i][0:len(i_labels)] = [np.argwhere(self.labels == label)[0][0] for label in i_labels]
            except RuntimeError as error:
                print('error with file: ' + slug)
                print(error)
                self.failed_loads.append(slug)

        if self.one_hot:
            y = np.asarray([keras.utils.to_categorical(y[i], num_classes=5, dtype='float') for i in
                            range(self.batch_size)])

        return x, y


class DataGeneratorReducedLabels(DataGenerator):
    def __init__(self, graph_src, label_src, labels, slugs, batch_size=32, node_count=512, node_shape=768,
                 edge_count=60000, edge_shape=5, shuffle=True, one_hot=False):
        super(DataGeneratorReducedLabels, self).__init__(graph_src,
                                                         label_src,
                                                         batch_size,
                                                         slugs,
                                                         shuffle,
                                                         node_count,
                                                         node_shape,
                                                         edge_count,
                                                         edge_shape,
                                                         labels,
                                                         one_hot)

    def __getitem__(self, index):
        super.__getitem__()

    def on_epoch_end(self):
        super.on_epoch_end()

    def __data_generation(self, list_slugs_temp):
        # Generates data containing batch_size samples
        x = [np.zeros(shape=(self.batch_size, self.node_count, self.node_shape)),
             np.zeros(shape=(self.batch_size, self.edge_count, self.edge_shape)),
             np.zeros(shape=(self.batch_size, self.edge_count), dtype='int32'),
             np.zeros(shape=(self.batch_size, self.edge_count), dtype='int32')]

        y_temp = np.full((self.batch_size, self.node_count), -1, dtype='int32')

        # Generate data
        for i, slug in enumerate(list_slugs_temp):
            try:
                # laden Graph
                graph = np.load(self.graph_src + slug + '.npy', allow_pickle=True)

                i  # i is batch index
                x[0][i][0:len(graph[1])] = graph[1]  # creating batch of nodes
                x[1][i][0:len(graph[2])] = graph[2]  # creating batch of edges

                x[2][i][0:len(graph[3])] = graph[3]  # creating batch of senders
                x[2][i][len(graph[3]):] = -1  # setting senders for added edges to -1

                x[3][i][0:len(graph[4])] = graph[4]  # creating batch of receivers
                x[3][i][len(graph[4]):] = -1  # setting receivers for added edges to -1

                # load Labels
                i_labels = np.load(self.label_src + slug + '.npy', allow_pickle=True)
                y_temp[i][0:len(i_labels)] = [np.argwhere(self.labels == label)[0][0] for label in i_labels]
                y_temp[i] = np.where((y_temp[i] == 1) | (y_temp[i] == 2) | (y_temp[i] == 3), 1, y_temp[i])
                y_temp[i] = np.where((y_temp[i] == 4) | (y_temp[i] == 5) | (y_temp[i] == 6), 2, y_temp[i])
                y_temp[i] = np.where((y_temp[i] == 7) | (y_temp[i] == 8) | (y_temp[i] == 9), 3, y_temp[i])
                y_temp[i] = np.where((y_temp[i] == 10) | (y_temp[i] == 11) | (y_temp[i] == 12), 4, y_temp[i])

            except RuntimeError as error:
                print('error with file: ' + slug)
                print(error)
                self.failed_loads.append(slug)

        if self.one_hot:
            y = np.asarray([keras.utils.to_categorical(y_temp[i], num_classes=5, dtype='float') for i in
                            range(self.batch_size)])
        else:
            y = y_temp

        return x, y