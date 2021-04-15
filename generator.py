import numpy as np
import tensorflow as tf


from config import Config


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, dim=12, batch_size=32, 
                 n_channels=1, n_classes=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.y_true = []

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X[i] = np.load(Config.DATAPATH + ID).reshape(*self.dim)
            y[i] = self.labels[int(ID[:-4])]
        self.y_true.extend(y)

        #Y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, y