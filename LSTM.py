import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import Config


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
  except RuntimeError as e:
    print(e)


class LSTM(tf.keras.Model):
    def __init__(self, modelname):
        super(LSTM, self).__init__()
        self.modelname = modelname

        self.X_train = np.load(Config.DATAPATH+"data_train.npy").astype('float32')
        self.X_test = np.load(Config.DATAPATH+"data_test.npy").astype('float32')

        self.X_train = self.normalization(self.X_train)
        self.X_test = self.normalization(self.X_test)
    
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, activation='relu', input_shape=(Config.TIMESTEP, Config.N_FEATURES), return_sequences=False),
            tf.keras.layers.RepeatVector(Config.TIMESTEP),
        ])
    
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, activation='relu', input_shape=(Config.TIMESTEP, 32), return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(Config.N_FEATURES)),
        ])

        self.optimizer = tf.keras.optimizers.Adam(1e-4)
    
    def compute_loss(self, x):
        y = self.encoder(x)
        x_ = self.decoder(y)
        loss = tf.reduce_mean(tf.square(tf.abs(x - x_)))

        return loss
    
    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def train(self, epochs=5):
        self.X_train = tf.data.Dataset.from_tensor_slices(self.X_train).batch(Config.BATCH_SIZE)
        for epoch in range(epochs):
            print("epoch: {} training".format(epoch))
            for batch in self.X_train:
                loss = self.train_step(batch)
            tf.print(loss)
        self.encoder.save(self.modelname[0])
        self.decoder.save(self.modelname[1])

    def show_test(self):
        x = self.X_test[:2]

        self.encoder = tf.keras.models.load_model(self.modelname[0])
        self.decoder = tf.keras.models.load_model(self.modelname[1])

        z = self.encoder(x, training=False)
        x_ = self.decoder(z, training=False)

        print("\nx: {}".format(np.array2string(np.array(x[0]), prefix="x: ",
                formatter={'float_kind':lambda x: "%7.4f" % x})))
        print("\nz: {}".format(np.array2string(np.array(z[0]), prefix="z: ",
                formatter={'float_kind':lambda z: "%7.4f" % z})))
        print("\nx_: {}".format(np.array2string(np.array(x_[0]), prefix="x_: ",
                formatter={'float_kind':lambda x: "%7.4f" % x})))

    def test(self):
        x = self.X_train

        self.encoder = tf.keras.models.load_model(self.modelname[0])
        self.decoder = tf.keras.models.load_model(self.modelname[1])

        z = self.encoder(x, training=False)
        #x_ = self.decoder(z, training=False)

        np.save(Config.DATAPATH+"features_train", z)
    
    def normalization(self, X):
        scaler = StandardScaler().fit(self.flatten(X))

        return self.scale(X, scaler)

    def flatten(self, X):
        flattened_X = np.empty((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1]-1), :]

        return (flattened_X)

    def scale(self, X, scaler):
        for i in range(X.shape[0]):
            X[i, :, :] = scaler.transform(X[i, :, :])

        return X


if __name__ == "__main__":
    lstm = LSTM(Config.MODELNAME)
    #lstm.train(Config.EPOCHS)
    lstm.show_test()
    lstm.test()
