import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

from config import Config
from generator import DataGenerator

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

        params = {'dim': (Config.TIMESTEP, Config.N_FEATURES),
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}

        data = os.listdir(Config.DATAPATH)
        data.remove('labels.npy')
        labels = np.load(Config.DATAPATH+"labels.npy")
        data_train = data[:int(len(data)/10*6)]
        data_valid = data[int(len(data)/10*6):int(len(data)/10*8)]
        data_test = data[int(len(data)/10*8):]
        self.gen_train = DataGenerator(data_train, labels, **params)
        self.gen_valid = DataGenerator(data_valid, labels, **params)
        self.gen_test = DataGenerator(data_test, labels, **params)

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

    def train(self, epochs=5, filename=['models/encoder.h5', 'models/decoder.h5']):
        for epoch in range(epochs):
            print("epoch: {} training".format(epoch))
            for batch_x, _ in self.gen_train:
                loss = self.train_step(tf.cast(batch_x, tf.float32))
            tf.print(loss)
        self.encoder.save(filename[0])
        self.decoder.save(filename[1])

    def show_test(self):
        x1 = np.load(Config.DATAPATH+"1.npy").astype('float32')
        x2 = np.load(Config.DATAPATH+"2.npy").astype('float32')

        x = np.array([x1, x2])

        y = self.encoder(x, training=False)
        x_ = self.decoder(y, training=False)

        print("\nx: {}".format(np.array2string(np.array(x), prefix="x: ",
                formatter={'float_kind':lambda x: "%7.4f" % x})))
        print("\ny: {}".format(np.array2string(np.array(y), prefix="y: ",
                formatter={'float_kind':lambda y: "%7.4f" % y})))
        print("\nx_: {}".format(np.array2string(np.array(x_), prefix="x_: ",
                formatter={'float_kind':lambda x: "%7.4f" % x})))

    def test(self):
        self.model = tf.keras.models.load_model(self.modelname)
        y_pred = self.model.predict(self.gen_test)
    

if __name__ == "__main__":
    lstm = LSTM(Config.MODELNAME)
    lstm.train()
    lstm.show_test()
    #lstm.test()
