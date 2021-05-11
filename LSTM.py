import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import Config
from preprocess import show_tsne


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
    def __init__(self, model_name):
        super(LSTM, self).__init__()
        self.model_name = model_name

        self.X_train, self.y_train = np.load(Config.DATAPATH+"data_train.npy"), np.load(Config.DATAPATH+"labels_train.npy")
        self.X_test, self.y_test = np.load(Config.DATAPATH+"data_test.npy"), np.load(Config.DATAPATH+"labels_test.npy")

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train.reshape(-1, self.X_train.shape[-1])).reshape(self.X_train.shape).astype('float32')
        self.X_test = scaler.transform(self.X_test.reshape(-1, self.X_test.shape[-1])).reshape(self.X_test.shape).astype('float32')

        # self.X_train = self.padding(self.X_train)
        # self.X_test = self.padding(self.X_test)
    
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, activation='relu', input_shape=(self.X_train.shape[1], self.X_train.shape[2]), return_sequences=False),
            tf.keras.layers.RepeatVector(self.X_train.shape[1]),
        ])
    
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, activation='relu', input_shape=(self.X_train.shape[1], 32), return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.X_train.shape[2])),
        ])

        self.optimizer = tf.keras.optimizers.Adam()
    
    def compute_loss(self, x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        loss = tf.reduce_mean(tf.square(tf.abs(x - x_)))

        return loss

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def train(self, epochs=Config.EPOCHS):
        self.X_train = tf.data.Dataset.from_tensor_slices(self.X_train).batch(Config.BATCH_SIZE)
        losses = []
        for epoch in range(epochs):
            print("epoch: {} training".format(epoch))
            for batch in self.X_train:
                #batch = self.padding(batch)
                loss = self.train_step(batch)
            losses.append(loss.numpy())
        self.encoder.save(self.model_name[0])
        self.decoder.save(self.model_name[1])

        plt.plot(losses, linewidth=2, label='Train')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig("figures/lstm_loss.png")
        plt.show()

    def show_test(self):
        x = self.X_test[:2]

        self.encoder = tf.keras.models.load_model(self.model_name[0])
        self.decoder = tf.keras.models.load_model(self.model_name[1])

        z = self.encoder(x, training=False)
        x_ = self.decoder(z, training=False)

        print("\nx: {}".format(np.array2string(np.array(x[0]), prefix="x: ",
                formatter={'float_kind':lambda x: "%7.4f" % x})))
        print("\nz: {}".format(np.array2string(np.array(z[0]), prefix="z: ",
                formatter={'float_kind':lambda z: "%7.4f" % z})))
        print("\nx_: {}".format(np.array2string(np.array(x_[0]), prefix="x_: ",
                formatter={'float_kind':lambda x: "%7.4f" % x})))

    def test(self, name='test'):
        if name=='train':
            x = self.X_train
            self.X_train = tf.data.Dataset.from_tensor_slices(self.X_train).batch(Config.BATCH_SIZE)
            y = self.y_train
        else:
            x = self.X_test
            self.X_test = tf.data.Dataset.from_tensor_slices(self.X_test).batch(Config.BATCH_SIZE)
            y = self.y_test

        self.encoder = tf.keras.models.load_model(self.model_name[0])
        self.decoder = tf.keras.models.load_model(self.model_name[1])

        z = self.encoder(x, training=False).numpy()
        x_ = self.decoder(z, training=False).numpy()

        np.save(Config.DATAPATH+f"features_{name}", z)
        show_tsne(z, y, 'lstm')

        mse = np.mean(np.power(x.reshape(x.shape[0], -1) - x_.reshape(x_.shape[0], -1), 2), axis=1)
        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                'True_class': y})
        groups = error_df.groupby('True_class')
        _, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                    label= "Attack" if name == 1 else "Normal")
        ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.savefig("figures/reconstruction_error.png")
        plt.show()

    def padding(self, X):
        T = int(Config.MAX_TIMESTEP/Config.UNIT_TIMESTEP)
        X_ = np.zeros((len(X)*T, Config.MAX_TIMESTEP, Config.N_FEATURES))
        for i, x in enumerate(X):
            for j in range(T):
                X_[i*T+j, T-j-1:] = x[T-j-1:]

        return X_.astype('float32')


if __name__ == "__main__":
    lstm = LSTM(Config.MODEL_NAME)
    #lstm.train()
    #lstm.show_test()
    lstm.test()
