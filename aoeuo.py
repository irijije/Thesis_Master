import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score

from preprocess import load_dataset
from metadata import LABELS


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
AE_ENCODER_NAME = "models/ae_encoder.h5"
AE_DECODER_NAME = "models/ae_decoder.h5"
BUFFER_SIZE = 60000
BATCH_SIZE = 1024
test_size = 10000


class DenseTranspose(tf.keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.biases = self.add_weight(
            name="bias",
            shape=[self.dense.input_shape[-1]],
            initializer="zeros",
        )
        super().build(batch_input_shape)
    
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)

        return self.activation(z + self.biases)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dense': self.dense,
            'activation': self.activation,
        })
        
        return config


class AE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(AE, self).__init__()
        self.latent_dim = latent_dim

        self.train_x, self.train_y = load_dataset("dataset/CICIDS2018_mal_train.csv")
        self.test_x, self.test_y = load_dataset("dataset/CICIDS2018_mal_test.csv")

        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_x.values).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


        dense1 = tf.keras.layers.Dense(78, activation='relu')
        dense2 = tf.keras.layers.Dense(32, activation='relu')
        dense3 = tf.keras.layers.Dense(latent_dim, activation='relu')

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(78,)),
            dense1,
            dense2,
            dense3,
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            DenseTranspose(dense3, activation='relu'),
            DenseTranspose(dense2, activation='relu'),
            DenseTranspose(dense1, activation='sigmoid')
        ])

        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def compute_loss(self, x):
        # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        y = self.encoder(x)
        x_ = self.decoder(y)
        # loss = cross_entropy(x, x_)
        loss = tf.reduce_mean(tf.square(tf.abs(x - x_)))

        return loss
    
    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

    def train(self, epochs=5, filename=[AE_ENCODER_NAME, AE_DECODER_NAME]):
        losses = []
        for epoch in range(epochs):
            print("epoch: {} training".format(epoch))
            for batch in self.train_dataset:
                loss = self.train_step(batch)
            losses.append(loss)
        print("loss: ", end="")
        tf.print(loss)
        self.encoder.save(filename[0])
        self.decoder.save(filename[1])

        return np.array(loss)

    def show_test(self):
        x, _ = load_dataset("test.csv")

        y = self.encoder(np.array(x), training=False)
        x_ = self.decoder(y, training=False)

        print("\nx: {}".format(np.array2string(np.array(x), prefix="x: ",
                formatter={'float_kind':lambda x: "%7.4f" % x})))
        print("\nx_: {}".format(np.array2string(np.array(x_), prefix="x_: ",
                formatter={'float_kind':lambda x: "%7.4f" % x})))


if __name__ == "__main__":
    ae = AE(10)
    loss = ae.train(5)
    ae.show_test()