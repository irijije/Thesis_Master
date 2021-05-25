import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tools import *
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


class CNN:
    def __init__(self):
        self.X_train, self.y_train, self.X_test, self.y_test = load_data()

        show_tsne(self.X_train[:100], self.y_train[:100])

        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train, test_size=0.2)

        self.X_train = self.X_train.reshape(list(self.X_train.shape)+[1])
        self.X_test = self.X_test.reshape(list(self.X_test.shape)+[1])
        print(self.X_train.shape)
        print(self.X_test.shape)

        if Config.isMC:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.X_train.shape[1], self.X_train.shape[2], 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(5, activation='softmax'),
            ])
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.X_train.shape[1], self.X_train.shape[2], 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid'),
            ])
            loss = tf.keras.losses.BinaryCrossentropy()

        self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(),
                    loss=loss,
                    metrics=['accuracy'],
        )

    def train(self):
        hist = self.model.fit(self.X_train, self.y_train,
                                validation_split=0.2,
                                batch_size=Config.BATCH_SIZE,
                                epochs=Config.EPOCHS,)
        self.model.save(Config.MODEL_NAME)

        #show_train_result(hist)

    def test(self):
        self.model = tf.keras.models.load_model(Config.MODEL_NAME)
        y_pred = self.model.predict_classes(self.X_test)
        y_true = self.y_test
        _, acc = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        print(acc)
        
        show_test_result(y_true, y_pred)


if __name__ == "__main__":
    cnn = CNN()
    cnn.train()
    cnn.test()