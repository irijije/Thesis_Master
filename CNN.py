import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score

from config import Config


class CNN:
    def __init__(self, modelname):
        self.modelname = modelname

        self.X_train, self.y_train = np.load(Config.DATAPATH+"features_train.npy"), np.load(Config.DATAPATH+"labels_train.npy")
        #self.X_vaild, self.y_vaild = np.load(Config.DATAPATH+"features_vaild.npy"), np.load(Config.DATAPATH+"labels_valid.npy")
        #self.X_test, self.y_test = np.load(Config.DATAPATH+"features_test.npy"), np.load(Config.DATAPATH+"labels_test.npy")
       
        self.X_train = self.X_train.reshape(list(self.X_train.shape)+[1])

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 32, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

        self.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'],
        )

    def show_result(self, hist):
        _, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()
        loss_ax.plot(hist.history['loss'], 'r', label='train loss')
        acc_ax.plot(hist.history['acc'], 'b', label='train acc')
        loss_ax.plot(hist.history['val_loss'], 'y', label='val loss')
        acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')
        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        plt.show()

    def train(self):
        hist = self.model.fit(self.X_train, self.y_train,
                                #validation_data=(self.X_valid, self.y_valid),
                                epochs=300,)
        self.model.save(self.modelname)
        self.show_result(hist)

    # def test(self):
    #     self.model = tf.keras.models.load_model(self.modelname)
    #     y_pred = self.model.predict_classes(self.X_test)
    #     y_true = self.y_test
    #     loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=2)
    #     precision, recall, f1, _ = score(y_true, y_pred, zero_division=1)
    #     print(acc)
    #     print(precision)
    #     print(recall)
        
    #     return (loss, acc, precision, recall, f1)


if __name__ == "__main__":
    cnn = CNN(Config.MODELNAME)
    cnn.train()
    #cnn.test()