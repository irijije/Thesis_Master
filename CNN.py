import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

from config import Config


class CNN:
    def __init__(self, modelname):
        self.modelname = modelname

        self.X_train, self.y_train = np.load(Config.DATAPATH+"features_train.npy"), np.load(Config.DATAPATH+"labels_train.npy")
        self.X_test, self.y_test = np.load(Config.DATAPATH+"features_test.npy"), np.load(Config.DATAPATH+"labels_test.npy")

        self.X_train = self.X_train.reshape(list(self.X_train.shape)+[1])
        self.X_test = self.X_test.reshape(list(self.X_test.shape)+[1])

        print(self.X_train.shape)


        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 29, 1)),
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

        self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'],
        )

    def show_result(self, hist):
        plt.title("Trainning result")
        loss_ax = plt.gca()
        acc_ax = loss_ax.twinx()
        loss_ax.plot(hist.history['loss'], 'C0', label='train loss')
        acc_ax.plot(hist.history['accuracy'], 'C1', label='train acc')
        loss_ax.plot(hist.history['val_loss'], 'C2', label='val loss')
        acc_ax.plot(hist.history['val_accuracy'], 'C3', label='val acc')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')
        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        plt.grid(b=True, which='major', linestyle='--')
        plt.savefig('figures/Trainning_result.png')
        plt.show()

    def train(self):
        hist = self.model.fit(self.X_train, self.y_train,
                                validation_split=0.2,
                                epochs=Config.EPOCHS,)
        self.model.save(self.modelname)
        self.show_result(hist)

    def test(self):
        self.model = tf.keras.models.load_model(self.modelname)
        y_pred = self.model.predict_classes(self.X_test)
        y_true = self.y_test
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        precision, recall, f1, _ = score(y_true, y_pred, zero_division=1)
        print(acc)
        print(precision)
        print(recall)
        
        conf_matrix = confusion_matrix(y_true, y_pred)

        labels = ["Normal", "Attack"]
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.savefig('figures/Confusion_matrix.png')
        plt.show()
        
        return (loss, acc, precision, recall, f1)


if __name__ == "__main__":
    cnn = CNN(Config.MODELNAME)
    cnn.train()
    cnn.test()