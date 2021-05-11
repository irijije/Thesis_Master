import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score

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
    def __init__(self, model_name):
        self.model_name = model_name

        self.X_train, self.y_train = np.load(Config.DATAPATH+"data_train.npy"), np.load(Config.DATAPATH+"labels_train.npy")
        self.X_test, self.y_test = np.load(Config.DATAPATH+"data_test.npy"), np.load(Config.DATAPATH+"labels_test.npy")

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train.reshape(-1, self.X_train.shape[-1])).reshape(self.X_train.shape).astype('float32')
        self.X_test = scaler.transform(self.X_test.reshape(-1, self.X_test.shape[-1])).reshape(self.X_test.shape).astype('float32')

        self.X_train = self.X_train.reshape(list(self.X_train.shape)+[1])
        self.X_test = self.X_test.reshape(list(self.X_test.shape)+[1])

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

        self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(),
                    loss='binary_crossentropy',
                    metrics=['accuracy'],
        )

    def train(self):
        hist = self.model.fit(self.X_train, self.y_train,
                                batch_size=Config.BATCH_SIZE,
                                validation_split=0.2,
                                epochs=Config.EPOCHS,)
        self.model.save(self.model_name)

        plt.title("Trainning result")
        loss_ax = plt.gca()
        acc_ax = loss_ax.twinx()
        loss_ax.plot(hist.history['loss'], color='C0', linestyle='-', label='train loss')
        acc_ax.plot(hist.history['accuracy'], color='C1', linestyle='-', label='train acc')
        loss_ax.plot(hist.history['val_loss'], color='C2', linestyle='--', label='val loss')
        acc_ax.plot(hist.history['val_accuracy'], color='C3', linestyle='--', label='val acc')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')
        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        plt.grid(b=True, which='major', linestyle='--')
        plt.savefig('figures/trainning_result.png')
        plt.show()

    def test(self):
        self.model = tf.keras.models.load_model(self.model_name)
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
        plt.savefig('figures/confusion_matrix.png')
        plt.show()
        
        return (loss, acc, precision, recall, f1)


if __name__ == "__main__":
    cnn = CNN(Config.MODEL_NAME)
    cnn.train()
    cnn.test()