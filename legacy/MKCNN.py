import os
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import StandardScaler

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


class MKCNN:
    def __init__(self, model_name):
        self.model_name = model_name

        self.X_train, self.y_train = np.load(Config.DATAPATH+"data_train.npy"), np.load(Config.DATAPATH+"labels_train.npy")
        self.X_test, self.y_test = np.load(Config.DATAPATH+"data_test.npy"), np.load(Config.DATAPATH+"labels_test.npy")

        self.X_train = self.normalization(self.X_train.astype('float32'))
        self.X_test = self.normalization(self.X_test.astype('float32'))

        model_input = Input(shape=(Config.MAX_TIMESTEP, ))
        conv_blocks = []
        for sz in [(i+1)*Config.UNIT_TIMESTEP for i in range(Config.N_INTVL)]:
            conv = Conv1D(filters = 3,
                kernel_size=sz,
                padding = "valid",
                activation="relu",)(model_input)
            conv = GlobalMaxPooling1D()(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        z = Dropout(0.5)(z)
        z = Dense(128, activation='relu')(z)
        model_output = Dense(1, activation="sigmoid")(z)

        model = Model(model_input, model_output)
        model.compile(loss="binary_crossentropy",
                        optimizer="adam",
                        metrics=['accuracy'])

    def train(self):
        hist = self.model.fit(self.X_train, self.y_train,
                                validation_split=0.2,
                                epochs=Config.EPOCHS,)
        self.model.save(self.model_name)

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
        plt.savefig('figures/Confusion_matrix.png')
        plt.show()
        
        return (loss, acc, precision, recall, f1)

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
    mkcnn = MKCNN(Config.MODEL_NAME)
    mkcnn.train()
    mkcnn.test()

