import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
import sklearn.manifold as manifold
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import StandardScaler
from config import Config



def test():
    df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(6)], header=None)
    df = df[['0', '1', '4']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)
    df['4'] = df['4'].apply(lambda x: 1 if x == 'Attack' else 0)

    Y = df.pop('4')
    X = df

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)
    X_a, X_b, Y_a, Y_b = train_test_split(X, Y, test_size = 0.0001, random_state = 100)
    
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print(X_train[0:5])
    
    C = 0.01
    
    clf = svm.SVC(kernel = 'linear', C=C, max_iter=10000)
    #clf = svm.SVC(kernel = 'rbf', gamma=0.1, C=C, max_iter=10000)
    #clf = svm.SVC(kernel = 'poly', degree=5, C=C, max_iter=1000)
    clf.fit(X_train, Y_train)

    Y_ = clf.predict(X_test)
    acc = clf.score(X_test, Y_test)*100

    precision, recall, _, _ = score(Y_test, Y_, zero_division=1)

    print(f'svm poly {acc} {precision} {recall}')

    tsneNDArray = manifold.TSNE(n_components = 2, init = "pca", random_state = 0).fit_transform(X_b)

    figure, axesSubplot = pp.subplots()
    axesSubplot.scatter(tsneNDArray[:, 0], tsneNDArray[:, 1], c = Y_b)
    axesSubplot.set_xticks(())
    axesSubplot.set_yticks(())

    pp.show()

if __name__ == "__main__":
    test()