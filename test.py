import os
import glob
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from tools import *
from config import Config
from preprocess import show_tsne

data = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[2, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[3, 3, 3], [1, 1, 1]]])

print(data.shape)

data_ = np.zeros((3, len(max(data, key=len)), 3))

for i, d in enumerate(data):
    data_[i, :len(d)] = d

for i, d in enumerate(data_):
    np.random.shuffle(d)
    print(d)
    data_[i] = d

print(data_)
print(data_.shape)

