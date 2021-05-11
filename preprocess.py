import os
import glob
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from config import Config


def id2bit(id):
    b = np.zeros(29)
    b1 = np.right_shift(id, 8).astype(np.uint8)
    b[18:21] = np.unpackbits(b1)[-3:]
    b2 = np.array(id%256, dtype=np.uint8)
    b[21:29] = np.unpackbits(b2)

    return b

def one_step(chunk, pre_time):
    count = np.zeros((Config.N_ID))
    sum_IAT = np.zeros((Config.N_ID))
    pre_time = np.zeros((Config.N_ID))
    for i in range(len(chunk)):
        idx = chunk['1'].iloc[i]
        count[idx] += 1
        if pre_time[idx] != 0:
            sum_IAT[idx] += chunk['0'].iloc[i] - pre_time[idx]
        pre_time[idx] = chunk['0'].iloc[i]
    
    return count, sum_IAT, pre_time

def show_tsne(X_test, y_test, name='lstm'):
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_test = PCA(n_components=30).fit_transform(X_test)
    X_embedded = TSNE(n_components=2).fit_transform(X_test)

    df = pd.DataFrame(np.concatenate((X_embedded, y_test[:, None]), axis=1), columns=['x', 'y', 'label'])

    plt.figure()
    sns.scatterplot(x='x', y='y', hue='label', data=df)
    plt.savefig(f'figures/tsne_{name}.png')
    plt.show()

def make_dataset():
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(6)], header=None)
    #df = df[['0', '1', '4']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)
    df['3'] = df['3'].apply(lambda x: int(str(x).replace(" ", ""), base=16))

    data_all = []
    labels = []

    for i in range(Config.N_INTVL, int(len(df)/Config.UNIT_TIMESTEP)):
        if i%100 == 0:
            print(f"{i}/{int(len(df)/Config.UNIT_TIMESTEP)}")
        
        data = df.iloc[i*Config.UNIT_TIMESTEP-Config.MAX_TIMESTEP:i*Config.UNIT_TIMESTEP]
        labels.append(1) if 'Attack' in data.values else labels.append(0)
        can_id = np.stack(data['1'].apply(lambda x : id2bit(x)).to_numpy())
        data_all.append(np.concatenate((np.array(data['0']).reshape(-1, 1), np.array(data['3']).reshape(-1, 1), can_id), axis=1))
        #data_all.append(can_id)

    X_train, X_test, y_train, y_test  = train_test_split(np.array(data_all), np.array(labels), test_size=0.2)

    np.save(Config.DATAPATH+"data_train", np.array(X_train))
    np.save(Config.DATAPATH+"data_test", np.array(X_test))
    np.save(Config.DATAPATH+"labels_train", np.array(y_train))
    np.save(Config.DATAPATH+"labels_test", np.array(y_test))
    
    show_tsne(X_test, y_test, 'raw')

def make_dataset_hand():
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(12)], header=None)
    df = df[['0', '1', '4']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    pre_time = np.zeros(Config.N_ID)
    counts = np.zeros((Config.N_INTVL, Config.N_ID))
    sum_IATs = np.zeros((Config.N_INTVL, Config.N_ID))
    labels = []
    hist = []
    data_all = []

    for i in range(int(len(df)/Config.UNIT_TIMESTEP)-1):
        if i%100 == 0:
            print(f"{i}/{int(len(df)/Config.UNIT_TIMESTEP)-1}")
        
        frequencys = []
        mean_IATs = []

        cur = (i+1)*Config.UNIT_TIMESTEP

        big_chunk = df.iloc[cur-Config.MAX_TIMESTEP:cur]
        labels.append(1) if 'Attack' in big_chunk.values else labels.append(0)
        
        cur_chunk = df.iloc[cur-Config.UNIT_TIMESTEP:cur]
        cur_count, cur_sum_IAT, pre_time = one_step(cur_chunk, pre_time)
        hist.append((cur_count, cur_sum_IAT))

        for j in range(Config.N_INTVL):
            idx = i-(j+1)
            if idx >= 0:
                pre_count, pre_sum_IAT = hist[idx]
            else:
                pre_count, pre_sum_IAT = np.zeros_like(cur_count), np.zeros_like(cur_sum_IAT)
            
            counts[j] = counts[j] + cur_count - pre_count
            frequency = counts[j]
            sum_IATs[j] = sum_IATs[j] + cur_sum_IAT - pre_sum_IAT
            mean_IAT = sum_IATs[j]/(frequency+0.000001)

            frequencys.append(frequency)
            mean_IATs.append(mean_IAT)
        
        frequencys = np.array(frequencys).transpose()
        mean_IATs = np.array(mean_IATs).transpose()
        mean_IATs = np.array(pd.DataFrame(mean_IATs).replace([0, np.nan], 1))
        data_all.append(np.concatenate([frequencys, mean_IATs], -1))

    X_train, X_test, y_train, y_test  = train_test_split(np.array(data_all), np.array(labels), test_size=0.2)
    
    np.save(Config.DATAPATH+"data_train", np.array(X_train))
    np.save(Config.DATAPATH+"data_test", np.array(X_test))
    np.save(Config.DATAPATH+"labels_train", np.array(y_train))
    np.save(Config.DATAPATH+"labels_test", np.array(y_test))

    show_tsne(X_test, y_test, 'hand')


if __name__ == "__main__":
    make_dataset()
    #make_dataset_hand()