import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from tools import *
from config import Config


def get_class(chunk):
    c = 0
    if 'Flooding' in chunk.values: c = 1
    elif 'Spoofing' in chunk.values: c = 2
    elif 'Replay' in chunk.values: c = 3
    elif 'Fuzzing' in chunk.values: c= 4

    return c

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
    for i in range(len(chunk)):
        idx = chunk['1'].iloc[i]
        count[idx] += 1
        if pre_time[idx] != 0:
            sum_IAT[idx] += chunk['0'].iloc[i] - pre_time[idx]
        pre_time[idx] = chunk['0'].iloc[i]
    
    return count, sum_IAT, pre_time

def temporalize_lstm(X, y, lookback):
    X_ = []
    y_ = []
    X[:, :, 1] = X[:, :, 1]/(X[:, :, 0]/0.000001)
    for i in range(len(X) - lookback + 1):
        if i%100 == 0:
            print(f"{i}/{len(X)} {int(i/len(X)*100)}%")
        t = []
        for j in range(lookback):
            t.append(X[i + j, :])
        X_.append(t)
        s = list(set(y[i:i+lookback])-{0})
        y_.append(0) if s==[] else y_.append(s[0])
        
    return np.squeeze(np.array(X_)), np.array(y_)

def temporalize_cnn(X, y, lookback):
    X_ = []
    y_ = []
    counts = np.zeros((Config.N_INTVL, Config.N_ID))
    sum_IATs = np.zeros((Config.N_INTVL, Config.N_ID))
    for i in range(lookback):
        counts[i] = np.sum(X[:i, :, 0], axis=0)
        sum_IATs[i] = np.sum(X[:i, :, 1], axis=0)
    for i in range(lookback-1, len(X)):
        if i%100 == 0:
            print(f"{i}/{len(X)} {int(i/len(X)*100)}%")
        frequencys = []
        mean_IATs = []
        cur_count, cur_sum_IAT = X[i, :, 0], X[i, :, 1]

        for j in range(lookback):
            idx = i-(j+1)
            if idx >= 0:
                pre_count, pre_sum_IAT = X[idx, :, 0], X[idx, :, 1]
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
        X_.append(np.concatenate([frequencys, mean_IATs], -1))

        s = list(set(y[i:i+lookback])-{0})
        y_.append(0) if s==[] else y_.append(s[0])
        
    return np.array(X_), np.array(y_)

def make_dataset_base():
    os.makedirs(Config.BASEPATH, exist_ok=True)
    df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(6)], header=None)
    df = df[['0', '1', '4', '5']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    end, start = df['0'].max(), df['0'].min()
    num_data = int((end-start)/Config.UNIT_INTVL)
    pre_time = np.zeros(Config.N_ID)
    frequencys = []
    sum_IATs = []
    labels = []

    for i in range(num_data):
        if i%100 == 0:
            print(f"{i}/{num_data} {int(i/num_data*100)}%")
    
        cur = start+(i+1)*Config.UNIT_INTVL

        chunk = df[(df['0'] >= cur-Config.UNIT_INTVL) & (df['0'] < cur)]
        if Config.isMC:
            labels.append(get_class(chunk)) if 'Attack' in chunk.values else labels.append(0)
        else:
            labels.append(1) if 'Attack' in chunk.values else labels.append(0)
        
        frequency, sum_IAT, pre_time = one_step(chunk, pre_time)

        frequencys.append(frequency)
        sum_IATs.append(sum_IAT)
        
    frequencys = np.array(frequencys)[:, :, None]
    sum_IATs = np.array(sum_IATs)[:, :, None]
    data = np.concatenate((frequencys, sum_IATs), 2)
    labels = np.array(labels)

    print(data.shape)
    print(np.unique(labels, return_counts=True))

    np.save(Config.BASEPATH+"data", data)
    np.save(Config.BASEPATH+"labels", labels)

    show_tsne(data[:1000], labels[:1000])

def make_dataset():
    os.makedirs(Config.DATAPATH, exist_ok=True)
    data, labels = np.load(Config.BASEPATH+"data1.npy"), np.load(Config.BASEPATH+"labels1.npy")
    print(data.shape)

    if Config.MODE == 'cnn':
        data, labels = temporalize_cnn(data, labels, Config.N_INTVL)
    elif Config.MODE == 'lstm':
        data, labels = temporalize_lstm(data, labels, Config.UNIT_TIMESTEP)

    data, labels = shuffle(data, labels)

    print(data.shape)
    print(np.unique(labels, return_counts=True))

    np.save(Config.DATAPATH+"data1", np.array(data))
    np.save(Config.DATAPATH+"labels1", np.array(labels))

    show_tsne(data[:1000], labels[:1000])

def make_dataset_raw():
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(6)], header=None)
    df = df[['0', '1', '4', '5']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    end, start = df['0'].max(), df['0'].min()
    num_data = int((end-start)/Config.UNIT_INTVL)
    data = []
    labels = []

    for i in range(Config.N_INTVL-1, num_data):
        if i%100 == 0:
            print(f"{i}/{num_data} {int(i/num_data*100)}%")
    
        cur = start+(i+1)*Config.UNIT_INTVL

        chunk = df[(df['0'] >= cur-Config.UNIT_INTVL*Config.N_INTVL) & (df['0'] < cur)]
        if Config.isMC:
            labels.append(get_class(chunk)) if 'Attack' in chunk.values else labels.append(0)
        else:
            labels.append(1) if 'Attack' in chunk.values else labels.append(0)
        d = chunk['1'].apply(lambda x : id2bit(x))
        if not d.empty: data.append(np.stack(d.to_numpy()))
        else: data.append(np.zeros((407, 29)))

    data = np.array(data)
    
    data_ = np.zeros((data.shape[0], len(max(data, key=len)), len(data[0][0])))
    for i, d in enumerate(data):
        data_[i, :len(d)] = d
    data = data_
    
    labels = np.array(labels)

    data, labels = shuffle(data, labels)

    print(data.shape)
    print(np.unique(labels, return_counts=True))

    np.save(Config.DATAPATH+"data", data)
    np.save(Config.DATAPATH+"labels", labels)

    show_tsne(data[:1000], labels[:1000])

def make_dataset_test():
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(6)], header=None)
    df = df[['0', '1', '4', '5']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    data = []
    labels = []
    num_data = int(len(df)/29)

    for i in range(Config.N_INTVL-1, num_data):
        if i%100 == 0:
            print(f"{i}/{num_data} {int(i/num_data*100)}%")

        chunk = df.iloc[i*29:(i+1)*29]
        if Config.isMC:
            labels.append(get_class(chunk)) if 'Attack' in chunk.values else labels.append(0)
        else:
            labels.append(1) if 'Attack' in chunk.values else labels.append(0)
        d = chunk['1'].apply(lambda x : id2bit(x))
        if not d.empty: data.append(np.stack(d.to_numpy()))
        else: data.append(np.zeros((29, 29)))

    data = np.array(data)
    labels = np.array(labels)

    data, labels = shuffle(data, labels)

    print(data.shape)
    print(np.unique(labels, return_counts=True))

    np.save(Config.DATAPATH+"data", data)
    np.save(Config.DATAPATH+"labels", labels)

    show_tsne(data[:1000], labels[:1000])

def merge_data():
    data1, labels1 = np.load(Config.DATAPATH+"data1.npy"), np.load(Config.DATAPATH+"labels1.npy")
    data2, labels2 = np.load(Config.DATAPATH+"data2.npy"), np.load(Config.DATAPATH+"labels2.npy")

    # data_ = np.zeros((data1.shape[0], 407, len(data1[0][0])))
    # for i, d in enumerate(data1):
    #     data_[i, :len(d)] = d
    # data1 = data_

    # data_ = np.zeros((data2.shape[0], 407, len(data2[0][0])))
    # for i, d in enumerate(data2):
    #     data_[i, :len(d)] = d
    # data2 = data_

    data = np.concatenate((data1, data2))
    labels = np.concatenate((labels1, labels2))

    print(data.shape)
    print(labels.shape)

    np.save(Config.DATAPATH+f"data", data)
    np.save(Config.DATAPATH+f"labels", labels)

if __name__ == "__main__":
    #make_dataset_base()
    #make_dataset()
    make_dataset_raw()
    #make_dataset_test()
    #merge_data()