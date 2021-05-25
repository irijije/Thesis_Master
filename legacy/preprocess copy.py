import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    pre_time = np.zeros((Config.N_ID))
    for i in range(len(chunk)):
        idx = chunk['1'].iloc[i]
        count[idx] += 1
        if pre_time[idx] != 0:
            sum_IAT[idx] += chunk['0'].iloc[i] - pre_time[idx]
        pre_time[idx] = chunk['0'].iloc[i]
    
    return count, sum_IAT, pre_time

def temporalize(X, y, lookback):
        output_X = []
        output_y = []
        for i in range(len(X) - lookback):
            t = []
            for j in range(lookback):
                t.append(X[i + j, :])
            output_X.append(t)
            s = list(set(y[i:i+lookback])-{0})
            output_y.append(0) if s==[] else output_y.append(list(s)[0])

        return np.squeeze(np.array(output_X)), np.array(output_y)

def make_dataset_lstm():
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(6)], header=None)
    df = df[['0', '1', '4', '5']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    end, start = df['0'].max(), df['0'].min()
    num_data = int((end-start)/Config.UNIT_INTVL)
    pre_time = np.zeros(Config.N_ID)
    frequencys = []
    mean_IATs = []
    labels = []

    for i in range(num_data):
        if i%100 == 0:
            print(f"{i}/{num_data} {int(i/num_data*100)}%")
    
        cur = start+i*Config.UNIT_INTVL

        chunk = df[(df['0'] >= cur-Config.UNIT_INTVL) & (df['0'] < cur)]
        if Config.isMC:
            labels.append(get_class(chunk)) if 'Attack' in chunk.values else labels.append(0)
        else:
            labels.append(1) if 'Attack' in chunk.values else labels.append(0)
        
        frequency, sum_IAT, pre_time = one_step(chunk, pre_time)

        mean_IAT = sum_IAT/(frequency+0.000001)
        frequencys.append(frequency)
        mean_IATs.append(mean_IAT)
        
    frequencys = np.array(frequencys)[:, :, None]
    mean_IATs = np.array(pd.DataFrame(mean_IATs).replace([0, np.nan], 1))[:, :, None]
    data = np.concatenate((frequencys, mean_IATs), 2)

    data, labels = temporalize(data, labels, Config.UNIT_TIMESTEP)

    X_train, X_test, y_train, y_test  = train_test_split(np.array(data), np.array(labels), test_size=0.2)
    
    np.save(Config.DATAPATH+"data_train", np.array(X_train))
    np.save(Config.DATAPATH+"data_test", np.array(X_test))
    np.save(Config.DATAPATH+"labels_train", np.array(y_train))
    np.save(Config.DATAPATH+"labels_test", np.array(y_test))

    show_tsne(X_test, y_test, 'lstm')

def make_dataset_cnn():
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(6)], header=None)
    df = df[['0', '1', '4', '5']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    end, start = df['0'].max(), df['0'].min()
    num_data = int((end-start)/Config.UNIT_INTVL)
    pre_time = np.zeros(Config.N_ID)
    counts = np.zeros((Config.N_INTVL, Config.N_ID))
    sum_IATs = np.zeros((Config.N_INTVL, Config.N_ID))
    data = []
    labels = []
    hist = []

    for i in range(num_data):
        if i%100 == 0:
            print(f"{i}/{num_data} {int(i/num_data*100)}%")
    
        frequencys = []
        mean_IATs = []

        cur = start+i*Config.UNIT_INTVL

        big_chunk = df[(df['0'] >= cur-Config.UNIT_INTVL*Config.N_INTVL) & (df['0'] < cur)]
        
        if Config.isMC:
            labels.append(get_class(big_chunk)) if 'Attack' in big_chunk.values else labels.append(0)
        else:
            labels.append(1) if 'Attack' in big_chunk.values else labels.append(0)
        
        cur_chunk = df[(df['0'] >= cur-Config.UNIT_INTVL) & (df['0'] < cur)]
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
        data.append(np.concatenate([frequencys, mean_IATs], -1))
    
    print(np.array(data).shape)

    print(np.unique(labels, return_counts=True))

    X_train, X_test, y_train, y_test  = train_test_split(np.array(data), np.array(labels), test_size=0.2)
    
    np.save(Config.DATAPATH+"data_train", np.array(X_train))
    np.save(Config.DATAPATH+"data_test", np.array(X_test))
    np.save(Config.DATAPATH+"labels_train", np.array(y_train))
    np.save(Config.DATAPATH+"labels_test", np.array(y_test))

    show_tsne(X_test, y_test, 'cnn')


if __name__ == "__main__":
    if Config.MODE == 'raw':
        pass
    elif Config.MODE == 'cnn' or Config.MODE == 'test':
        make_dataset_cnn()
    elif Config.MODE == 'lstm':
        make_dataset_lstm()