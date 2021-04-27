import os
import glob
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config


def id2bit(id):
    b = np.zeros(29)
    b1 = np.right_shift(id, 8).astype(np.uint8)
    b[18:21] = np.unpackbits(b1)[-3:]
    b2 = np.array(id%256, dtype=np.uint8)
    b[21:29] = np.unpackbits(b2)

    return b

def make_dataset(filename):
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(filename, names=[str(x) for x in range(6)], header=None)
    df = df[['0', '1', '4']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    data_all = []
    labels = []

    for i in range(int(len(df)/Config.TIMESTEP)):
        if i%100 == 0:
            print(f"{i}/{int(len(df)/Config.TIMESTEP)}")
        
        data = df.iloc[i*Config.TIMESTEP:(i+1)*Config.TIMESTEP]
        labels.append(1) if 'Attack' in data.values else labels.append(0)
        can_id = np.stack(data['1'].apply(lambda x : id2bit(x)).to_numpy())
        data_all.append(np.concatenate((np.array(data['0']).reshape(-1, 1), can_id), axis=1))
        #data_all.append(can_id)

    X_train, X_test, y_train, y_test  = train_test_split(np.array(data_all), np.array(labels), test_size=0.2)
    
    np.save(Config.DATAPATH+"data_train", np.array(X_train))
    np.save(Config.DATAPATH+"data_test", np.array(X_test))
    np.save(Config.DATAPATH+"labels_train", np.array(y_train))
    np.save(Config.DATAPATH+"labels_test", np.array(y_test))
    

if __name__ == "__main__":
    make_dataset(Config.FILENAME)