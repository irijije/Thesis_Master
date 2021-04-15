import os
import glob
import h5py
import numpy as np
import pandas as pd

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

    labels = []

    for i in range(int(len(df)/Config.TIMESTEP)):
        if i%100 == 0:
            print(f"{i}/{int(len(df)/Config.TIMESTEP)}")
        
        data = df.iloc[i*Config.TIMESTEP:(i+1)*Config.TIMESTEP]
        labels.append(1) if 'Attack' in data.values else labels.append(0)
        can_id = np.stack(data['1'].apply(lambda x : id2bit(x)).to_numpy())
        data = np.concatenate((np.array(data['0']).reshape(-1, 1), can_id), axis=1)

        np.save(Config.DATAPATH+str(i), data)
    np.save(Config.DATAPATH+"labels", np.array(labels))
    

if __name__ == "__main__":
    make_dataset(Config.FILENAME)