import numpy as np
import pandas as pd

from config import Config

df = pd.read_csv(Config.FILENAME, names=[str(x) for x in range(6)], header=None)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

df['1'] = df['1'].apply(int, base=16)
df['3'] = df['3'].apply(lambda x: int(str(x).replace(" ", ""), base=16))

print(df.head())