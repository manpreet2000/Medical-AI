import pandas as pd
import numpy as np
import config

def load(th):
  x=pd.read_csv(config.data_dir+"NHANESI_X.csv")
  y=pd.read_csv(config.data_dir+"NHANESI_y.csv")["y"]
  y=np.array(y)
  df = x.drop(['Unnamed: 0'], axis=1)
  df.loc[:, 'time'] = y
  df.loc[:, 'death'] = np.ones(len(x))
  df.loc[df.time < 0, 'death'] = 0
  df.loc[:, 'time'] = np.abs(df.time)
  df = df.dropna(axis='rows')
  mask = (df.time > th) | (df.death == 1)
  df = df[mask]
  x = df.drop(['time', 'death'], axis='columns')
  y = df.time < th
  return x,y

def cindex(y_true, scores):
    return lifelines.utils.concordance_index(y_true, scores)