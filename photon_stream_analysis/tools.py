import numpy as np
from glob import glob
import pandas as pd

def cartesian2polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def read_and_concat_DataFrames(wild_card='*.msg'):
    dataframes = []
    for input_path in glob(wild_card):
        dataframes.append(
        	reduce_DataFrame_to_32_bit(
        		pd.read_msgpack(input_path)
        	)
        )
    return pd.concat(dataframes)


def reduce_DataFrame_to_32_bit(df):
    for key in df.keys():
        if df[key].dtype == 'float64':
            df[key] = df[key].astype(np.float32)
        if df[key].dtype == 'int64':
            df[key] = df[key].astype(np.int32)
    return df