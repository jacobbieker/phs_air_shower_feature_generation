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



def histogram(wild_card='*thrown', key='energy', bins=10):
    paths = glob(wild_card)
    df = pd.read_msgpack(paths[0])
    df.dropna(inplace=True)
    counts, bins = np.histogram(df[key], bins=bins)
    total_bincounts = np.zeros(len(bins)-1)

    for path in paths:
        df = pd.read_msgpack(path)
        df.dropna(inplace=True)
        bincounts = np.histogram(df[key], bins=bins)[0]
        total_bincounts += bincounts
    return total_bincounts, bins


def histogram2d(wild_card='*thrown', x='energy', y='number_photons', bins=10):
    paths = glob(wild_card)
    df = pd.read_msgpack(paths[0])
    df.dropna(inplace=True)
    h = np.histogram2d(x=df[x], y=df[y], bins=bins)
    total_counts = h[0] 
    bins = h[1]
    total_counts = 0

    for path in paths:
        df = pd.read_msgpack(path)
        df.dropna(inplace=True)
        bincounts = np.histogram2d(x=df[x], y=df[y], bins=bins)[0]
        total_counts += bincounts
    return total_counts, bins