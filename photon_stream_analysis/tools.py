import numpy as np


def cartesian2polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def read_and_concat_DataFrames(wild_card='*.msg'):
    dataframes = []
    for input_path in glob(wild_card):
        dataframes.append(reduce_to_32_bit(pd.read_msgpack(input_path)))
    return pd.concat(dataframes)

