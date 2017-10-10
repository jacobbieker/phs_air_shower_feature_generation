import numpy as np


def not_unique(list_of_index_arrays):
    l = len(list_of_index_arrays)
    idxs = np.concatenate(list_of_index_arrays)
    u = np.unique(idxs, return_counts=True)
    return u[0][u[1] == l]
