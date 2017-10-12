from sklearn.cluster import DBSCAN
import numpy as np


def two_dimensinal_cluster(x, y, eps=0.1, min_samples=5):
    assert x.shape[0] == y.shape[0]
    points = np.zeros(shape=(x.shape[0],2))
    points[:,0] = x
    points[:,1] = y

    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = dbscan.labels_

    # Number of clusters in labels, ignoring background if present.
    number = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, number