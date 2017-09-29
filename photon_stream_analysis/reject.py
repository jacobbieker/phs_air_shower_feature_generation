import numpy as np


def early_or_late_clusters(cluster):
    cluster_arrival_times = np.zeros(cluster.number)
    cluster_sizes = np.zeros(cluster.number)
    for c in range(cluster.number):
        cluster_sizes[c] = (cluster.labels==c).sum()
        cluster_arrival_times[c] = cluster.xyt[cluster.labels==c,2].mean()
    order = np.argsort(cluster_sizes)
    order = np.flip(order, axis=0)
    cluster_arrival_times = cluster_arrival_times[order]
    rejected_clusters = []
    for c in range(cluster.number):
        delay = cluster_arrival_times[c] - cluster_arrival_times[0]
        if np.abs(delay) > 0.75:
            rejected_clusters.append(c)
    accepted_cluster = cluster
    for c in rejected_clusters:
        accepted_cluster.labels[accepted_cluster.labels == c] = -1
        accepted_cluster.number -= 1
    return accepted_cluster