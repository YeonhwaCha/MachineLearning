import numpy as np


def euclidean_distance(p1, p2):
    p1 = np.asarray(p1).flatten()
    p2 = np.asarray(p2).flatten()
    return np.sqrt(np.sum(np.power((p1-p2),2)))