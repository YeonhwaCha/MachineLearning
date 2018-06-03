import numpy as np

# convert from rectangle image to feature vector : N * n(feature)
def asRowMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((0, X[0].size), dtype = X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1, -1)))
        # reshape(row, -1) -> row * size(input)

    return mat

# convert from rectangle image to feature vector : n(feature) * N
def asColumnMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((X[0].size, 0), dtype = X[0].dtype)
    for col in X:
        mat = np.hstack((mat, np.asarray(col).reshape(-1, 1)))

    return mat