import numpy as np


############################################################################
# PCA
# (X : input, Z : X - mean, C : covariance, V : eigenvector)
# return [eigenvalues, eigenvectors, mean]
############################################################################
def pca(X, T, num_principal_componant = 0):
    [n, d] = X.shape

    if (num_principal_componant <= 0) or (num_principal_componant >n):
        num_principal_componant = n

    mean = X.mean(axis=0)
    X = X - mean

    # [PCA - eigenvector] : use the normalized eigenvector
    if n > d:
        covariance = np.dot(X.T, X)
        # the result of eigenvectors is the normalized eigenvectors by using np.linalg.eigh()
        # eigenvalues : type - ndarray, eigenvectors : type - ndarray
        [eigenvalues, eigenvectors] = np.linalg.eigh(covariance)
    else:
        covariance = np.dot(X, X.T)  # feature * feature
        [eigenvalues, eigenvectors] = np.linalg.eigh(covariance)
        eigenvectors = np.dot(X.T, eigenvectors)
        for idx in xrange(n):
            eigenvectors[: ,idx] = eigenvectors[: ,idx ] /np.linalg.norm(eigenvectors[: ,idx]) # v/|v|

    # - np.argsort(eigenvalues)  : sort index based on ascending order (0, 1, 2, 3)
    # - np.argsort(-eigenvalues) : sort index based on descending order ( 3, 2, 1, 0)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # choose principal componants based on the setting by user.
    eigenvalues = eigenvalues[0:num_principal_componant].copy()
    eigenvectors = eigenvectors[:, 0:num_principal_componant].copy()

    return [eigenvalues, eigenvectors, mean]


def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    X = X - float(minX)
    X = X / float((maxX - minX))
    X = X * (high - low)
    X = X + low

    if dtype is None:
        return np.asarray(X)

    return np.asarray(X, dtype=dtype)


def project(eigen_vector, X, mean=None):
    if mean is None:
        return np.dot(X, eigen_vector)
    # Z * eigen_vector :: Principal Components
    return np.dot(X - mean, eigen_vector)


def reconstruct(eigen_vector, P, mu=None):
    if mu is None:
        return np.dot(P, eigen_vector.T)
    # X : original image 'X' is recovered.
    return np.dot(P, eigen_vector.T) + mu
