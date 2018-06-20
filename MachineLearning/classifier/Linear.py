import numpy as np

from sklearn.metrics import confusion_matrix

def train(X, T, type):
    print "[train] Linear Regression.."
    #[[1, x(0,1), x(0,2).... x(0, n)]
    # [1, x(1,1), x(1,2).... x(1, n)]
    # [1, x(n,1), x(n,2).... x(n, n]]
    Augment_X = np.hstack((np.ones((X.shape[0], 1)), X))
    # In the case of non-square X, we can use pseudo-inverse(pinv)
    # refer to note
    W = np.dot(np.linalg.pinv(Augment_X), T)

    return W

def predict(weight, X):
    print "[predict] Linear Regression.."
    Augment_X = np.hstack((np.ones((X.shape[0], 1)), X))
    P = np.dot(Augment_X, weight)

    return P

def get_confusion_matrix(X, T):
    return confusion_matrix(X, T)
