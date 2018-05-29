import numpy as np

from sklearn import linear_model

def train(X, T, type):
    print "[train] Linear Regression.."
    linear_regression= linear_model.LinearRegression()
    linear_regression.fit(X, T)

    coef = linear_regression.coef_  # w_1 ~ w_n
    intercept = linear_regression.intercept_  # w_0
    if type == "binary":
        W = np.append(intercept, coef)
    else:
        W = np.hstack((intercept.reshape((intercept.shape[0],1)), coef))
        W = np.transpose(W)

    return linear_regression, W

def predict(model, X):
    print "[predict] Linear Regression.."
    P = model.predict(X)

    return P
