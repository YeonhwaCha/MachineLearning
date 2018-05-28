from array import array

import numpy as np
import struct


###################################################
# load dataset
###################################################
def load_dataset(dataset, path, set, **arg):
    if dataset == "mnist":
        selecteddigits = arg['selecteddigits']
        dataset = load_mnist(path, set, selecteddigits)

    return dataset


###################################################
# mnist
###################################################
def load_mnist(path, dataset="training", selecteddigits = range(10)):
    if dataset == "training":
        fname_digits = path + 'train-images-idx3-ubyte'
        fname_labels = path + 'train-labels-idx1-ubyte'
    elif dataset == 'testing':
        fname_digits = path + 't10k-images-idx3-ubyte'
        fname_labels = path + 't10k-labels-idx1-ubyte'
    else:
        raise ValueError("dataset must be 'testing' or 'trainig'")

    # Import digits data
    digitsfileobject = open(fname_digits, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", digitsfileobject.read(16))
    digitsdata = array("B", digitsfileobject.read())
    digitsfileobject.close()

    #Import label data
    labelsfileobject = open(fname_labels, 'rb')
    magic_nr, size = struct.unpack(">II", labelsfileobject.read(8))
    labelsdata = array("B", labelsfileobject.read())
    labelsfileobject.close()

    #Find indices of selected digits
    indices = [k for k in range(size) if labelsdata[k] in selecteddigits]
    N = len(indices)

    #Create empty arrays for X and T
    X = np.zeros((N, rows*cols), dtype = np.uint8)
    T = np.zeros((N, 1), dtype = np.uint8)

    #Fill x from digitsdata
    #Fill T from labelsdata
    for i in range(N):
        X[i] = digitsdata[indices[i]*rows*cols:(indices[i]+1)*rows*cols]
        T[i] = labelsdata[indices[i]]

    return X,T