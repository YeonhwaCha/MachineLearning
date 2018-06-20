from array import array
from utils.tools import asRowMatrix

import numpy as np
import os
import PIL.Image as Image
import struct


###################################################
# load dataset
###################################################
def load_dataset(dataset, path, **arg):
    if dataset == "mnist":
        selecteddigits = arg['selecteddigits']
        set = arg['set']
        return load_mnist(path, set, selecteddigits)
    elif dataset == "att":
        return load_att(path)

    return dataset


#####################################################################
# AT&T Facedatabase
# DataSet CopyRight : AT&T Laboratories Cambridge
# http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
#####################################################################
def load_att(dataset_path):
    db_cls = 0
    X, T = [], []

    # dirnames = [s1, s2, s3,,,,]
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for dirname in dirnames:
            database_path = os.path.join(dirpath, dirname)
            for filename in os.listdir(database_path):
                image = Image.open(os.path.join(database_path, filename))   #size 92*112 image
                image = image.convert("L")   #convert to grayscale
                X.append(np.asarray(image, dtype=np.uint8))
                T.append(db_cls)
            db_cls = db_cls+1

    img_shape = X[0].shape

    return asRowMatrix(X), T, img_shape



###################################################
# mnist
# DataSet CopyRight : New York Univ. Google Labs.
# http://yann.lecun.com/exdb/mnist/
###################################################
def load_mnist(path, dataset="training", selecteddigits = range(10)):
    if dataset == "training":
        fname_digits = path + 'train-images-idx3-ubyte'
        fname_labels = path + 'train-labels-idx1-ubyte'
    elif dataset == 'testing':
        fname_digits = path + 't10k-images-idx3-ubyte'
        fname_labels = path + 't10k-labels-idx1-ubyte'
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

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

    image_shape = (rows, cols)

    return X, T, image_shape