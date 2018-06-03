import numpy as np
from classifier.Bayesian import apply_bayesian_classifier
from classifier.Histogram import apply_1Dhistogram_classifier, build_1Dhistogram_classifier
from utils.excels import read_excels
from utils.visualize import plot_xy_histogram

from pandas import DataFrame


if __name__ == "__main__":

    ################################################
    # load train set
    ################################################
    train_file_path = r"../../DataSet/Histogram1D/histogram_train.xlsx"
    train_data = read_excels(train_file_path)

    ################################################
    # load test set
    ################################################
    test_file_path = r"../../DataSet/Histogram1D/histogram_test.xlsx"
    test = read_excels(test_file_path)

    #################################################
    # data manipulation
    #################################################
    # input data = train_data[:,0] * 12 + train_data[:,1]
    X=np.array(train_data[:,0] * 12 + train_data[:,1], dtype=float)
    # target data = 1 or -1
    T=np.array([str(g) for g in train_data[:,2]])


    #################################################
    # train : Histogram
    #################################################
    B = 32;  # bin
    min = np.amin(X)
    max = np.amax(X)

    [H1, H2, H1_count, H2_count, H1_mean, H2_mean, H1_cov, H2_cov] = \
        build_1Dhistogram_classifier(X, T, B, 'Female', 'Male', min, max)
    # show histogram
    plot_xy_histogram(H1, H2, B, min, max)

    ##################################################################################
    # Classifier
    ##################################################################################
    # Apply 1D Histogram Classifier..
    [resultHlabel, resultHprob] = apply_1Dhistogram_classifier(test, H1, H2, 'Female', 'Male', min, max)
    print "[ 1D Histogram Classifier ]"
    print(DataFrame([resultHlabel, resultHprob]).T)

    # Apply Bayesian Classifier
    [resultHlabel, resultHprob] = apply_bayesian_classifier(test, 1, 'Female', 'Male', H1_count, H2_count, H1_mean, H2_mean, H1_cov, H2_cov)
    print "[ Bayesian Classifier ]"
    print(DataFrame([resultHlabel, resultHprob]).T)







