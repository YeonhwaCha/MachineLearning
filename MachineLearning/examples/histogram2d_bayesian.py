import numpy as np

from classifier.Bayesian import apply_bayesian_classifier
from classifier.Histogram import apply_2Dhistogram_classifier, build_2Dhistogram_classifier, reconstruct, reconstruct_hist
from pandas import DataFrame
from utils.excels import read_excels
from utils.visualize import plot_xyz_histogram

if __name__ == "__main__":

    ################################################
    # load train set
    ################################################
    train_file_path = r"../../DataSet/histogram_train.xlsx"
    train_data = read_excels(train_file_path)

    ################################################
    # load test set
    ################################################
    test_file_path = r"../../DataSet/histogram_test.xlsx"
    test = read_excels(test_file_path)

    #################################################
    # data manipulation
    #################################################
    # input data
    X = np.array(train_data[:, [1,2]],dtype=float)
    # target data
    T=np.array([str(g) for g in train_data[:,0]])


    #################################################
    # train : Histogram
    #################################################
    B=5;
    min = np.amin(X, axis=0);
    max = np.amax(X, axis=0);

    [H1, H2, H1_count, H2_count, H1_mean, H2_mean, H1_cov, H2_cov] = \
        build_2Dhistogram_classifier(X, T, B, 'Female', 'Male', min, max);

    # # Draw 3D Histogram Graph for two different data.
    plot_xyz_histogram(H1, H2)

    ##################################################################################
    # Classifier
    ##################################################################################
    # Apply 2D Histogram Classifier..
    [resultHlabel, resultHprob] = apply_2Dhistogram_classifier(test, H1, H2, 'Female', 'Male', min, max)
    print"[ 2D Histogram Classifier ]"
    print(DataFrame([resultHlabel, resultHprob]).T)

    # Apply Bayesian Classifier..
    [resultHlabel, resultHprob] = apply_bayesian_classifier(test, 2, 'Female', 'Male', H1_count, H2_count, H1_mean, H2_mean, H1_cov, H2_cov)
    print"[ Bayesian Classifier ]"
    print(DataFrame([resultHlabel, resultHprob]).T)


    #####################################################################################
    # Reconstruct Histogram
    ######################################################################################
    data1 = reconstruct(H1_count, H1_mean, H1_cov, 'gaussian')
    data2 = reconstruct(H2_count, H2_mean, H2_cov, 'gaussian')
    data = np.hstack((data1, data2))

    # [problem 1] - calculate max and min
    min = np.amin(data, axis=1).reshape((2,1));
    max = np.amax(data, axis=1).reshape((2,1));

    reconstruct = reconstruct_hist(data1, B, min, max)
    reconstruct = reconstruct_hist(data2, B, min, max)

