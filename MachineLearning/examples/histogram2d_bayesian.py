import numpy as np

from classifier.Bayesian import apply_bayesian_classifier
from classifier.Histogram import apply_2Dhistogram_classifier, build_2Dhistogram_classifier
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


# #####################################################################################
#
# #####################################################################################
# # Reconstruct Histogram
# ######################################################################################
# data1 = Reconstruct(B, female_sample, female_mean, female_cov, 'gaussian')
# data2 = Reconstruct(B, male_sample, male_mean, male_cov, 'gaussian')
# data = np.hstack((data1, data2))
#
# # [problem 1] - calculate max and min
# heightX_min = np.amin(data[0, :]);
# heightX_max = np.amax(data[0, :]);
# handspanX_min = np.amin(data[1, :]);
# handspanX_max = np.amax(data[1, :]);
#
# minAndMax = np.zeros((4, 1)).astype('float32')
# minAndMax[0, 0] = heightX_min;
# minAndMax[1, 0] = heightX_max;
# minAndMax[2, 0] = handspanX_min;
# minAndMax[3, 0] = handspanX_max;
#
# reconstruct = ReconstructHist(data1, B, heightX_min, heightX_max, handspanX_min, handspanX_max)
#
# reconstruct = ReconstructHist(data2, B, heightX_min, heightX_max, handspanX_min, handspanX_max)

