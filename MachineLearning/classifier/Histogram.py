import numpy as np
import scipy


def apply_2Dhistogram_classifier(queries, H1, H2, class1, class2, xmin, xmax):
    B=np.alen(H1);
    binindices = (np.round((B-1)*(queries-xmin)/(xmax-xmin))).astype('int32');

    count1 = np.zeros(queries.shape[0]).astype('float32');
    count2 = np.zeros(queries.shape[0]).astype('float32');

    for i, row in enumerate(binindices):
        count1[i] = H1[row[0], row[1]];
        count2[i] = H2[row[0], row[1]];

    resultlabel=np.full(np.alen(binindices),"Indeterminate",dtype=object);
    resultprob=np.full(np.alen(binindices),np.nan,dtype=object);
    indices1 = count1 > count2;
    indices2 = count2 > count1;
    resultlabel[indices1] = str(class1);
    resultlabel[indices2] = str(class2);
    probF=count1/(count1+count2);
    probM=count2/(count1+count2);
    resultprob[indices1]=probF[indices1];
    resultprob[indices2]=probM[indices2];
    return resultlabel, resultprob


# nominal scale (Male / Female)
def build_2Dhistogram_classifier(X, T, B, class1, class2, xmin, xmax):
    H1 = np.zeros((B ,B)).astype('int32');
    H2 = np.zeros((B, B)).astype('int32');

    # make indices based on bin(B)
    binindices = (np.round((B-1)*(X-xmin)/(xmax-xmin))).astype('int32');

    class1_set = [];
    class2_set = [];

    for i, row in enumerate(binindices):
        if T[i] == class1:
            H1[row[0], row[1]] += 1;
            class1_set.append(X[i, :])
        elif T[i] == class2:
            H2[row[0], row[1]] += 1;
            class2_set.append(X[i, :])

    class1_data = np.array(class1_set)
    class2_data = np.array(class2_set)
    class1_mean = np.mean(class1_data, axis=0)
    class2_mean = np.mean(class2_data, axis=0)
    class1_cov = np.cov(class1_data.T)
    class2_cov = np.cov(class2_data.T)

    # print "negative sample cnt : {0}, positive sample cnt : {1}".format(len(negative_set), len(positive_set))
    # print "negative mean : {0},       positive mean : {1}".format(negative_mean, positive_mean)
    # print "negative cov : \n {0},     positive cov : \n {1}".format(negative_cov, positive_cov)

    return [H1, H2,
            len(class1_set), len(class2_set),
            class1_mean, class2_mean,
            class1_cov, class2_cov]



def reconstruct(N, mean, cov, reconstruct_fun):
    if reconstruct_fun == "gaussian":
        data = np.random.multivariate_normal(mean, cov, N)
        data = data.T
    else:
        L = scipy.linalg.cholesky(cov)
        uncorrelated = np.random.standard_normal((2, N))
        data = np.dot(L, uncorrelated) + np.array(mean).reshape(2, 1)

    return data


def reconstruct_hist(data, B, x_min, x_max, y_min, y_max):
    reconstruct = np.zeros((B, B)).astype('int32');
    # make indices based on bin(B)
    binindices_x = (np.round((B - 1) * (data[0,:]- x_min) / (x_max - x_min))).astype('int32');
    binindeces_y = (np.round((B - 1) * (data[1,:]- y_min) / (y_max - y_min))).astype('int32');
    columns = list(enumerate(binindeces_y));

    for i, row in enumerate(binindices_x):
        column = columns[i];
        reconstruct[row][column[1]] += 1;

    return reconstruct


