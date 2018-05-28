import math
import numpy as np
from scipy.stats import multivariate_normal

def ApplyBayesianClassifier(queries, class1, class2, class1_sample, class2_sample, class1_mean, class2_mean, class1_cov, class2_cov):

    conditional_class1 = GetPdf(queries, class1_mean, class1_cov);
    conditional_class2 = GetPdf(queries, class2_mean, class2_cov);

    posterior_class1 =  class1_sample * conditional_class1 / ((class2_sample * conditional_class2) + (class1_sample * conditional_class1));
    posterior_class2 = class2_sample * conditional_class2 / ((class2_sample * conditional_class2) + (class1_sample * conditional_class1));

    indices =range(queries.shape[0])
    count1 = posterior_class1[indices]
    count2 = posterior_class2[indices]
    resultlabel = np.full(np.alen(indices), "Indeterminate", dtype=object);
    resultprob = np.full(np.alen(indices), np.nan, dtype=object);
    indices1 = count1 > count2;
    indices2 = count2 > count1;
    resultlabel[indices1] = class1;
    resultlabel[indices2] = class2;
    probF = count1 / (count1 + count2);
    probM = count2 / (count1 + count2);
    resultprob[indices1] = probF[indices1];
    resultprob[indices2] = probM[indices2];
    return resultlabel, resultprob


def GetPdf(queries, mean, cov):
    # pdf_test = multivariate_normal.pdf(queries, female_mean, female_cov)
    # print "using built in functino : {0}".format(pdf_test)

    exp_result = [];

    cov_det = np.linalg.det(cov);
    cov_inv = np.linalg.inv(cov);
    pdf_const = 1.0 / (2 * np.pi * math.pow(cov_det, 1.0 / 2));

    for idx in xrange(0, len(queries)):
        # center value (x-mu)
        z = queries[idx, :] - mean
        exp_component = -0.5 * (z.dot(cov_inv)).dot(z.T);
        exp_result.append(np.exp(exp_component)*pdf_const);

    # print "female_mean : \n {0}".format(mean)
    # print "female_cov : \n {0}".format(np.asmatrix(cov))
    # print "center_queries : \n {0}".format(z)
    # print "cov_inv : \n {0}".format(cov_inv)
    # print "cov_det : {0}".format(cov_det)
    # print "pdf_const : {0}".format(pdf_const)
    # print "exp_result : {0}".format(exp_result)

    return np.array(exp_result)
