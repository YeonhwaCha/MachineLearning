import math
import numpy as np
from scipy.stats import multivariate_normal

def apply_bayesian_classifier(queries, dim, class1, class2, class1_sample, class2_sample, class1_mean, class2_mean, class1_cov, class2_cov):

    queries = queries.reshape(queries.shape[0],)

    conditional_class1 = get_pdf(queries, class1_mean, class1_cov, dim);
    conditional_class2 = get_pdf(queries, class2_mean, class2_cov, dim);

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


def get_pdf(queries, mean, var, dim):
    # pdf_test = multivariate_normal.pdf(queries, female_mean, female_cov)
    # print "using built in functino : {0}".format(pdf_test)

    exp_result = [];

    if dim == 1:
        pdf_const = 1.0 / (math.pow(2 * np.pi * var, 1.0 / 2));
    else:
        cov_det = np.linalg.det(var);
        cov_inv = np.linalg.inv(var);
        pdf_const = 1.0 / (2 * np.pi * math.pow(cov_det, 1.0 / 2));

    for idx in xrange(0, len(queries)):
        # center value (x-mu)
        if dim == 1:
            z = queries[idx] - mean
            exp_component = -0.5 * (pow(z, 2) / var);
        else:
            z = queries[idx, :] - mean
            exp_component = -0.5 * (z.dot(cov_inv)).dot(z.T);

        exp_result.append(np.exp(exp_component)*pdf_const);

    # print "class1_mean : \n {0}".format(mean)
    # print "class1_cov : \n {0}".format(np.asmatrix(cov))
    # print "center_queries : \n {0}".format(z)
    # print "cov_inv : \n {0}".format(cov_inv)
    # print "cov_det : {0}".format(cov_det)
    # print "pdf_const : {0}".format(pdf_const)
    # print "exp_result : {0}".format(exp_result)

    return np.array(exp_result)


