import numpy as np

from utils.distance import euclidean_distance

from pre_processing.PCA import pca, project

class BaseModel(object):
    def __init__(self, X = None, T = None, principal_componant_num = 0):
        self.principal_componant_num = principal_componant_num
        # updated by modeling
        self.principals = []
        self.eigen_vector = []
        self.mean = []
        if (X is not None) and (T is not None):
            self.model(X, T, principal_componant_num)

    def model(self, X, T, principal_componant_num):
        raise NotImplementedError("Model need to be defined.. ")

    # find a principal which match with input data 'X'
    def predict(self, x):
        print "[prediction] :: classification based on euclidean_distance.. "
        # min_dist : initialized the maximum number in the range of float.
        min_dist = np.finfo('float').max
        min_class = -1
        input_data_principal = project(self.eigen_vector, x, self.mean)
        for i in xrange(len(self.principals)):
            dist = euclidean_distance(self.principals[i], input_data_principal)
            if dist < min_dist:
                min_dist = dist
                min_class = self.T[i]

        return min_class


#####################################################################
# - input
# X : input data
# T : target data (ground truth)
#
# - constants
# distance_metric : euclidean_distance
# principal_componant_num : 0 -> I'm gonna using all eigen_vectors.
#
# - update (self)
# principals, eigen_vector, mean / T
#####################################################################
class EigenVectorModel(BaseModel):
    def __init__(self, X = None, T = None, principal_componant_num = 0):
        print "[Init] :: eigen_vector_model"
        super(EigenVectorModel, self).__init__(X, T, principal_componant_num)


    def model(self, X, T, principal_componant_num):
        print "[modeling] :: model based on eigen vector"
        [eigen_value, self.eigen_vector, self.mean] = pca(X, T, principal_componant_num)
        self.T = T
        for x in X:
            self.principals.append(project(self.eigen_vector, x, self.mean))