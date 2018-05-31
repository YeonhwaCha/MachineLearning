import numpy as np
from collections import Counter


class KNearestNeighbor(object):

    def __init__(self):
        pass

    def train(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    #def cross_validation(self):
    #    return validation_images, validation_labels, subset_train_images, subset_train_labels

    def predict(self, test_images, metrics_mode, k):
        num_test_images = test_images.shape[0]

        Ypred = np.zeros(num_test_images, dtype=self.train_labels.dtype)

        for idx in xrange(num_test_images):
            if metrics_mode == "L1":
                distances = np.sum(np.abs(self.train_images - test_images[idx, :]), axis=1)
            elif metrics_mode == "L2":
                distances = np.sqrt(np.sum(np.square(self.train_images - test_images[idx, :], axis=1)))

            # min_index = np.argmin(distances)
            neighbors = self.train_labels[distances.argsort()[:k]]
            # temp = Counter(neighbors).most_common(1)
            Ypred[idx] = Counter(neighbors).most_common(1)[0][0]

        return Ypred