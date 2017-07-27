"""
k-Nearest Neighbour machine learning algorithm
"""

import numpy as np


class KNNLearner(object):

    def __init__(self, k, verbose=False):
        self.k = k
        self.train_x = None
        self.train_y = None

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        if self.train_x is None:
            self.train_x = dataX
            self.train_y = dataY
        else:
            self.train_x = np.r_[self.train_x, dataX]
            self.train_y = np.r_[self.train_y, dataY]

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: a numpy array with each row corresponding to a query.
        @returns the estimated values according to the saved model.
        """
        pred_y = np.zeros(points.shape[0])

        for i in range(pred_y.shape[0]):
            test_x = points[i, :]
            distances = self._calculate_distance(test_x)
            smallest_index = np.argpartition(distances, self.k)
            pred_y[i] = self.train_y[smallest_index[:self.k]].mean()

        return pred_y

    def _calculate_distance(self, test_x):
        """
        @summary: Calculates Euclidian distance
        @param test_x: one instance of test data
        @returns Euclidian distance between given point and training data.
        """
        dist = self.train_x - test_x
        dist = np.sqrt(np.power(dist, 2).sum(axis=1))
        return dist

    def get_info(self):
        """
        @summary: prints internal info about the learner.
        """
        return 'knn learner (k={})'.format(self.k)


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
