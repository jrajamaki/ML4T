"""
Random forest learning algorithm
"""

import numpy as np


class RTLearner(object):

    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # concatenate the data and build the random decision tree
        # as described in A. Cutler's paper in here:
        # http://www.interfacesymposia.org/I01/I2001Proceedings/ACutler/ACutler.pdf
        data = np.c_[dataX, dataY]
        self.tree = self.build_tree(data)

        # fix NaN values
        self.tree[np.isnan(self.tree)] = dataY.mean()

    def build_tree(self, data):
        # recursion finishing conditions
        # case 0: 'no data'
        if data.shape[0] == 0:
            # make a row for empty data slices and
            # later estimate them as mean of the data
            # not perfect but optimal enough
            return np.array([[-1, np.NaN, 0, 0]])

            # returning empty numpy array would break indexing
            # return np.empty(shape=(0, 4))

        # case 1: 'minimum leaf size reached'
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, data[:, -1].mean(), 0, 0]])

        # case 2: 'all values the same'
        if (data[:, -1] == data[0, -1]).all():
            return np.array([[-1, data[0, -1], 0, 0]])

        else:
            # random generation of splitting
            # deduct Y column from the random generators's range
            feature = np.random.randint(0, data.shape[1] - 1)
            # splitting value is mean of two random data points
            index = np.random.randint(0, data.shape[0], 2)
            value = data[index, feature].mean()

            left_tree = self.build_tree(data[data[:, feature] <= value])
            right_tree = self.build_tree(data[data[:, feature] > value])
            root = np.array([[feature, value, 1, left_tree.shape[0] + 1]])
            return np.r_['0,2', root, left_tree, right_tree]

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: a numpy array of data to make prediction on.
        @returns the estimated values according to the saved model.
        """
        # variables indexes correct leaf in preconstructed decision tree
        # initially each data point indexes root node
        prediction = np.zeros(points.shape[0], dtype=int)

        # the features of each data point to be evaluated
        features = self.tree[prediction, 0].astype(int)

        prediction_ready = False
        # predict until every point has reached leaf node
        while not prediction_ready:

            # values of data points at feature to be evaluated
            values = points[np.arange(len(points)), features]
            # splitting values of each node
            split_value = self.tree[prediction, 1]
            # evaluation of index whether to take right step or left step
            next_step = (values > split_value) * 1 + 2

            # update indexes of predictions
            prediction += self.tree[prediction, next_step].astype(int)

            # check termination condition
            features = self.tree[prediction, 0].astype(int)
            prediction_ready = np.all(self.tree[prediction, 0] == -1)

        return self.tree[prediction, 1]


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
