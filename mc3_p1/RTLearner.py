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
        print self.tree

    def build_tree(self, data):
        # recursion finishing conditions
        # case 1: 'minimum leaf size reached'
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, data[:, -1].mean(), -1, -1]])

        # case 2: 'all values the same'
        if (data[:, -1] == data[0, -1]).all():
            return np.array([[-1, data[0, -1], -1, -1]])

        # else further parse tree
        else:
            # select randomly feature and data range on which to split on
            # deduct Y values from the random generators's range
            feature = np.random.randint(0, data.shape[1] - 1)

            # splitting value is mean of two random data points
            index = np.random.randint(0, data.shape[0], 2)
            value = data[index, feature].mean()

            # execute recursion on the left side and the right side
            left_tree = self.build_tree(data[data[:, feature] <= value])
            right_tree = self.build_tree(data[data[:, feature] > value])

            # build the tree
            root = np.array([[feature, value, 1, left_tree.shape[0] + 1]])
            return np.r_['0,2', root, left_tree, right_tree]

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: a numpy array of data to make prediction on.
        @returns the estimated values according to the saved model.
        """
        return 0


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
