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
        data = append(dataX, dataY, axis=1)
        self.tree = build_tree(data)

    def build_tree(data):
        # recursion finishing conditions
        # case 1: 'minimum leaf size reached'
        if data[-1].shape[0] < self.leaf_size:
            return [-1, data[-1].mean(), -1, -1]

        # case 2: 'all values the same'
        if (data[-1] == data[-1, 0]).all():
            return [-1, data[-1, 0], -1, -1]

        # else further parse tree
        else:
            # select randomly feature and data range on which to split on
            # deduct Y values from the random generators's range
            random_feature = np.random.randint(0, data.shape[1] - 1)

            # split val is mean of two random data points
            random_index = np.random.randint(0, data.shape[0], 2)
            split_val = data[random_index, random_feature].mean()

            # random_index1 = np.random.randint(data.shape[0])
            # random_index2 = np.random.randint(data.shape[0])
            # split_val = (data[random_index1, random_feature] +
            #              data[random_index2, random_feature]) / 2

            # execute recursion on the left side and the right side trees
            left_tree = build_tree(data[data[:, random_feature] <= split_val])
            right_tree = build_tree(data[data[:, random_feature] > split_val])

            # build the tree
            root = [random_feature, split_val, 1, left_tree.shape[0] + 1]
            return append(root, left_tree, right_tree)

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: a numpy array of data to make prediction on.
        @returns the estimated values according to the saved model.
        """
        return 0


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
