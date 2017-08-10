"""
Bagging machine learning algorithm
Utilises random tree algorithm
"""

import numpy as np
from scipy.stats import mode


class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):

        self.bags = bags
        self.boost = boost
        self.learners = []
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))

    def addEvidence(self, data_x, data_y):
        """
        @summary: Add training data to learners
        @param data_x: X values of data to add
        @param data_y: the Y training values
        """
        weights = np.ones(data_y.shape[0]) / data_y.shape[0]

        if self.boost:
            for i in np.arange(1, self.bags + 1):

                # new bootstrap sample for current learner
                bootstrap = np.random.choice(data_x.shape[0],
                                             size=(data_x.shape[0]),
                                             replace=True,
                                             p=weights)

                # train and predict results current and
                # all previous learners using new bootstrap sample
                pred_y = np.ndarray(shape=(data_y.shape[0], i))
                for j in np.arange(i):
                    self.learners[j].addEvidence(data_x[bootstrap],
                                                 data_y[bootstrap])
                    pred_y[:, j] = self.learners[j].query(data_x)

                # Calculate new weights using mean of errors
                # and normalise them so that weigths.sum() == 1
                pred_y = pred_y.mean(axis=1)
                errors = self._calculate_errors(data_y, pred_y)
                weights = errors / errors.sum()

        # train all learners
        # if boosting was used, sample using optimised weights
        # if no boosting, use uniform weights
        bootstrap = np.random.choice(data_x.shape[0],
                                     size=(data_x.shape[0], self.bags),
                                     replace=True, p=weights)

        for i in np.arange(self.bags):
            self.learners[i].addEvidence(data_x[bootstrap[:, i], :],
                                         data_y[bootstrap[:, i]])

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: a numpy array of data to make prediction on.
        @returns the estimated values according to the saved model.
        """
        results = []
        for i in np.arange(len(self.learners)):
            results.append(self.learners[i].query(points))

        # majority voting
        prediction = mode(np.array(results), axis=0)[0].flatten()
        return prediction

    def _calculate_errors(self, data_y, pred_y):
        """
        @summary: calculate means square error between data and prediction
        @param data_y: original y values
        @param pred_y: predictions of data
        @returns errors for each prediction
        """

        errors = data_y - pred_y
        errors = np.sqrt(np.power(errors, 2))
        return errors

    def get_info(self):
        """
        @summary: prints internal info about the learner.
        """
        weak_learner = self.learners[0].get_info()
        info = 'bag learner'
        info += '(bags={}, weak learner={}, boosting={})'.format(self.bags,
                                                                 weak_learner,
                                                                 self.boost)
        return info


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
