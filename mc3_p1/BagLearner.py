"""
Bagging machine learning algorithm
Utilises random tree algorithms
"""

import numpy as np
import RTLearner as rt


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
        bootstrap = np.random.choice(data_x.shape[0],
                                     size=(data_x.shape[0], self.bags),
                                     replace=True)

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

        prediction = np.array(results).mean(axis=0)
        return prediction


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
