"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import RTLearner as rt
import BagLearner as bl
import sys


def print_results(data_y, pred_y):
    rmse = math.sqrt(((data_y - pred_y) ** 2).sum() / data_y.shape[0])
    print "RMSE: {:f}".format(rmse),
    c = np.corrcoef(pred_y, y=data_y)
    print "corr: {:f}".format(c[0, 1])


def rollforward_datasplit(length, k_fold):
    no_of_splits = k_fold + 1
    set_length = int(math.floor(length / (no_of_splits)))

    for i in xrange(1, no_of_splits):
        set_end = set_length * i
        yield np.arange(0, set_end), np.arange(set_end, set_end + set_length)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)

    f = open(sys.argv[1])
    data = np.array([map(float, s.strip().split(',')) for s in f.readlines()])
    k_fold = 10

    print '-- LINEAR REGRESSION --'
    for train, test in rollforward_datasplit(data.shape[0], k_fold):
        train_x = data[train, 0:-1]
        train_y = data[train, -1]
        test_x = data[test, 0:-1]
        test_y = data[test, -1]

        learner = lrl.LinRegLearner(verbose=False)
        learner.addEvidence(train_x, train_y)
        print "In sample results    ",
        pred_y = learner.query(train_x)
        print_results(train_y, pred_y)
        print "Out of sample results",
        pred_y = learner.query(test_x)
        print_results(test_y, pred_y)

    print '-- RANDOM TREE (leaf_size=1) --'
    for train, test in rollforward_datasplit(data.shape[0], k_fold):
        train_x = data[train, 0:-1]
        train_y = data[train, -1]
        test_x = data[test, 0:-1]
        test_y = data[test, -1]

        learner = rt.RTLearner(leaf_size=1, verbose=False)
        learner.addEvidence(train_x, train_y)
        print "In sample results    ",
        pred_y = learner.query(train_x)
        print_results(train_y, pred_y)
        print "Out of sample results",
        pred_y = learner.query(test_x)
        print_results(test_y, pred_y)

    print '-- BAGGING (bags=1, leaf_size=20) --'
    for train, test in rollforward_datasplit(data.shape[0], k_fold):
        train_x = data[train, 0:-1]
        train_y = data[train, -1]
        test_x = data[test, 0:-1]
        test_y = data[test, -1]

        learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 20},
                                bags=1, boost=False, verbose=False)
        learner.addEvidence(train_x, train_y)
        print "In sample results    ",
        pred_y = learner.query(train_x)
        print_results(train_y, pred_y)
        print "Out of sample results",
        pred_y = learner.query(test_x)
        print_results(test_y, pred_y)

    print '-- BAGGING (bags=20, leaf_size=20) --'
    for train, test in rollforward_datasplit(data.shape[0], k_fold):
        train_x = data[train, 0:-1]
        train_y = data[train, -1]
        test_x = data[test, 0:-1]
        test_y = data[test, -1]

        learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 20},
                                bags=20, boost=False, verbose=False)
        learner.addEvidence(train_x, train_y)
        print "In sample results    ",
        pred_y = learner.query(train_x)
        print_results(train_y, pred_y)
        print "Out of sample results",
        pred_y = learner.query(test_x)
        print_results(test_y, pred_y)
