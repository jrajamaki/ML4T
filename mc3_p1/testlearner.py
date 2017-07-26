"""
Test a learner.  originally (c) 2015 Tucker Balch
but significant changes were made
"""

import numpy as np
import math
import LinRegLearner as lrl
import RTLearner as rt
import BagLearner as bl
import KNNLearner as knn
import sys


def print_results(in_sample_results, out_of_sample_results):
    in_sample_results = np.array(in_sample_results)
    print "In sample results",
    print "RMSE: {:f}".format(in_sample_results[:, 0].mean()),
    print "corr: {:f}".format(in_sample_results[:, 1].mean()),

    out_of_sample_results = np.array(out_of_sample_results)
    print "Out of sample results",
    print "RMSE: {:f}".format(out_of_sample_results[:, 0].mean()),
    print "corr: {:f}".format(out_of_sample_results[:, 1].mean())


def calculate_results(data_y, pred_y):
    rmse = math.sqrt(((data_y - pred_y) ** 2).sum() / data_y.shape[0])
    c = np.corrcoef(pred_y, y=data_y)
    return rmse, c[0, 1]


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
    k_fold = 3
    verbose = False

    learners = []

    learner = lrl.LinRegLearner
    learner_name = 'LINEAR REGRESSION'
    learner_kwargs = {'verbose': verbose}
    learners.append((learner, learner_name, learner_kwargs))

    learner = rt.RTLearner
    learner_name = 'RANDOM TREE'
    learner_kwargs = {'leaf_size': 1, 'verbose': verbose}
    learners.append((learner, learner_name, learner_kwargs))

    learner = rt.RTLearner
    learner_name = 'RANDOM TREE'
    learner_kwargs = {'leaf_size': 50, 'verbose': verbose}
    learners.append((learner, learner_name, learner_kwargs))

    learner = knn.KNNLearner
    learner_name = 'kNN'
    learner_kwargs = {'k': 3, 'verbose': verbose}
    learners.append((learner, learner_name, learner_kwargs))

    learner = bl.BagLearner
    learner_name = 'BAGGING'
    learner_kwargs = {'learner': rt.RTLearner, 'kwargs': {'leaf_size': 20},
                      'bags': 1, 'boost': False, 'verbose': verbose}
    learners.append((learner, learner_name, learner_kwargs))

    learner = bl.BagLearner
    learner_name = 'BAGGING'
    learner_kwargs = {'learner': rt.RTLearner, 'kwargs': {'leaf_size': 20},
                      'bags': 20, 'boost': False, 'verbose': verbose}
    learners.append((learner, learner_name, learner_kwargs))

    learner = bl.BagLearner
    learner_name = 'BAGGING'
    learner_kwargs = {'learner': rt.RTLearner, 'kwargs': {'leaf_size': 20},
                      'bags': 20, 'boost': True, 'verbose': verbose}
    learners.append((learner, learner_name, learner_kwargs))


    for learner, name, args in learners:
        print '-- ', name, args, ' --'
        in_sample_results = []
        out_of_sample_results = []
        for train, test in rollforward_datasplit(data.shape[0], k_fold):
            train_x = data[train, 0:-1]
            train_y = data[train, -1]
            test_x = data[test, 0:-1]
            test_y = data[test, -1]

            lrn = learner(**args)
            lrn.addEvidence(train_x, train_y)
            pred_y = lrn.query(train_x)
            in_sample_results.append(calculate_results(train_y, pred_y))
            pred_y = lrn.query(test_x)
            out_of_sample_results.append(calculate_results(test_y, pred_y))
        print_results(in_sample_results, out_of_sample_results)
