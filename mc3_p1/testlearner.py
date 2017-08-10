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
    print "in sample",
    print "RMSE: {:f}".format(in_sample_results[:, 0].mean()),
    print "corr: {:f}".format(in_sample_results[:, 1].mean()),

    out_of_sample_results = np.array(out_of_sample_results)
    print "out of sample",
    print "RMSE: {:f}".format(out_of_sample_results[:, 0].mean()),
    print "corr: {:f}".format(out_of_sample_results[:, 1].mean())


def calculate_results(data_y, pred_y):
    rmse = math.sqrt(((data_y - pred_y) ** 2).sum() / data_y.shape[0])
    c = np.corrcoef(pred_y, y=data_y)
    return rmse, c[0, 1]


def rollforward_datasplit(length, no_of_folds):
    no_of_splits = no_of_folds + 1
    set_length = int(math.floor(length / (no_of_splits)))

    for i in xrange(1, no_of_splits):
        set_end = set_length * i
        yield np.arange(0, set_end), np.arange(set_end, set_end + set_length)


def determine_best_args(learners, data, verbose=False):
    no_of_folds = 3
    best_args = None
    best_score = -np.inf
    for learner, args in learners:
        lrn = learner(**args)
        in_sample_results = []
        out_of_sample_results = []

        for train, test in rollforward_datasplit(data.shape[0], no_of_folds):
            train_x = data[train, 0:-1]
            train_y = data[train, -1]
            test_x = data[test, 0:-1]
            test_y = data[test, -1]

            lrn.addEvidence(train_x, train_y)
            pred_y = lrn.query(train_x)
            in_sample_results.append(calculate_results(train_y, pred_y))
            pred_y = lrn.query(test_x)
            out_of_sample_results.append(calculate_results(test_y, pred_y))

            if out_of_sample_results[-1][1] > best_score:
                best_score = out_of_sample_results[-1][1]
                best_args = args
            if verbose:
                print_results(in_sample_results, out_of_sample_results)
    return best_args


def det_best_learner(learners, data, training_idx, test_idx, verbose=False):
    best_kwargs = determine_best_args(learners, data[training_idx])
    best_learner = learner(**best_kwargs)
    print best_learner.get_info()

    train_x = data[training_idx, 0:-1]
    train_y = data[training_idx, -1]
    test_x = data[test_idx, 0:-1]
    test_y = data[test_idx, -1]

    best_learner.addEvidence(train_x, train_y)

    pred_y = best_learner.query(train_x)
    in_sample_results = calculate_results(train_y, pred_y)
    pred_y = best_learner.query(test_x)
    out_of_sample_results = calculate_results(test_y, pred_y)

    print "in sample",
    print "RMSE: {:f}".format(in_sample_results[0]),
    print "corr: {:f}".format(in_sample_results[1]),

    print "out of sample",
    print "RMSE: {:f}".format(out_of_sample_results[0]),
    print "corr: {:f}".format(out_of_sample_results[1])


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)

    f = open(sys.argv[1])
    data = np.array([map(float, s.strip().split(',')) for s in f.readlines()])

    split_idx = int(data.shape[0] * 0.6)
    training_idx = np.arange(split_idx)
    test_idx = np.arange(split_idx, data.shape[0])
    no_of_folds = 3
    verbose = False

    learners = []
    learner = lrl.LinRegLearner
    learner_kwargs = {'verbose': verbose}
    learners.append((learner, learner_kwargs))
    det_best_learner(learners, data, training_idx, test_idx)

    learners = []
    learner = rt.RTLearner
    for i in range(1, 20):
        learner_kwargs = {'leaf_size': i, 'verbose': verbose}
        learners.append((learner, learner_kwargs))
    det_best_learner(learners, data, training_idx, test_idx)

    learners = []
    learner = knn.KNNLearner
    for i in range(1, 20):
        learner_kwargs = {'k': i, 'verbose': verbose}
        learners.append((learner, learner_kwargs))
    det_best_learner(learners, data, training_idx, test_idx)

    learners = []
    learner = bl.BagLearner
    for i in range(5, 10):
        for j in range(1, 20):
            learner_kwargs = {'learner': rt.RTLearner,
                              'kwargs': {'leaf_size': j},
                              'bags': i, 'boost': False, 'verbose': verbose}
            learners.append((learner, learner_kwargs))
    det_best_learner(learners, data, training_idx, test_idx)

    learners = []
    learner = bl.BagLearner
    for i in range(5, 10):
        for j in range(1, 20):
            learner_kwargs = {'learner': knn.KNNLearner,
                              'kwargs': {'k': j},
                              'bags': i, 'boost': False, 'verbose': verbose}
            learners.append((learner, learner_kwargs))
    det_best_learner(learners, data, training_idx, test_idx)

    learners = []
    learner = bl.BagLearner
    for i in range(5, 10):
        for j in range(1, 20):
            learner_kwargs = {'learner': rt.RTLearner,
                              'kwargs': {'leaf_size': j},
                              'bags': i, 'boost': True, 'verbose': verbose}
            learners.append((learner, learner_kwargs))
    det_best_learner(learners, data, training_idx, test_idx)
