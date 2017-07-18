"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import RTLearner as rt
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    f = open(sys.argv[1])
    data = np.array([map(float, s.strip().split(',')) for s in f.readlines()])

    # compute how much of the data is training and testing
    train_rows = int(math.floor(0.6 * data.shape[0]))
    test_rows = int(data.shape[0] - train_rows)

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print test_x.shape
    print test_y.shape

    print '-- LINEAR REGRESSION --'
    learner = lrl.LinRegLearner(verbose=True)
    learner.addEvidence(train_x, train_y)

    print "In sample results"
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print "RMSE: ", rmse
    c = np.corrcoef(pred_y, y=train_y)
    print "corr: ", c[0, 1]

    print "Out of sample results"
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print "RMSE: ", rmse
    c = np.corrcoef(pred_y, y=test_y)
    print "corr: ", c[0, 1]

    print '-- RANDOM TREE --'
    learner = rt.RTLearner(leaf_size=1, verbose=False)  # constructor
    learner.addEvidence(train_x, train_y)  # training step

    print "In sample results"
    pred_y = learner.query(train_x)  # query
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print "RMSE: ", rmse
    c = np.corrcoef(pred_y, y=train_y)
    print "corr: ", c[0, 1]
