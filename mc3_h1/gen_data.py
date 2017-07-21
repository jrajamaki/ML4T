"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np


# this function should return a dataset (X and Y) that will work
# better for linear regresstion than random trees
def best4LinReg(seed=5):
    prng = np.random.RandomState(int(seed))
    X = prng.normal(size=(1000, 10))
    Y = X[:, 0]
    return X, Y


def best4RT(seed=5):
    prng = np.random.RandomState(int(seed))
    X = prng.normal(size=(2, 10))
    Y = (X ** 2).sum(axis=1)
    return X, Y


if __name__ == "__main__":
    print "they call me Tim."
