'''

'''

import datetime as dt
import RTLearner as rt
import util as ut
import marketsim
import rule_based as rb
import numpy as np
import indicators as ind
import pandas as pd
import BagLearner as bl





def get_data(sym, train_st, train_end, test_st, test_end, time_per):
    # get data
    train_data = ind.run_analysis(sym=sym,
                                  start_date=train_st,
                                  end_date=train_end,
                                  lookback_pers=time_per,
                                  draw_charts=False,
                                  normalise=False)

    train_data.ix[:, 0] = train_data.ix[:, 0].rolling(time_per).sum().shift(-time_per)
    train_data.dropna(how='any', axis=0, inplace=True)

    train_x = train_data.ix[:, 1:]
    train_mean = train_x.mean(axis=0)
    train_std = train_x.std(axis=0)
    train_x = train_x.sub(train_mean).divide(train_std, axis=1)
    train_y = train_data.ix[:, 0]

    test_data = ind.run_analysis(sym=sym,
                                 start_date=test_st,
                                 end_date=test_end,
                                 lookback_pers=time_per,
                                 draw_charts=False,
                                 normalise=False)

    test_data.ix[:, 0] = test_data.ix[:, 0].rolling(time_per).sum().shift(-time_per)
    test_data.dropna(how='any', axis=0, inplace=True)

    test_x = test_data.ix[:, 1:]
    test_x = test_x.sub(train_mean).divide(train_std, axis=1)
    test_y = test_data.ix[:, 0]

    return train_x, train_y, test_x, test_y


def create_ml_portfolio(sym,
                        train_st, train_end, test_st, test_end,
                        initial_cash, time_per):

    ml_orders = './orders/ML_based.csv'
    y_buy = 0.02
    y_sell = -0.02
    train_x, train_y, test_x, test_y = get_data(sym,
                                                train_st, train_end,
                                                test_st, test_end,
                                                time_per)

    train_y = np.where(train_y > y_buy, 1,
                       np.where(train_y < y_sell, -1, 0))

    learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 10},
                            bags=20, boost=True, verbose=False)

    learner.addEvidence(train_x.values, train_y)
    pred_y = learner.query(train_x.values)

    position = pd.DataFrame(0, index=test_x.index, columns=[sym])
    start = 0
    while start < pred_y.shape[0]:
        if pred_y[start] == 0:
            start += 1
        else:
            pos = np.sign(pred_y[start]) * 200
            position.ix[start:start + time_per] = pos
            start += time_per

    position.ix[-1, :] = 0
    orders = position.diff(1)
    orders.ix[0, :] = position.ix[0, :]  # fix initial day position
    orders.dropna(axis=0, how='all', inplace=True)  # drop 'no-order' rows
    orders = orders[(orders != 0).all(1)]  # drop no-trade days

    ut.write_orders_to_csv(orders, ml_orders)
    ml_portfolio = marketsim.compute_portvals(orders_file=ml_orders,
                                              start_val=initial_cash)
    ml_portfolio = ml_portfolio[ml_portfolio.columns[0]]
    ml_portfolio = ut.standardise_timeperiod(ml_portfolio, test_st,
                                             test_end, initial_cash)
    ml_portfolio.rename('ML-based portfolio', inplace=True)
    return ml_portfolio, orders


def test_code(sym, initial_cash, train_st, train_end, test_st, test_end,
              verbose=False):

    # benchmark, buy 200 at AAPL at initial date, sell them at ending date
    amount = 200
    benchmark = ut.create_benchmark(sym, amount, initial_cash,
                                    test_st, test_end)

    # run manual strategy
    time_per = 21
    rule_portfolio, orders = rb.create_rule_based_portfolio(sym,
                                                            test_st, test_end,
                                                            initial_cash,
                                                            time_per)

    # run machine learning strategy
    ml_portfolio, orders = create_ml_portfolio(sym,
                                               train_st, train_end,
                                               test_st, test_end,
                                               initial_cash, time_per)

    # print information
    if verbose:
        ut.draw_charts([benchmark, rule_portfolio, ml_portfolio], orders)


if __name__ == '__main__':
    sym = 'AAPL'
    initial_cash = 100000
    training_start = dt.datetime(2008, 1, 1)
    training_end = dt.datetime(2009, 12, 31)
    test_start = dt.datetime(2010, 1, 1)
    test_end = dt.datetime(2011, 12, 31)

    test_code(sym, initial_cash,
              training_start, training_end, training_start, training_end,
              verbose=True)

    test_code(sym, initial_cash,
              training_start, training_end, test_start, test_end,
              verbose=True)
