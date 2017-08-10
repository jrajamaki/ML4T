import pandas as pd
import util as ut
import datetime as dt
import marketsim
import numpy as np


def determine_optimal_orders(syms, start_date, end_date):
    # get data
    dates = pd.date_range(start_date, end_date)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols

    # calculate daily returns and shift them to previous day,
    # so that we can know whether stock will go or down next day
    # and calculate optimal positions for each day
    daily_rets = prices.diff(periods=1).shift(-1)
    position = np.sign(daily_rets) * 200

    # then calculate optimal orders per date
    orders = position.diff(1)
    orders.ix[0, :] = position.ix[0, :]  # fix initial day position
    orders.dropna(axis=0, how='all', inplace=True)  # drop 'no-order' rows

    return orders


def create_optimal_portfolio(sym, initial_cash, start_date, end_date):

    orders = determine_optimal_orders(syms=[sym],
                                      start_date=start_date,
                                      end_date=end_date)
    optimal_orders = './orders/optimal_orders.csv'
    ut.write_orders_to_csv(orders, optimal_orders)
    optimal_portfolio = marketsim.compute_portvals(orders_file=optimal_orders,
                                                   start_val=initial_cash)

    optimal_portfolio = optimal_portfolio[optimal_portfolio.columns[0]]
    optimal_portfolio.rename('Optimal portfolio', inplace=True)
    return optimal_portfolio


def test_code(sym, start_date, end_date, initial_cash, verbose=False):

    # benchmark, buy 200 at AAPL at initial date, sell them at ending date
    amount = 200
    benchmark = ut.create_benchmark(sym, amount, initial_cash,
                                    start_date, end_date)

    # the most optimal, future-peeking portfolio
    optimal_portfolio = create_optimal_portfolio(sym, initial_cash,
                                                 start_date, end_date)

    # print information
    if verbose:
        ut.draw_charts([benchmark, optimal_portfolio])


if __name__ == '__main__':
    sym = 'AAPL'
    initial_cash = 100000

    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    test_code(sym, start_date, end_date, initial_cash, verbose=True)

    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    test_code(sym, start_date, end_date, initial_cash, verbose=True)
