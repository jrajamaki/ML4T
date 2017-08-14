"""
This modules creates optimal portfolio by peeking into the future.
This way it is easy to calculate maximum return on the given time period.
"""

import pandas as pd
import util as ut
import datetime as dt
import marketsim
import numpy as np


def determine_optimal_orders(syms, start_date, end_date):
    """
    @summary: Given the time period determine optimal orders
              by peeking into future.
    @param sym: Stock symbol(s) to determine optimal orders for.
    @param start_date: The starting date from which orders are created.
    @param end_date: The ending date for the order creation.
    @return: Dataframe consisting of optimal orders for dates that have orders.
    """

    # get price data
    dates = pd.date_range(start_date, end_date)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols

    # calculate daily returns and shift them to previous day,
    # so that we can know whether stock will go or down next day
    # and calculate optimal positions for each day
    daily_rets = prices.diff(periods=1).shift(-1)
    position = np.sign(daily_rets) * 200
    position.ix[-1, :] = 0  # sell everything on last day

    # then calculate optimal orders per date
    orders = pd.DataFrame(0, index=position.index, columns=syms)
    orders = position.diff(1)
    orders.ix[0, :] = position.ix[0, :]  # fix initial day position

    # drop 'no-trade' days
    ords = orders[(orders != 0).any(axis=1)]
    return ords


def create_optimal_portfolio(syms, initial_cash, start_date, end_date):
    """
    @summary: Calculates optimal portfolio's daily value
    @param syms: Stock symbol(s) that are included in the optimal portfolio.
    @param initial_cash: Amount of cash available to use in trading.
    @param start_date: Starting date for order creation.
    @param end_date: Ending date for order creation.
    @return: The optimal portfolio's daily value in Pandas dataframe
    """
    orders = determine_optimal_orders(syms=syms,
                                      start_date=start_date,
                                      end_date=end_date)
    optimal_orders = './orders/optimal_orders.csv'
    ut.write_orders_to_csv(orders, optimal_orders)

    # calculate portfolio value
    optimal_portfolio = marketsim.compute_portvals(orders_file=optimal_orders,
                                                   start_val=initial_cash)

    optimal_portfolio = optimal_portfolio[optimal_portfolio.columns[0]]
    optimal_portfolio.rename('Optimal portfolio', inplace=True)

    return optimal_portfolio


def test_code(syms, start_date, end_date, initial_cash, draw_charts=False):

    # benchmark buy 200 of stocks of the first stock symbol in 'sym' stock list
    # at initial date, and sell them at ending date
    amount = 200
    benchmark = ut.create_benchmark(syms[0], amount, initial_cash,
                                    start_date, end_date)

    # the most optimal, future-peeking portfolio
    optimal_portfolio = create_optimal_portfolio(syms, initial_cash,
                                                 start_date, end_date)

    # print information
    if draw_charts:
        ut.draw_charts([benchmark, optimal_portfolio])


if __name__ == '__main__':
    syms = ['AAPL', 'GOOG']
    initial_cash = 100000

    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    test_code(syms, start_date, end_date, initial_cash, draw_charts=True)

    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    test_code(syms, start_date, end_date, initial_cash, draw_charts=True)
