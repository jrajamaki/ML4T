'''
Executes manually created creating strategy using following rules:

Go long when three of the following indicators are true:
 - Price/SMA ratio > 1.05
 - Bollinger Band % > 1
 - RSI > 70
 - Momentum > 0
Go short when three of the following indicators are true:
 - Price/SMA ratio < 0.95
 - Bollinger Band % < 0
 - RSI < 30
 - Momentum < 0
'''

import pandas as pd
import util as ut
import datetime as dt
import marketsim
import indicators as ind
import numpy as np


def run_rule_based_strategy(syms, start_date, end_date, look_back):
    """
    @summary: Creates orders for manually created trading strategy.
    @param syms: List of stock symbols to be included in strategy.
    @param start_date: Start date of trading.
    @param end_date: End date of trading.
    @param look_back: Look back period for indicator analysis
    @return: Dataframe consisting of orders for dates that have orders.
    """

    dates = pd.date_range(start_date, end_date)
    orders = pd.DataFrame(np.nan, index=dates, columns=syms)

    for sym in syms:
        # for each stock get its' indicator data
        data = ind.run_analysis(sym=sym,
                                start_date=start_date,
                                end_date=end_date,
                                look_back=look_back,
                                draw_charts=False,
                                normalise=True)

        # drop useless stock price data
        data.drop(sym, axis=1, inplace=True)

        # signals for buying
        long_signal = pd.DataFrame(0, index=data.index, columns=data.columns)
        long_signal['SMA / P'] = (data['SMA / P'] < 0.95)
        long_signal['BBI'] = (data['BBI'] < 0)
        long_signal['RSI'] = (data['RSI'] < 0.3)
        long_signal['Momentum'] = (data['Momentum'] > 0)
        long_signal = (long_signal.sum(axis=1) > 2)

        # signals for shorting
        short_signal = pd.DataFrame(0, index=data.index, columns=data.columns)
        short_signal['SMA / P'] = (data['SMA / P'] > 1.05)
        short_signal['BBI'] = (data['BBI'] > 1)
        short_signal['RSI'] = (data['RSI'] > 0.7)
        short_signal['Momentum'] = (data['Momentum'] < 0)
        short_signal = (short_signal.sum(axis=1) > 2)

        position = pd.DataFrame(np.nan, index=data.index, columns=[sym])
        # need to go through every signal one by one,
        # because there is required holding period of 21 trading days
        mask = (long_signal | short_signal).values
        for coord, index in np.ndenumerate(np.where(mask)):
            amount = 200 if long_signal.iloc[index] else -200
            # if not in some position only then can enter into new position
            if np.isnan(position.iloc[index, 0]):
                position[index:index + look_back] = amount

        position.ix[-1] = 0  # sell everything on the last day
        position.fillna(0, inplace=True)
        orders[sym] = position.diff()

    orders.dropna(axis=0, how='all', inplace=True)  # drop non-business days
    orders = orders[(orders != 0).any(axis=1)]  # drop no-trade days
    return orders


def create_rule_based_portfolio(syms, start_date, end_date,
                                initial_cash, look_back):
    """
    @summary: runs manual trading strategy and calculates its' daily value.
    @param syms: List of stock symbols to be included in strategy.
    @param start_date: Start date of trading.
    @param end_date: End date of trading.
    @param initial_cash: Amount of cash available to use for trading.
    @param look_back: Look back period for indicator analysis
    @return rule_portfolio: Portfolio's daily value in dataframe
    @return orders: Dataframe consisting of orders for dates that have orders.

    """

    orders = run_rule_based_strategy(syms=syms,
                                     start_date=start_date,
                                     end_date=end_date,
                                     look_back=look_back)

    rule_based_orders = './orders/rule_based.csv'
    ut.write_orders_to_csv(orders, rule_based_orders)

    rule_portfolio = marketsim.compute_portvals(orders_file=rule_based_orders,
                                                start_val=initial_cash)
    rule_portfolio = rule_portfolio[rule_portfolio.columns[0]]
    rule_portfolio = ut.standardise_timeperiod(rule_portfolio, start_date,
                                               end_date, initial_cash)
    rule_portfolio.rename('Rule-based portfolio', inplace=True)
    return rule_portfolio, orders


def test_code(syms, start_date, end_date, initial_cash, verbose=False):

    # benchmark, buy 200 at AAPL at initial date, sell them at ending date
    amount = 200
    benchmark = ut.create_benchmark(syms[0], amount, initial_cash,
                                    start_date, end_date)

    # run manual strategy
    look_back = 21
    rule_portfolio, orders = create_rule_based_portfolio(syms,
                                                         start_date, end_date,
                                                         initial_cash,
                                                         look_back)

    # print information
    if verbose:
        ut.draw_charts([benchmark, rule_portfolio], orders)


if __name__ == '__main__':
    syms = ['AAPL', 'GOOG']
    initial_cash = 100000

    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    test_code(syms, start_date, end_date, initial_cash, verbose=True)

    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    test_code(syms, start_date, end_date, initial_cash, verbose=True)
