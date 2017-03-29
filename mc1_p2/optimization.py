"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo
import math


def compute_portfolio_stats(prices,
                            allocs=[0.1, 0.2, 0.3, 0.4],
                            rfr=0.0, sf=252.0):

    # Calculate portfolio performance
    normalized_prices = prices / prices.ix[0]
    alloced = normalized_prices * allocs
    port_val = alloced.sum(axis=1)

    # Calculate daily returns
    daily_returns = port_val / port_val.shift(1) - 1
    daily_returns = daily_returns[1:]

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr = port_val.ix[-1] / port_val.ix[0] - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = math.sqrt(sf) * (adr - rfr) / sddr

    return cr, adr, sddr, sr


# This is the function to minimize, negative of Sharpe ratio
def get_negative_sharpe(allocs, prices):
    # Calculate portfolio performance
    normalized_prices = prices / prices.ix[0]
    alloced = normalized_prices * allocs
    port_val = alloced.sum(axis=1)

    # Calculate daily returns
    daily_returns = port_val / port_val.shift(1) - 1
    daily_returns = daily_returns[1:]

    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = math.sqrt(252) * (adr - 0) / sddr

    return -1 * sr


def optimize_sharpe_ratio(prices):
    # Generate initial guess
    allocs = np.empty(len(prices.columns), dtype=np.float)
    allocs.fill(1. / len(prices.columns))

    # Constraints
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})

    # Bounds: variables must be inbetween [0,1]
    bnds = ((0, 1), (0, 1), (0, 1), (0, 1))

    result = spo.minimize(get_negative_sharpe, allocs, args=(prices,),
                          bounds=bnds, constraints=cons,
                          method='SLSQP', options={'disp': False})

    return result.x


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1),
                       syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    allocs = optimize_sharpe_ratio(prices)
    # compute stats
    cr, adr, sddr, sr = compute_portfolio_stats(prices, allocs)

    # Get daily portfolio value
    normalized_prices = prices / prices.ix[0]
    alloced = normalized_prices * allocs
    port_val = alloced.sum(axis=1)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY],
                            keys=['Portfolio', 'SPY'], axis=1)
        df_temp = df_temp / df_temp.ix[0]
        plot_data(df_temp, "Daily portfolio value and SPY",
                  'Normalized price', 'Time')

    return allocs, cr, adr, sddr, sr


# This function WILL NOT be called by the auto grader
# Don't assume that any variables defined here are available to your code
# It is only here to help you set up and test your code
def test_code():

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    '''
    # Example 1
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']

    # Example 2
    start_date = dt.datetime(2004,1,1)
    end_date = dt.datetime(2006,1,1)
    symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']

    # Example 3
    start_date = dt.datetime(2004,12,1)
    end_date = dt.datetime(2006,5,31)
    symbols = ['YHOO', 'XOM', 'GLD', 'HNZ']
    '''
    # Example 4
    start_date = dt.datetime(2005, 12, 1)
    end_date = dt.datetime(2006, 5, 31)
    symbols = ['YHOO', 'HPQ', 'GLD', 'HNZ']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date,
                                                        ed=end_date,
                                                        syms=symbols,
                                                        gen_plot=False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
