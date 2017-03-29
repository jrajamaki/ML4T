"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import math
from util import get_data, plot_data


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    # Get data
    # FORMAT: Date, Symbol, Order(Buy/Sell), #Shares
    orders = pd.read_csv(orders_file, index_col=[0])
    orders.sort_index()

    # Determine stock symbols, start and end date from the 'orders'
    ls_symbols = list(set(orders['Symbol'].values))
    start_date = orders.index[0]
    end_date = orders.index[-1]
    dates = pd.date_range(start_date, end_date)

    # Read prices to 'prices' dataframe, use adjusted close price
    # Automatically adds SPY to ensure correct market dates
    prices = get_data(ls_symbols, dates)
    # SPY has to dropped if not in traded stocks
    if 'SPY' not in ls_symbols:
        prices.drop('SPY', axis=1, inplace=True)

    # Add cash column to prices and set 1
    prices = prices.assign(Cash=np.ones(len(prices)))

    # Create 'trades' dataframe by copying 'prices' dataframe
    # Fill with zeros, then populate with trade amounts, needs for-loop
    values = prices.copy() * 0
    values['Cash'] = start_val

    for order in orders.iterrows():
        multiplier = 1
        if order[1]['Order'] == 'SELL':
            multiplier = -1
        date = order[0]
        stock = order[1]['Symbol']
        amount = order[1]['Shares']

        values.ix[date:, stock] += multiplier * amount * prices.ix[date:, stock]

        # Update cash column
        values.ix[date:, 'Cash'] += -1 * multiplier * amount * prices.ix[date, stock]

    # Calculate daily portfolio value by summing 'values' dataframe's columns
    port_val = values.sum(axis=1).to_frame()

    # Secret: on June 15th, 2011 ignore all orders

    # Leverage-function: borrowing money property
    # How much invested in market / liquidation value of account
    # leverage = (sum(abs(all stock positions))) / (sum(all stock positions) + cash)
    # Constraint: Leverage cannot exceed 1.5, if so reject the trade
    return port_val


def compute_portfolio_stats(port_val,
                            rfr=0.0, sf=252.0):

    # Calculate daily returns
    daily_returns = port_val / port_val.shift(1) - 1
    daily_returns = daily_returns[1:]

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr = port_val.ix[-1] / port_val.ix[0] - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = math.sqrt(sf) * (adr - rfr) / sddr

    return cr, adr, sddr, sr


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-short.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    '''
    # Get portfolio stats
    start_date = dt.datetime.date(portvals.index[0])
    end_date = dt.datetime.date(portvals.index[-1])
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals)

    spy_vals = get_data(['SPY'], pd.date_range(start_date, end_date))
    spy_vals = spy_vals[spy_vals.columns[0]]

    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_portfolio_stats(spy_vals)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])
    '''


if __name__ == "__main__":
    test_code()
