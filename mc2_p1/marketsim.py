"""MC2-P1: Market simulator."""

import pandas as pd
import datetime as dt
import math
from util import get_data


# Leverage-function: borrowing money property
# How much invested in market / liquidation value of account
def check_leverage(new_pos):
    leverage = (new_pos[:-1].abs().sum()) / (new_pos.sum())

    if abs(leverage.values) > 2:
        return False

    return True


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    # Get data
    # FORMAT: Date, Symbol, Order(Buy/Sell), # of shares
    orders = pd.read_csv(orders_file, index_col=[0])
    orders = orders.sort_index()

    # Determine stock symbols, start and end date from the 'orders'
    ls_symbols = list(set(orders['Symbol'].values))
    start_date = orders.index[0]
    end_date = orders.index[-1]
    dates = pd.date_range(start_date, end_date)

    # Read prices to 'prices' dataframe, use adjusted close price
    # Automatically adds SPY to ensure correct market dates
    prices = get_data(ls_symbols, dates)

    # Create dataframe to follow daily positions
    # At first no trades, thus every day only cash
    position = pd.DataFrame(index=prices.index, columns=prices.columns)
    position = position.fillna(0)
    position['Cash'] = start_val

    for order in orders.iterrows():
        multiplier = 1
        if order[1]['Order'] == 'SELL':
            multiplier = -1

        # Parsing
        date = order[0]
        stock = order[1]['Symbol']
        amount = order[1]['Shares']
        trade_val = multiplier * amount * prices.ix[date:, stock]

        # Create new position
        new_pos = position.ix[date].copy().to_frame()
        new_pos.ix[stock] += trade_val
        new_pos.ix['Cash'] -= trade_val

        # Constraint: Leverage cannot exceed 2, if so reject the trade
        # in 2016 limit 2, in 2017 limit 1.5
        under_leverage_limit = check_leverage(new_pos)
        if under_leverage_limit:
            position.ix[date:, stock] += trade_val
            # Update cash column
            position.ix[date:, 'Cash'] -= trade_val[date]

    # Calculate daily portfolio value by summing 'position' dataframe's columns
    port_val = position.sum(axis=1).to_frame()

    # Secret: on June 15th, 2011 ignore all orders
    # This has not been implemented

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

    of = "./testcases2016/orders-12-modified.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    start_date = dt.datetime.date(portvals.index[0])
    end_date = dt.datetime.date(portvals.index[-1])
    cr, adr, stddr, sr = compute_portfolio_stats(portvals)

    # Portfolio statistics
    print "Date Range: {} to {}".format(start_date, end_date)
    print "Sharpe Ratio of Fund: {}".format(sr)
    print "Cumulative Return of Fund: {}".format(cr)
    print "Standard Deviation of Fund: {}".format(stddr)
    print "Average Daily Return of Fund: {}".format(adr)
    print "Final Portfolio Value: {}".format(portvals[-1])


if __name__ == "__main__":
    test_code()
