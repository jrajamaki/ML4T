"""MLT: Utility code."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import math
import csv
import numpy as np
import marketsim


def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True, colname='Adj Close'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', colname],
                              na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def compute_portfolio_stats(port_val, rfr=0.0, sf=252.0):

    # Calculate daily returns
    daily_returns = port_val / port_val.shift(1) - 1
    daily_returns = daily_returns[1:]
    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr = port_val.ix[-1] / port_val.ix[0] - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = math.sqrt(sf) * (adr - rfr) / sddr

    return cr * 100, adr * 100, sddr, sr, port_val[-1]


def draw_charts(portfolios, *orders):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(2, 1)
    colors = ['black', 'blue', 'green']
    ax[0].set_prop_cycle('color', colors)

    data = []
    columns = []

    for portfolio in portfolios:
        data.append(compute_portfolio_stats(portfolio))
        columns.append(portfolio.name)
        portfolio.divide(portfolio.ix[0]).plot(ax=ax[0], legend=True)

    if orders:
        orders = orders[0]
        pos = 0
        for index, order in orders.iterrows():
            pos += int(order[0])
            if pos != 0:
                color = 'green' if pos == 200 else 'red'
                ax[0].axvline(index, color=color)

    data = np.transpose(np.asarray(data))
    rows = ("Cumulative return (%)", "Average daily return (%)",
            "Standard deviation", "Sharpe ratio", "Final value ($)")
    ax[1].table(cellText=data, colLabels=columns, rowLabels=rows,
                cellLoc='center', loc='center')
    ax[1].axis('tight')
    ax[1].axis('off')

    plt.show()


def write_orders_to_csv(orders, filename):
    syms = orders.columns

    # FORMAT: Date, Symbol, Order(Buy/Sell), # of shares
    with open(filename, 'wb') as file:
        writer = csv.writer(file)
        writer.writerow(('Date', 'Symbol', 'Order', 'Shares'))
        for index, row in orders.iterrows():
            date = index.strftime('%Y-%m-%d')

            for i in range(len(syms)):
                amount = abs(row[i])
                if amount == 0:
                    continue

                symbol = syms[i]
                if row[i] < 0:
                    order_type = 'SELL'
                else:
                    order_type = 'BUY'

                writer.writerow((date, symbol, order_type, amount))


def print_full(df):
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')


def create_benchmark(sym, amount, initial_cash, start_date, end_date):
    # get data
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([sym], dates)  # automatically adds SPY
    prices = prices_all[[sym]]  # only portfolio symbols

    output = './orders/benchmark.csv'
    with open(output, 'wb') as file:
        writer = csv.writer(file)
        writer.writerow(('Date', 'Symbol', 'Order', 'Shares'))
        start = prices.index[0]
        end = prices.index[-1]
        writer.writerow((start, sym, 'BUY', '200'))
        writer.writerow((end, sym, 'SELL', '200'))

    benchmark = marketsim.compute_portvals(orders_file=output,
                                           start_val=initial_cash)
    benchmark = benchmark[benchmark.columns[0]]
    benchmark.rename('200 of ' + sym, inplace=True)
    return benchmark


def standardise_timeperiod(portfolio, start_date, end_date, initial_cash):
    dates = pd.date_range(start_date, end_date)  # real analysis period
    portfolio = portfolio.reindex(dates)
    if np.isnan(portfolio[0]):  # no trade on first day
        portfolio[0] = initial_cash

    portfolio.ffill(inplace=True)
    return portfolio

'''
def print_results(portfolio):
    # Get portfolio stats
    start_date = dt.datetime.date(portfolio.index[0])
    end_date = dt.datetime.date(portfolio.index[-1])
    cr, adr, stddr, sr, fv = compute_portfolio_stats(portfolio)

    # Portfolio statistics
    print portfolio.name, "results"
    print "Date Range: {} to {}".format(start_date, end_date)
    print "Sharpe Ratio: {:2.4}".format(sr)
    print "Cumulative Return: {:2.4}%".format(cr * 100)
    print "Standard Deviation: {:.4}".format(stddr)
    print "Average Daily Return: {:.2}%".format(adr * 100)
    print "Final Value: {}".format(fv)


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()    


def save_benchmark_data(sym, start_date, end_date, output='benchmark.csv'):
    # get data
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([sym], dates)  # automatically adds SPY
    prices = prices_all[[sym]]  # only portfolio symbols

    with open(output, 'wb') as file:
        writer = csv.writer(file)
        writer.writerow(('Date', 'Symbol', 'Order', 'Shares'))
        start = prices.index[0]
        end = prices.index[-1]
        writer.writerow((start, sym, 'BUY', '200'))
        writer.writerow((end, sym, 'SELL', '200'))
'''