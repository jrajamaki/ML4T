"""
    Calculates following technical indicators:
 - Simple moving average / price index
 - Bollinger Bands Index
 - Relative strength index
 - Momentum

For more information about the indicators please refer
https://en.wikipedia.org/wiki/Technical_analysis
"""

import pandas as pd
import util as ut
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def standard_score(df):
    """
    @summary: Normalises Pandas dataframe.
    @param df: Dataframe to be normalised.
    @return: Normalised dataframe.
    """
    return (df - df.mean()) / df.std()


def run_analysis(sym, start_date, end_date, look_back=21,
                 normalise=False, draw_charts=False):
    """
    @summary: Normalises Pandas dataframe.
    @param sym: Stock symbol to make the analysis on.
    @param start_date: The starting date to commence analysis.
    @param end_date: The ending date to stop analysis.
    @param look_back: Number of days to the past to calculate indicators.
    @param normalise: If True normalises indicators in resulting dataframe.
    @param draw_charts: If True draws charts of indicators.
    @return: Pandas dataframe of stock price and indicator data.
    """

    # to simplify code
    per = look_back
    sym = [sym]

    # get data
    dates = pd.date_range(start_date, end_date)
    prices_all = ut.get_data(sym, dates)  # automatically adds SPY
    prices = prices_all[sym]  # only portfolio symbols
    n_prices = prices.divide(prices.ix[0])  # normalise

    # Smooth moving average / price index
    smap = n_prices.copy()
    smap['SMA'] = n_prices.rolling(per, min_periods=per).mean()
    smap['SMA / P'] = prices.rolling(per, min_periods=per).mean() / prices

    # Bolling bands index
    bb = n_prices.copy()
    bb['SMA'] = n_prices.rolling(per, min_periods=per).mean()
    bb['MSD'] = n_prices.rolling(per, min_periods=per).std()  # moving std
    bb['Upper BB'] = bb['SMA'] + 2 * bb['MSD']
    bb['Lower BB'] = bb['SMA'] - 2 * bb['MSD']
    bb['BBI'] = (bb.ix[:, 0] - bb['Lower BB']) / (bb['Upper BB'] -
                                                  bb['Lower BB'])

    # Relative strenght index
    rsi = n_prices.copy()
    rsi['Rets'] = rsi.diff(periods=1)
    rsi['Gain'] = rsi['Rets'][rsi['Rets'] > 0].rolling(per).sum()
    rsi['Loss'] = rsi['Rets'][rsi['Rets'] < 0].rolling(per).sum() * -1
    rsi['Avg loss'] = (rsi['Loss'].fillna(method='ffill')) / per
    rsi['Avg gain'] = (rsi['Gain'].fillna(method='ffill')) / per
    rsi['RS'] = rsi['Avg gain'] / rsi['Avg loss']
    rsi['RSI'] = 100 - (100 / (1 + rsi['RS']))

    # Momentum index
    momentum = n_prices.copy()
    momentum['Momentum'] = momentum.divide(momentum.shift(per)) - 1

    if normalise:
        smap['SMA / P'] = standard_score(smap['SMA / P'])
        bb['BBI'] = standard_score(bb['BBI'])
        rsi['RSI'] = standard_score(rsi['RSI'])
        momentum['Momentum'] = standard_score(momentum['Momentum'])

    daily_rets = prices.divide(prices.shift(1)) - 1
    key_data = pd.concat([daily_rets, smap['SMA / P'], bb['BBI'],
                          rsi['RSI'], momentum['Momentum']], axis=1)

    # plot indicators
    plt.close('all')
    if draw_charts:
        f, ax = plt.subplots(2, 2, sharex='col', sharey='row')
        '''
        smap.plot(ax=ax[0, 0], legend=True, title='SMA / P')
        bb.plot(ax=ax[0, 1], legend=True, title='BB Index')
        rsi.plot(ax=ax[1, 0], legend=True, title='RS Index')
        momentum.plot(ax=ax[1, 1], legend=True, title='Momentum')
        '''
        smap[sym + ['SMA / P']].plot(ax=ax[0, 0], legend=True, title='SMA / P')
        bb[sym + ['BBI']].plot(ax=ax[0, 1], legend=True, title='BB Index')
        rsi[sym + ['RSI']].plot(ax=ax[1, 0], legend=True, title='RS Index')
        momentum[sym + ['Momentum']].plot(ax=ax[1, 1], legend=True,
                                          title='Momentum')

        f.autofmt_xdate()

        f, ax = plt.subplots()
        corr = key_data.corr()
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax)

        plt.show()

    return key_data


if __name__ == '__main__':
    run_analysis(sym='AAPL',
                 start_date=dt.datetime(2008, 1, 1),
                 end_date=dt.datetime(2009, 12, 31),
                 look_back=21,
                 normalise=True,
                 draw_charts=True)
