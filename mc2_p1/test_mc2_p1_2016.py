import unittest
from marketsim import compute_portvals, compute_portfolio_stats
import datetime as dt
import numpy as np
import pandas as pd


# Dynamic test injection adapted from ideas found at:
# http://stackoverflow.com/questions/347109/how-do-i-concisely-implement-multiple-similar-unit-tests-in-the-python-unittest
# Originally developeb by Laura Hamilton 
# https://github.com/lauradhamilton/gatech_cs7646_machine_learning_for_trading_unit_tests

# We will be generating test methods from a data structure, so our TestCase
# class doesn't actually contain any test_* methods yet.
class PortfolioTestCase(unittest.TestCase):

    # Defines what we will actually do for each injected no-leverage test case.
    def check_no_leverage(self, name, filename, start_val, num_days,
                          last_day_portval, sharpe_ratio, avg_daily_ret):
        print name
        portvals = compute_portvals(orders_file=filename, start_val=start_val)
        portvals = portvals[portvals.columns[0]]
        days = len(portvals)
        cr, adr, sddr, sr = compute_portfolio_stats(portvals)

        np.testing.assert_allclose([num_days, last_day_portval,
                                    sharpe_ratio, avg_daily_ret],
                                   [days, portvals[-1], sr, adr],
                                   rtol=0.001)

    # Defines what we will actually do for each injected leverage test case.
    def check_leverage(self, name, filename, start_val, last_day_portval):
        print name
        portvals = compute_portvals(orders_file=filename, start_val=start_val)
        portvals = portvals[portvals.columns[0]]

        np.testing.assert_allclose([last_day_portval],
                                   [portvals[-1]],
                                   rtol=0.001)


# This is a function that generates and returns a method, suitable
# for inclusion in the PortfolioTestCase class, that calls check_analysis
# with the given parameters.
def generate_no_leverage_test(name, filename, start_val, num_days,
                                last_day_portval, sharpe_ratio, avg_daily_ret):
    def test_method(self):
        self.check_no_leverage(name, filename, start_val, num_days,
                               last_day_portval, sharpe_ratio, avg_daily_ret)
    return test_method


def generate_leverage_test(name, filename, start_val, last_day_portval):

    def test_method(self):
        self.check_leverage(name, filename, start_val, last_day_portval)
    return test_method


# This function loops through a data structure of desired test cases and
# dynamically injects test_* methods into the PortfolioTestCase class.
def add_test_cases():

    # Iterate through descriptions of normal test cases
    for name, file, start_val, num_days,\
        last_day_portval, sharpe_ratio, avg_daily_ret in [

            ('Wiki-orders-short.csv',
             './orders/orders-short.csv', 1000000,
             11, 998035.0, -0.446948390642, -0.000178539446839),

            ('Wiki-orders.csv',
             './orders/orders.csv', 1000000,
             240, 1133860.0, 1.21540888742, 0.000551651296638),

            ('Wiki-orders2.csv',
             './orders/orders2.csv', 1000000,
             232, 1078752.6, 0.788982285751, 0.000353426354584),

            ('Wiki-orders3.csv',
             './orders/orders3.csv', 1000000,
             141, 1050160.0, 1.03455887842, 0.000365289198877),

            ('Orders 1',
             './testcases2016/orders-01.csv', 1000000,
             245, 1115569.2, 0.612340613407, 0.00055037432146),

            ('Orders 2',
             './testcases2016/orders-02.csv', 1000000,
             245, 1095003.35, 1.01613520942, 0.000390534819609),

            ('Orders 3',
             './testcases2016/orders-03.csv', 1000000,
             240, 857616.0, -0.759896272199, -0.000571326189931),

            ('Orders 4',
             './testcases2016/orders-04.csv', 1000000,
             233, 923545.4, -0.266030146916, -0.000240200768212),

            ('Orders 5',
             './testcases2016/orders-05.csv', 1000000,
             296, 1415563.0, 2.19591520826, 0.00121733290744),

            ('Orders 6',
             './testcases2016/orders-06.csv', 1000000,
             210, 894604.3, -1.23463930987, -0.000511281541086),

            ('Orders 7 (modified)',
             './testcases2016/orders-07-modified.csv', 1000000,
             237, 1104930.8, 2.07335994413, 0.000428245010481),

            ('Orders 8 (modified)',
             './testcases2016/orders-08-modified.csv', 1000000,
             229, 1071325.1, 0.896734443277, 0.000318004442115),

            ('Orders 9 (modified)',
             './testcases2016/orders-09-modified.csv', 1000000,
             37, 1058990.0, 2.54864656282, 0.00164458341408),

            ('Orders 10 (modified)',
             './testcases2016/orders-10-modified.csv', 1000000,
             141, 1070819.0, 1.0145855303, 0.000521814978394)]:

        # For each test case, dynamically generate a method and inject it into
        # the PortfolioTestCase class, with a name starting with 'test' so it
        # will be picked up by unittest.main().
        setattr(PortfolioTestCase, 'test_' + name,
                generate_no_leverage_test(name, file, start_val, num_days,
                                            last_day_portval, sharpe_ratio,
                                            avg_daily_ret))

    # Iterate through descriptions of leverage test cases
    for name, file, start_val, last_day_portval in [

            ('Orders 11 Leverage SELL (modified)',
             './testcases2016/orders-11-modified.csv', 1000000,
             1053560.0),

            ('Orders 12 Leverage BUY (modified)',
             './testcases2016/orders-12-modified.csv', 1000000,
             1044437.0),

            ('Wiki leverage example #1',
             './orders/orders-leverage-1.csv', 1000000,
             1050160.0),

            ('Wiki leverage example #2',
             './orders/orders-leverage-2.csv', 1000000,
             1074650.0),

            ('Wiki leverage example #3',
             './orders/orders-leverage-3.csv', 1000000,
             1050160.0)]:

        # For each test case, dynamically generate a method and inject it into
        # the PortfolioTestCase class, with a name starting with 'test' so it
        # will be picked up by unittest.main().
        setattr(PortfolioTestCase, 'test_' + name,
                generate_leverage_test(name, file, start_val,
                                              last_day_portval))


# Inject our test cases into PortfolioTestCase
add_test_cases()

# Run Unittest only if we are the main module
if __name__ == '__main__':
    unittest.main()
