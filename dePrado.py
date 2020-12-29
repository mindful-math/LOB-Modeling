import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import scipy.io as sio
from pathlib import Path


class dePrado2014:
    """
    Description: This is a collection of programs/code relatd
    to the work of de Prado et al 2012 - "Optimal Execution Horizon"
    """
    def __init__(self, S_0 = 10, MU = 0.7, EPSILON = 0.3, ALPHA = 0.5, S_low = 5, S_high = 100, DELTA = 0.3, lob_path = 'Nov-2014/AMZN_20141103.mat'):
        """
        :param S_0: initial price of stock
        :param MU: rate of informed trades
        :param EPSILON: rate of uninformed trades
        :param ALPHA: probability event/news arrives
        :param S_low: price of event after bad news
        :param S_high: price of event after good news
        :param DELTA: probability of bad news
        :param lob_path: there are 3/4 sample mat files in the repo
        """
        self.S_0 = float(S_0)
        self.MU = float(MU)
        self.EPSILON = float(EPSILON)
        self.ALPHA = float(ALPHA)
        self.S_low = float(S_low)
        self.S_high = float(S_high)
        self.DELTA = float(DELTA)
        self.lob_raw = sio.loadmat(Path(lob_path).absolute())['LOB']
        #self.yfinance_data('SPY', '1wk', '1m')
        self.lob_data = self.get_lob_data(self.lob_raw)
        print(self.lob_data)

    def initial_spread(self):
        PIN = (self.ALPHA * self.MU) / (self.ALPHA * self.MU + 2 * self.EPSILON)
        return PIN * (self.S_high - self.S_low)

    def prob_buy_sell(self, X, Y, t):
        # returns probability of X buy & Y sell orders at time t
        prob_good_news = (self.ALPHA * (1 - self.DELTA) * math.exp(-(self.MU + 2 * self.EPSILON)) *\
               ((self.MU + self.EPSILON) ** X)(self.EPSILON ** Y)) /\
               (math.factorial(X) * math.factorial(Y))
        prob_bad_news = (self.ALPHA * self.DELTA * math.exp(-(self.MU + 2 * self.EPSILON)) *
                         ((self.MU + self.EPSILON) ** Y) * (self.EPSILON ** X)) /\
                        (math.factorial(X) * math.factorial(Y))
        prob_no_news = ((1 - self.ALPHA) * math.exp(-2 * self.EPSILON) * (self.EPSILON ** (X + Y))) /\
                       (math.factorial(X) * math.factorial(Y))
        prob = prob_good_news + prob_bad_news + prob_no_news
        return prob

    def get_yfinance_data(self, ticker, period, interval):
        """
        :param ticker: string i.e. 'SPY'
        :param period: string i.e. '1d',...,'10y'
        :param interval: string i.e. '1m',...,'3mo'
        :return: yfinance data as dataframe
        """
        return yf.Ticker(ticker).history(period=period, interval=interval)

    def get_lob_data(self, stock):
        """
        :param stock: path to matlab file of price info
        :return: time (t), change in time (dt), bid, bidvol, ask, askvol, market_orders, midprice, microprice, spread in a class
        """
        t = (np.array((stock['EventTime'][0][0][:, 0])) - 3600000 * 9.5) * 1e-3
        bid = np.array(stock['BuyPrice'][0][0] * 1e-4)
        bidvol = np.array(stock['BuyVolume'][0][0] * 1.0)
        ask = np.array(stock['SellPrice'][0][0] * 1e-4)
        askvol = np.array(stock['SellVolume'][0][0] * 1.0)
        market_order = np.array(stock['MO'][0][0] * 1.0)
        dt = t[1] - t[0]
        midprice = 0.5 * (bid[:, 0] + ask[:, 0])
        microprice = (bid[:, 0] * askvol[:, 0] + ask[:, 0] * bidvol[:, 0]) / (bidvol[:, 0] + askvol[:, 0])
        spread = ask[:, 0] - bid[:, 0]
        return {'t': t, 'bid': bid, 'bidvol': bidvol, 'ask': ask, 'askvol': askvol, 'market_order': market_order,
                'dt': dt, 'midprice': midprice, 'microprice': microprice, 'spread': spread}

    def bulk_volume_classification(self, data):
        price = data[0]
        bulk = [0, 0]
        while True:
            if data == None:
                break
            z = float(data[0] - price) / stDev
            z = stats.t.cdf(z, df)
            bulk[0] += min(data[1] * z, data[2])
            bulk[0] += min(data[1]  * (1 - z), data[1] - data[2])
            bulk[1] += data[1]
            price = data[0]



if __name__=="__main__":
    dePrado2014()