import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
np.set_printoptions(threshold=sys.maxsize)

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
        self.lob_data = self.get_lob_data(self.lob_raw)
        self.data_vis()

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
        microprice = (bid[:, 0] * askvol[:, 0] + ask[:, 0] * bidvol[:, 0]) / (
                    bidvol[:, 0] + askvol[:, 0])
        spread = ask[:, 0] - bid[:, 0]
        order_imbalance = np.array((bidvol[:, 0] - askvol[:, 0]) /
                                   (bidvol[:, 0] + askvol[:, 0]), ndmin=2).T
        return {'t': t, 'bid': bid, 'bidvol': bidvol, 'ask': ask,
                'askvol': askvol, 'market_order': market_order,
                'dt': dt, 'midprice': midprice, 'microprice': microprice,
                'spread': spread, 'order_imbalance': order_imbalance}

    def data_vis(self):
        # 10 is hard-coded percentile count
        percentiles = np.linspace(0, 100, 11)
        colormap = cm.Blues
        spread = np.zeros((int(self.lob_data['t'][-1]), 11))
        for i in range(11):
            for time in range(int(self.lob_data['t'][-1])):
                spread[time, i] = np.percentile(self.lob_data['spread'], percentiles[i])

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].set_title('Mid - Micro')
        axs[0, 0].plot(self.lob_data['t'], self.lob_data['midprice'] - self.lob_data['microprice'], color='r')
        axs[1, 0].set_title('Order Imbalance')
        axs[1, 0].plot(self.lob_data['t'], self.lob_data['order_imbalance'])
        axs[0, 1].set_title('Spread with IQR')
        axs[0, 1].plot(np.arange(0, int(self.lob_data['t'][-1]), 1), spread[:, 5], color='k')
        for i in range(5):
            axs[0, 1].fill_between(np.arange(0, int(self.lob_data['t'][-1]), 1), spread[:, i], spread[:, -(i + 1)], color=colormap(i / 5))

        buy_orders = self.lob_data['market_order'][:, 7].clip(0,1)
        sell_orders = self.lob_data['market_order'][:, 7].clip(-1, 0)
        axs[1, 1].set_title('Cumulative Buy & Sell MOs')
        axs[1, 1].plot(np.arange(len(buy_orders)), np.cumsum(buy_orders), color='g')
        axs[1, 1].plot(np.arange(len(sell_orders)), np.cumsum(sell_orders))
        axs[1, 1].legend(loc="upper right")
        fig.tight_layout()
        plt.show()

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
