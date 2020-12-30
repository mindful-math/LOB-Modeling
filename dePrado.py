import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import scipy.io as sio
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg, AutoRegResults
from random import random
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
    def __init__(self, S_0 = 10, MU = 0.7, EPSILON = 0.3, ALPHA = 0.5, S_low = 5, S_high = 100, DELTA = 0.3, lob_path = 'Nov-2014/AMZN_20141103.mat', plot = False, window = 5):
        """
        :param S_0: initial price of stock
        :param MU: rate of informed trades
        :param EPSILON: rate of uninformed trades
        :param ALPHA: probability event/news arrives
        :param S_low: price of event after bad news
        :param S_high: price of event after good news
        :param DELTA: probability of bad news
        :param lob_path: there are 3/4 sample mat files in the repo
        :param plot: true -> plot microstructure stuff
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
        self.plot = plot
        if self.plot == True:
            self.data_vis()
        self.AR_order_imbalance(plot_regress=False)


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
        t = (np.array((stock['EventTime'][0][0][:, 0])) - 34200000) * 1e-3
        minute_incremented_t = t[::600]
        hour_incremented_t = t[::36000]
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

        buy_orders = self.lob_data['market_order'][:, 7].clip(-1,0)
        sell_orders = self.lob_data['market_order'][:, 7].clip(0, 1)
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

    def AR_order_imbalance(self, plot_regress):
        # autoregresses market order buy volume
        MO_buy_vol = np.array((self.lob_data['market_order'][:, 0] / 3.6e6, self.lob_data['market_order'][:, 6], self.lob_data['market_order'][:, 7])).T
        MO_buy_vol[:, 1] = np.where(MO_buy_vol[:, 2] < 0, 0, MO_buy_vol[:, 1])
        MO_buy_vol = MO_buy_vol[:, 0:2]
        MO_sell_vol = np.array((self.lob_data['market_order'][:, 0] / 3.6e6, self.lob_data['market_order'][:, 6], self.lob_data['market_order'][:, 7])).T
        MO_sell_vol[:, 1] = np.where(MO_sell_vol[:, 2] > 0, 0, MO_sell_vol[:, 1])
        MO_sell_vol = MO_sell_vol[:, 0:2]
        plt.title('Market Order Buy Volumes Against Time')
        plt.xlabel('Time Since Midnight (Hours)')
        plt.ylabel('Volume')
        plt.scatter(MO_buy_vol[:, 0], MO_buy_vol[:, 1], c=np.random.rand(len(MO_buy_vol)))
        plt.show()
        # Regressing Bid Volume - might change this to something more autocorrelated
        train = self.lob_data['bidvol'][0:int(len(self.lob_data['bidvol'][:, 0]) / 3), 0]
        test = self.lob_data['bidvol'][int(len(self.lob_data['bidvol'][:, 0]) / 3):len(self.lob_data['bidvol'][:, 0]) - 1, 0]
        model = AutoReg(train, lags = 5).fit()
        coef = model.params

        def predict(params, history):
            Y = params[0]
            for i in range(1, len(params)):
                Y += params[i] * history[-i]
            return Y

        history = [train[i] for i in range(len(train))]
        pred = []
        for t in range(len(test)):
            Y = predict(coef, history)
            observ = test[t]
            pred.append(observ)
            history.append(observ)

        rmse = math.sqrt(mean_squared_error(test, pred))
        print(f'RMSE: {rmse}')
        if plot_regress == True:
            plt.plot(test)
            plt.plot(pred, color='blue')
            plt.show()
        
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
