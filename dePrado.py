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
    def __init__(self, S_0 = 10, MU = 0.7, EPSILON = 0.3, ALPHA = 0.5, S_low = 5, S_high = 100, DELTA = 0.3, lob_path = 'Nov-2014/AMZN_20141103.mat', tick_data_path = 'SampleEquityData_US/Trades/14081.csv', plot = False, window = 5):
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
        self.lobster_raw = sio.loadmat(Path(lob_path).absolute())['LOB']
        self.lobster_data = self.get_lobster_data(self.lobster_raw)
        self.tick_data_raw = pd.read_csv(Path(tick_data_path).absolute())
        self.tick_data = self.clean_data()
        self.plot = plot
        if self.plot == True:
            self.data_vis()
        self.AR_order_imbalance(plot_regress=False)
        self.bulk_volume_classification(500)

    def get_yfinance_data(self, ticker, period, interval):
        """
        :param ticker: string i.e. 'SPY'
        :param period: string i.e. '1d',...,'10y'
        :param interval: string i.e. '1m',...,'3mo'
        :return: yfinance data as dataframe
        """
        return yf.Ticker(ticker).history(period=period, interval=interval)

    def get_lobster_data(self, stock):
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

    def get_TICKDATA(self, xlsx_path):
        pass

    def data_vis(self):
        # 11 is number of percentiles to color - (11-1)/2=5 is the bands shown
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

    def AR_order_imbalance(self, plot_regress):
        # separating buy & sell MO data
        MO_buy_vol = np.array((self.lobster_data['market_order'][:, 0] / 3.6e6,
                               self.lobster_data['market_order'][:, 6],
                               self.lobster_data['market_order'][:, 7])).T
        MO_buy_vol[:, 1] = np.where(MO_buy_vol[:, 2] < 0, 0, MO_buy_vol[:, 1])
        MO_buy_vol = MO_buy_vol[:, 0:2]
        MO_sell_vol = np.array((self.lobster_data['market_order'][:, 0] / 3.6e6,
                                self.lobster_data['market_order'][:, 6],
                                self.lobster_data['market_order'][:, 7])).T
        MO_sell_vol[:, 1] = np.where(MO_sell_vol[:, 2] > 0, 0,
                                     MO_sell_vol[:, 1])
        MO_sell_vol = MO_sell_vol[:, 0:2]
        plt.title('Market Order Buy Volumes Against Time')
        plt.xlabel('Time Since Midnight (Hours)')
        plt.ylabel('Volume')
        plt.scatter(MO_buy_vol[:, 0], MO_buy_vol[:, 1],
                    c=np.random.rand(len(MO_buy_vol)))
        plt.show()
        # Regressing Bid Volume - might change this to something more autocorrelated
        train = self.lobster_data['bidvol'][
                0:int(len(self.lobster_data['bidvol'][:, 0]) / 3), 0]
        test = self.lobster_data['bidvol'][
               int(len(self.lobster_data['bidvol'][:, 0]) / 3):len(
                   self.lobster_data['bidvol'][:, 0]) - 1, 0]
        model = AutoReg(train, lags=5).fit()
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

    def init_PIN(self):
        # probability of informed order conditional on sell
        prob_sell = (self.ALPHA * self.DELTA * self.MU) / (self.EPSILON + (self.ALPHA * self.DELTA * self.MU))
        # probability of informed order conditional on buy
        prob_buy = (self.ALPHA * (1 - self.DELTA) * self.MU) / (self.EPSILON + (self.ALPHA * (1 - self.DELTA) * self.MU))
        PIN = prob_sell + prob_buy
        return PIN

    def init_trade_range(self):
        # range MMs are willing to provide liquidity at time = 0
        return self.init_PIN() * (self.S_high - self.S_low)

    def PIN_estimate(self):
        # TODO - use MLE to estimate/update ALPHA, MU, DELTA, EPSILON over time
        pass

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

    def clean_data(self):
        df = self.tick_data_raw.drop(columns=['8:36:37', 'D', 'TB', '0', '1957', 'N', 'C', 'T', 'X', 'Unnamed: 12'])
        df.columns = ['Time', 'Price', 'Volume']
        return df

    def bulk_volume_classification(self, n):
        """
        :param n: n is the number of volume buckets you want
        :return: volume buckets of LOB data ready for VPIN/etc
        """
        total_volume = self.tick_data['Volume'].sum()
        volume_bucket_size = total_volume / n
        price_change = np.diff(self.tick_data['Price'], prepend=self.tick_data['Price'][0])
        price_deviation = np.std(price_change)
        buy_volume_buckets = []
        sell_volume_buckets = []
        total_volume = []
        v_i = []
        P_i = []
        for tick in self.tick_data.itertuples():
            if sum(v_i) + tick[3] < volume_bucket_size:
                v_i.append(tick[3])
                P_i.append(tick[2])
                continue

            else:
                if len(v_i) == 1:
                    buy_volume_buckets.append(v_i[0] / 2)
                    sell_volume_buckets.append(v_i[0] / 2)
                    total_volume.append(v_i)
                    v_i = []
                    P_i = []
                else:
                    price_change = abs(max(P_i[0:-1]) - min(P_i[0:-1]))
                    buy_volume = sum(v_i[0:-1]) * stats.norm.cdf(price_change / price_deviation)
                    sell_volume = sum(v_i[0:-1]) - buy_volume
                    total_volume.append(sum(v_i[0:-1]))
                    buy_volume_buckets.append(buy_volume)
                    sell_volume_buckets.append(sell_volume)
                    v_i = [v_i[-1]]
                    P_i = [P_i[-1]]

        return buy_volume_buckets, sell_volume_buckets

if __name__=="__main__":
    dePrado2014()
