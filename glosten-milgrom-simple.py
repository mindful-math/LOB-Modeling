import os
import csv
import string
import random
import numpy as np
from scipy import optimize as optim
from scipy.signal import savgol_filter
from scipy.stats import norm, truncnorm
import matplotlib.pyplot as plt


class Glosten_And_Milgrom_Simplest:


    def __init__(self, N = 50, ALPHA = 0.5, BETA = 0.3, V_low = 0, V_high = 10):
        """
        :param N: number of trades observed in time t (only buy/sell)
        :param ALPHA: probability of low vs high price (1-alpha - high price)
        :param BETA: proportion of informed traders
        :param V_low: the low price
        :param V_high: the high price
        """
        self.N        = N
        self.ALPHA    = float(ALPHA)
        self.BETA      = float(BETA)
        self.V_low    = float(V_low)
        self.V_high = float(V_high)

        self.ask  = []
        self.bid  = []
        self.lob   = [1 if random.random() < 0.5 else -1 for n in range(0, self.N+1)]
        self.bidask_price()
        self.plot()


    def bidask_price(self):
        """
        just using bayes to update and improve estimate on price after time t
        The numerator is prior probability of V (high or low) * conditional probability
        which is prob of # of buy + sell orders given V (high or low) * event (v_low or v_high)
        denominator is the constrained sample space to prob of x buy and y sell orders
        :return: n-array of bid & ask expected values
        """
        for n in range(0, self.N):
            buy_orders = sum([x if x > 0 else 0 for x in self.lob[0:n]])
            sell_orders = n - buy_orders
            bid_numerator = (self.V_low * (((1 - self.BETA) ** buy_orders) * ((1 + self.BETA) ** (sell_orders + 1))) +
                            self.V_high * ((1 - self.ALPHA) * ((1 + self.BETA) ** buy_orders) * ((1 - self.BETA) ** (sell_orders + 1))))
            bid_denominator = ((1 - self.ALPHA) * ((1 + self.BETA) ** buy_orders) * ((1 - self.BETA) ** (sell_orders + 1)) +
                           self.ALPHA * ((1 - self.BETA) ** buy_orders) * ((1 + self.BETA) ** (sell_orders + 1)))
            ask_numerator = (self.V_low * (((1 - self.BETA) ** (buy_orders + 1)) * ((1 + self.BETA) ** sell_orders)) +
                            self.V_high * ((1 - self.ALPHA) * ((1 + self.BETA) ** (buy_orders + 1)) * ((1 - self.BETA) ** sell_orders)))
            ask_denominator = ((1 - self.ALPHA) * ((1 + self.BETA) ** (buy_orders + 1)) * ((1 - self.BETA) ** sell_orders) +
                           self.ALPHA * ((1 - self.BETA) ** (buy_orders + 1)) * ((1 + self.BETA) ** sell_orders))
            self.bid.append(bid_numerator / bid_denominator)
            self.ask.append(ask_numerator / ask_denominator)

    def plot(self):
        # prices vs trades/time
        plt.plot(np.arange(self.N), self.bid, label = 'bid price')
        plt.plot(np.arange(self.N), self.ask, label = 'ask price')
        plt.xlabel('trades/time')
        plt.ylabel('Price')
        plt.title('Simplified Glosten-Milgrom')
        plt.legend()
        plt.show()

if __name__ == "__main__":

    test = Glosten_And_Milgrom_Simplest()
