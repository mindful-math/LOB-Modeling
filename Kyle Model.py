import random
import math
import numpy as np

class KyleModel:

    def __init__(self, V_0 = 5, SIGMA_V = 1, SIGMA = 2):
        """
        one MM, one Informed, many Noise
        :param V_0: security value initially
        :param SIGMA_V: volatlity of security
        :param SIGMA: variance of net order flow of noise traders
        """
        self.V_0     = float(V_0)
        self.SIGMA_V = float(SIGMA_V)
        self.SIGMA   = float(SIGMA)
        self.one_period_price()


    def one_period_price(self):
        """
        one-time period model calculates expected price MM sets after
        seeing the LOB (informed order + noise orders), this is just a conditional
        expectation and simplifies nicely to the formula below for MM_price &
        informed trader profit after time period
        :return:
        """
        V_n = np.random.normal(self.V_0, self.SIGMA_V, 1)
        ALPHA = self.V_0 * (self.SIGMA / math.sqrt(self.SIGMA_V))
        BETA = self.SIGMA / math.sqrt(self.SIGMA_V)
        MU = self.V_0
        LAMBDA = math.sqrt(self.SIGMA_V) / (2 * self.SIGMA)
        informed_order = (BETA * V_n) + ALPHA
        net_order = informed_order + np.random.normal(0, self.SIGMA, 1)
        mm_price = ((math.sqrt(self.SIGMA_V) / (2 * self.SIGMA)) * net_order) + self.V_0
        informed_profit = (((V_n - self.V_0) ** 2) * self.SIGMA) / (2 * math.sqrt(self.SIGMA_V))
        print(f'Market Maker Price: {mm_price}')
        print(f'Informed Trader Expected Profit: {informed_profit}')
        return {'MM Price': mm_price, 'Informed Profit': informed_profit}


    def multiperiod_price(self):
        """

        :return:
        """


if __name__ == "__main__":
    test = KyleModel()

        