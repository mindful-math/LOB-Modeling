import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class Criscuolo2014:
    """
    Description: Criscuolo & Waelbroek published a paper in 2014 in the Journal
    of Trading on the effect of stochastic volatility on optimal execution.
    The stockastic volatility model is a deviant of the Heston that is time
    dependent. There's a messy total cost model they constructed which they attempt
    to minimize using simulated annealing, but I will try using sci-py as their
    method only worked fast/well for small N.
    """
    def __init__(self, KAPPA = 3, THETA = 0.01, GAMMA = 0.1, RHO = 0.0, V_0 = 0.5, r = 0.05, T = 0.5, N = 4, S_0 = 100, ALPHA_INFINITY = 0.4, MU_1 = 0.4, MU_2 = 0.8, ALPHA = 1.5, BETA = 0.3, XI = 365):
        """

        :param KAPPA: exponential decay constant for volatility to mean-revert to THETA
        :param THETA: long-run average volatility
        :param GAMMA: variance of volatility
        :param RHO: correlation between stock and volatility
        :param V_0: initial volatility
        :param r: risk-free rate
        :param T: time till expiry
        :param N: discretized time periods for optimization
        :param S_0: initial stock price
        :param ALPHA_INFINITY: magnnitude of the alpha term (in bps)
        :param MU_1: speed at which alpha reaches max value
        :param MU_2: length till alpha approaches 0
        :param ALPHA: parameter for impact from share turnover
        :param BETA: parameter for impact from trader schedule/execution
        :param XI: another param for impact....
        """
        self.KAPPA = float(KAPPA)
        self.THETA = float(THETA)
        self.GAMMA = float(GAMMA)
        self.RHO = float(RHO)
        self.V_0 = float(V_0)
        self.r = float(r)
        self.T = float(T)
        self.N = int(N)
        self.S_0 = float(S_0)
        self.ALPHA_INFINITY = float(ALPHA_INFINITY)
        self.MU_1 = float(MU_1)
        self.MU_2 = float(MU_2)
        self.ALPHA = float(ALPHA)
        self.BETA = float(BETA)
        self.XI = float(XI)
        self.VOL_RATIO = self.V_0 / math.sqrt(self.THETA)
        self.optimal_execution()

    def optimal_execution(self):
        """
        With stochastic vol & cost model in the paper, compute the trading
        trajectory (\pi_i) that minimizes the cost
        :return: N-array of trajectories & share turnover
        """
        def total_cost(trades):
            """
            This gets really ugly - I'm sorry, but it's how the paper is written. You can
            read the paper Criscuolo & Waelbroeck (2014) or visit my website to see what's going
            on.
            :param trades[i][0]: Volume traded in an interval divided by average daily volume - <x_i>
            :param trades[i][1]: array dictacting participation rate for institution
            :return: Total Cost = Alpha Cost + Impact Cost
            """
            trades = np.reshape(trades, (4, 2))
            share_turnover = [trade[0] for trade in trades]
            participation = [trade[1] for trade in trades]
            turnover_dif = np.diff(share_turnover, prepend=share_turnover[0])
            inst_time = np.cumsum(turnover_dif / participation)
            inst_time_diff = np.diff(inst_time, prepend=inst_time[0])
            alpha_cost = self.ALPHA_INFINITY * (1 / trades[self.N - 1][0])\
                         * np.sum([(trades[k][1] * self.MU_2)
                                   * (math.exp(-1 * (inst_time[k - 1]
                                                     / self.MU_2)) -
                                      math.exp(-1 * (inst_time[k] / self.MU_2)))
                                   + ((self.MU_1 / (self.MU_1 + self.MU_2))
                                      * (math.exp(-1 * inst_time[k] *
                                                  ((1 / self.MU_1) +
                                                   (1 / self.MU_2)))
                                         - math.exp(-1 * inst_time[k - 1]
                                                    * ((1 / self.MU_1)
                                                       + (1 / self.MU_2)))))
                                   for k in range(0, len(share_turnover) - 1)])
            #
            # Compute Stochastic Volatility - deviant of Heston that is time dependent
            #
            F_func = [math.exp(-0.5 * (self.KAPPA * inst_time[k]))
                      * math.sqrt((self.VOL_RATIO ** 2) - 1 +
                                  math.exp(self.KAPPA * inst_time[k]))
                      for k in range(0, len(share_turnover) - 1)]

            F_func_diff = np.diff(F_func, prepend=F_func[0])
            stochastic_vol = [math.sqrt(self.THETA) + ((3 * self.GAMMA) /
                                                       (16 * self.KAPPA *
                                                        math.sqrt(self.THETA)))
                              + (((2 * math.sqrt(self.THETA)) /
                                  (self.KAPPA * inst_time_diff[k]))
                                 * (math.log((1 + F_func[k]) /
                                             (1 + F_func[k - 1])) - F_func_diff[k]))
                              + ((2 * math.sqrt(self.THETA) * (self.GAMMA ** 2))
                                 / (16 * (self.KAPPA ** 2) * F_func_diff[k] * self.THETA))
                              * ((3 * math.log((1 + F_func[k]) / (1 + F_func[k - 1]))
                                  + ((F_func_diff[k] * 2 * (self.VOL_RATIO ** 2) - 3)
                                     / (((self.VOL_RATIO ** 2) - 1) ** 2))
                                  - (((self.VOL_RATIO ** 4) * F_func_diff[k])
                                     / (F_func[k] * F_func[k - 1]
                                        * (((self.VOL_RATIO ** 2) - 1) ** 2)))))
                              for k in range(0, len(share_turnover) - 1)]

            impact_cost = (self.XI / (self.ALPHA - 1)) *\
                          np.sum([stochastic_vol[k] *
                                  (trades[k][1] ** self.BETA)
                                  * ((trades[k][0] ** (self.ALPHA - 1))
                                     - (trades[k - 1][0] ** (self.ALPHA - 1)))
                                  for k in range(0, len(share_turnover) - 1)])\
                          + (((self.GAMMA * self.RHO) / (2 * trades[self.N - 1][0]))
                             * (self.KAPPA * trades[self.N - 1][0]
                                - np.sum(share_turnover[0: self.N - 1])))

            total_cost = impact_cost + alpha_cost
            return total_cost

        optimal_trades = np.zeros((self.N, 2))
        opt_sale = minimize(total_cost, optimal_trades.flatten(), method='SLSQP')
        print(opt_sale)
        return opt_sale

if __name__ == "__main__":
    Criscuolo2014()
