import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class Almgren_Chriss2000:
    """
    Description:
    Select few deviations from the seminal model/work of Almgren & Chriss in their 2000 paper
    "Optimal Execution of Portfolio Transactions"
    """
    def __init__(self, ALPHA = 1, ETA = 5e-6, GAMMA = 5e-5, BETA = 1, LAMBDA = 0.0000081, SIGMA = 0.2, EPSILON = 0.0625, N = 50, T = 0.5, X = 500):
        """
        :param ALPHA: power of temporary impact function
        :param ETA: linear coefficient of temporary impact function
        :param GAMMA: linear coefficient of permanent impact function
        :param BETA: power of permanent impact function
        :param LAMBDA: risk aversion measure**
        :param SIGMA: annual vol
        :param EPSILON: bid-ask spread + fees ~ fixed cost of selling ($/share)
        :param N: number of time steps/buckets (integer)
        :param T: expiry - when holdings must be depleted (days)
        :param X: number of shares holding initially (integer)
        """
        self.ALPHA = float(ALPHA)
        self.ETA = float(ETA)
        self.GAMMA = float(GAMMA)
        self.BETA = float(BETA)
        self.LAMBDA = float(LAMBDA)
        self.SIGMA = float(SIGMA)
        self.EPSILON = float(EPSILON)
        self.N = int(N)
        self.T = float(T)
        self.TAU = math.sqrt(self.T / self.N)
        self.KAPPA = math.sqrt((self.LAMBDA * (self.SIGMA ** 2)) /
                                (self.ETA * (1 + ((0.5 * self.GAMMA * self.TAU) / self.ETA))))
        self.X = int(X)
        #self.bellman_solve()
        self.basic_almgren()

    def temp_impact(self, x):
        return self.ETA * (x ** self.ALPHA)

    def perm_impact(self, x):
        return self.GAMMA * (x ** self.BETA)

    def hamiltonian(self, x, n):
        eq = self.LAMBDA * n * self.perm_impact(n / self.TAU) + \
             self.LAMBDA * (x - n) * self.TAU * self.temp_impact(n / self.TAU) +\
             (0.5 * (self.LAMBDA ** 2) * (self.SIGMA ** 2) * self.TAU * ((x - n) ** 2))
        return eq

    def variance_IS(self, opt_sale):
        variance_shortfall = 0
        step = -1
        while step < len(opt_sale) - 1:
            step += 1
            temp = (self.X - np.sum(opt_sale[0:step])) ** 2
            variance_shortfall += temp

        variance_shortfall = self.TAU * (self.SIGMA ** 2) * variance_shortfall
        return variance_shortfall

    def bellman_solve(self, plot='False'):
        """
        Using stochastic control to solve for best execution strategy
        :param plot: graph of strategy
        :return: 5-tup of value function, optimal trading set/strategy,
        inventory over time, as well as basic implentational shortfall expectation & variance for comparison purposes
        """
        value_func = np.zeros(shape=(self.N, self.X + 1), dtype='float64')
        opt_moves = np.zeros(shape=(self.N, self.X + 1), dtype='int')
        inventory = np.zeros(shape=(self.N, 1), dtype='int')
        inventory[0] = self.X
        opt_sale = []

        for x in range(self.X + 1):
            value_func[self.N - 1, x] = np.exp(x * self.temp_impact((x / self.TAU)))
            opt_moves[self.N - 1, x] = x

        for step in range(self.N - 2, -1, -1):
            for x in range(self.X + 1):
                best_value = value_func[step + 1, 0] * np.exp(self.hamiltonian(x, x))
                best_n = x

                for n in range(x):
                    current_value = value_func[step + 1, x - n] * np.exp(self.hamiltonian(x, n))
                    if current_value < best_value:
                        best_value = current_value
                        best_n = n

                value_func[step, x] = best_value
                opt_moves[step, x] = best_n

        for step in range(1, self.N):
            inventory[step] = inventory[step - 1] - opt_moves[step, inventory[step - 1]]
            opt_sale.append(opt_moves[step - 1])

        expected_shortfall = 0.5 * self.GAMMA * (self.X ** 2) + self.EPSILON * np.sum(opt_sale) +\
                             ((self.ETA - 0.5 * self.GAMMA) / self.TAU) * np.sum([sale ** 2 for sale in opt_sale])

        variance_shortfall = self.variance_IS(opt_sale)

        opt_sale = np.asarray(opt_sale)
        if plot=='True':
            plt.plot(inventory, color='blue', lw=1.6)
            plt.title('Optimal Execution - Dynamic Programming')
            plt.xlabel('Trading Step')
            plt.ylabel('Number of Shares')
            plt.grid(True)
            plt.show()

        return value_func, opt_moves, inventory, opt_sale, expected_shortfall, variance_shortfall

    def basic_almgren(self, plot='True'):
        """
        The baseline quadratic cost 2000 method for optimal execution from the paper - linear impact
        :return: opt_moves,
        """
        def optimal_objective(sale):
            """
            verbatim quadratic functuon of the control params
            :param sale: array
            :return: objective func
            """
            expected_cost = 0.5 * self.GAMMA * (self.X ** 2) + self.EPSILON * np.sum(sale) + ((self.ETA - 0.5 * self.GAMMA) / self.TAU) * np.sum([order ** 2 for order in sale])
            variance_cost = self.variance_IS(sale)
            objective_cost = expected_cost + self.LAMBDA * variance_cost
            return objective_cost

        trades = np.zeros((self.N, 1))
        BOUNDS = tuple((0.0, self.X) for x in range(len(trades)))
        CONSTRAINTS = ({'type': 'eq', 'fun': lambda x: np.sum(x) - self.X})
        opt_sale = minimize(optimal_objective, trades.flatten(), method='SLSQP', bounds=BOUNDS, constraints=CONSTRAINTS)
        opt_sale = np.array(opt_sale.x)
        inventory = [self.X]
        for t in range((len(opt_sale))):
            inventory.append(inventory[t] - opt_sale[t])

        if plot == 'True':
            plt.plot(inventory, color='blue', lw=1.6)
            plt.title('Optimal Execution - Quadratic Programming')
            plt.xlabel('Trading Step')
            plt.ylabel('Number of Shares')
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    execute = Almgren_Chriss2000()






