import numpy as np
import pandas as pd
import math
import matplotlib as plt

class Almgren_Chriss2000:
    """
    Description:
    asdf
    asdf
    asdf

    """
    def __init__(self, ALPHA = 1, ETA = 0.05, GAMMA = 0.5, BETA = 1, LAMBDA = 0.001, SIGMA = 0.3, EPSILON = 0.0625, N = 500, T = 1, X = 10000):
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
        self.X = int(X)
        self.bellman_solve(plot='True')

    def temp_impact(self, x):
        return self.ETA * (x ** self.ALPHA)

    def perm_impact(self, x):
        return self.GAMMA * (x ** self.BETA)

    def hamiltonian(self, x, n):
        eq = self.LAMBDA * n * self.perm_impact(n / self.TAU) + self.LAMBDA * (x - n) * self.TAU * self.temp_impact(n / self.TAU) + (0.5 * (self.LAMBDA ** 2) * (self.SIGMA ** 2) * self.TAU * ((x - n) ** 2))
        return eq

    def bellman_solve(self, plot='True'):
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

        expected_shortfall = 0.5 * self.GAMMA * (self.X ** 2) + self.EPSILON * np.sum(opt_sale) + ((self.ETA - 0.5 * self.GAMMA) / self.TAU) * np.sum(opt_sale ** 2)
        variance_shortfall = 0
        step = -1
        while step < len(opt_sale) - 1:
            step += 1
            temp = (self.X - np.sum(N[0:step])) ** 2
            variance_shortfall += temp

        variance_shortfall = self.TAU * (self.SIGMA ** 2) * variance_shortfall

        opt_sale = np.asarray(opt_sale)
        if plot=='True':
            plt.figure(figsize=(7, 5))
            plt.plot(inventory, color='blue', lw=1.6)
            plt.xlabel('Trading Step')
            plt.ylabel('Number of Shares')
            plt.grid(True)
            plt.show()

        return value_func, opt_moves, inventory, opt_sale, expected_shortfall, variance_shortfall

if __name__ == "__main__":
    execute = Almgren_Chriss2000()







