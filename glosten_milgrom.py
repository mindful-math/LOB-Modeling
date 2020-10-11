import os
import csv
import string
import random
import numpy as np
from scipy import optimize as optim
import matplotlib.pyplot as plt


class Glosten_And_Milgrom_1985:


    def __init__(self, N = 50, BETA = 0.50, KAPPA = 1.0, ALPHA = 0.10, SIGMA = 0.10, MAX_ITER = 1000, TOL = 0.01):
        """
        :param N: number of trades observed in time t
        :param BETA:
        :param KAPPA:
        :param ALPHA:
        :param SIGMA:
        :param MAX_ITER:
        :param TOL:
        """
        self.N        = N
        self.BETA     = float(BETA)
        self.KAPPA    = float(KAPPA)
        self.ALPHA    = float(ALPHA)
        self.SIGMA    = float(SIGMA)
        self.MAX_ITER = MAX_ITER
        self.TOL      = float(TOL)

        self.p    = np.arange(self.N+1) / float(self.N)
        self.ask  = np.zeros(self.N+1)
        self.bid  = np.zeros(self.N+1)
        self.mu   = np.zeros((self.N+1, 2))
        self.tr   = np.zeros((self.N+1, 2))
        self.d    = np.zeros((self.N+1, 2))
        self.w    = np.zeros((self.N+1, 2))
        self.dwdp = np.zeros((self.N+1, 2))

        for n in range(0, self.N+1):
            self.w[n, 0] = (np.exp(2 * (1 - self.p[n])) - 1.0) / 10.0 ## High type value function
            self.w[n, 1] = (np.exp(2 * self.p[n]) - 1.0) / 10.0       ## Low type value function
        self.w[0, 0] = (np.exp(2) - 1.0) / 10.0
        self.w[self.N, 1] = (np.exp(2) - 1.0) / 10.0

        i = 0
        self.err = []
        self.has_converged = False
        while ((self.has_converged == False) and (i < self.MAX_ITER)):
            self.num_deriv()
            self.compute_bid()
            self.compute_ask()
            self.compute_drift()
            self.check_for_bluffing()
            self.update_w()
            self.compute_err()
            i = i + 1
            print([i, np.std(self.err[0:9])/np.mean(self.err[0:9])])
            self.plot_interim()

        self.plot_final()


    def smooth_values(self, x):

        K = len(x)
        for k in range(3,K-3):
            x[k] = (x[k+3] + x[k+2] + x[k+1] + x[k] + x[k-1] + x[k-2] + x[k-3])/7
        for k in range(K-3,3):
            x[k] = (x[k+3] + x[k+2] + x[k+1] + x[k] + x[k-1] + x[k-2] + x[k-3])/7
        x[1]   = x[1] + 0.20 * ((x[0] + x[1])/2 - x[1])
        x[2]   = x[2] + 0.10 * ((x[0] + x[1] + x[2])/3 - x[2])
        x[3]   = x[3] + 0.05 * ((x[0] + x[1] + x[2] + x[3])/4 - x[3])
        x[K-2] = x[K-2] + 0.20 * ((x[K-1] + x[K-2])/2 - x[K-2])
        x[K-3] = x[K-3] + 0.10 * ((x[K-1] + x[K-2] + x[K-3])/3 - x[K-3])
        x[K-4] = x[K-4] + 0.05 * ((x[K-1] + x[K-2] + x[K-3] + x[K-4])/4 - x[K-4])
        for k in range(2,K-2):
            x[k] = (x[k+2] + x[k+1] + x[k] + x[k-1] + x[k-2])/5
        for k in range(K-2,2):
            x[k] = (x[k+2] + x[k+1] + x[k] + x[k-1] + x[k-2])/5
        x[1]   = x[1] + 0.20 * ((x[0] + x[1])/2 - x[1])
        x[2]   = x[2] + 0.10 * ((x[0] + x[1] + x[2])/3 - x[2])
        x[3]   = x[3] + 0.05 * ((x[0] + x[1] + x[2] + x[3])/4 - x[3])
        x[K-2] = x[K-2] + 0.20 * ((x[K-1] + x[K-2])/2 - x[K-2])
        x[K-3] = x[K-3] + 0.10 * ((x[K-1] + x[K-2] + x[K-3])/3 - x[K-3])
        x[K-4] = x[K-4] + 0.05 * ((x[K-1] + x[K-2] + x[K-3] + x[K-4])/4 - x[K-4])
        for k in range(1,K-1):
            x[k] = (x[k+1] + x[k] + x[k-1])/3
        for k in range(K-1,1):
            x[k] = (x[k+1] + x[k] + x[k-1])/3
        return x


    def num_deriv(self):

        for n in range(1, self.N):
            self.dwdp[n,0] = 0.5 * ( (self.w[n+1,0] - self.w[n,0])/(self.p[n+1] - self.p[n]) + (self.w[n,0] - self.w[n-1,0])/(self.p[n] - self.p[n-1]) )
            self.dwdp[n,1] = 0.5 * ( (self.w[n+1,1] - self.w[n,1])/(self.p[n+1] - self.p[n]) + (self.w[n,1] - self.w[n-1,1])/(self.p[n] - self.p[n-1]) )
        self.dwdp[0,0]      = (self.w[1,0] - self.w[0,0])/(self.p[1] - self.p[0])
        self.dwdp[self.N,0] = 0
        self.dwdp[0,1]      = 0
        self.dwdp[self.N,1] = (self.w[self.N,1] - self.w[self.N-1,1])/(self.p[self.N] - self.p[self.N-1])


    def interpolate(self, x):

        nearest_p_found = False
        n = 0
        while ((nearest_p_found == False) and (self.N > n)):
            if (x <= self.p[n+1]):
                nearest_p_found = True
            else:
                n = n + 1
        if (n == self.N):
            w_at_x = self.w[self.N,:]
        else:
            w_at_x = [self.w[n,0] * ((x - self.p[n]) / (self.p[n+1] - self.p[n])) + self.w[n + 1,0] * ((self.p[n+1] - x) / (self.p[n+1] - self.p[n])),
                      self.w[n,1] * ((x - self.p[n]) / (self.p[n+1] - self.p[n])) + self.w[n + 1,1] * ((self.p[n+1] - x) / (self.p[n+1] - self.p[n]))]
        return w_at_x


    def compute_bid(self):

        tol              = 0.0001
        self.bid[0]      = 0
        self.bid[self.N] = 1
        for n in range(1, self.N):
            err_sq = 1
            i = 0
            b = self.p[n]/2
            while ((err_sq > tol) and (i < 10000)):
                err    = self.w[n,1] - (b - self.interpolate(b)[1])
                b      = b + err / 1000
                err_sq = err * err
                i      = i + 1
            if ((i < 10000) and (b > 0)):
                self.bid[n] = b
            else:
                self.bid[n] = 0.00001
        self.bid = self.smooth_values(self.bid)


    def compute_ask(self):

        tol              = 0.0001
        self.ask[0]      = 0
        self.ask[self.N] = 1
        for n in range(1, self.N):
            err_sq = 1
            i = 0
            a = (1 - self.p[n])/2
            while ((err_sq > tol) and (i < 10000)):
                err    = self.w[n,0] - ((1 - a) - self.interpolate(a)[0])
                a      = a - err / 1000
                err_sq = err * err
                i      = i + 1
            if ((i < 10000) and (a < 1)):
                self.ask[n] = a
            else:
                self.ask[n] = 1 - 0.00001
        self.ask = self.smooth_values(self.ask)


    def compute_drift(self):

        for n in range(0, self.N + 1):
            if ((self.ask[n] != 1) and (self.bid[n] != 0)):
                m = (self.BETA * self.p[n] * (self.p[n] - self.bid[n])) / self.bid[n] - (self.BETA * (1 - self.p[n]) * (self.ask[n] - self.p[n])) / (1 - self.ask[n])
                self.mu[n,0] = m
                self.mu[n,1] = m
            else:
                self.mu[n,0] = (self.bid[n] == 0) * (0.10) + (self.ask[n] == 1) * (-0.10)
                self.mu[n,1] = (self.bid[n] == 0) * (0.10) + (self.ask[n] == 1) * (-0.10)
        self.mu[:,0] = self.smooth_values(self.mu[:,0])
        self.mu[:,1] = self.smooth_values(self.mu[:,1])


    def check_for_bluffing(self):

        for n in range(0, self.N + 1):
            self.tr[n,0] = self.w[n,0] - ((self.bid[n] - 1) + self.interpolate(self.bid[n])[0])
            self.tr[n,1] = self.w[n,1] - (- self.ask[n] + self.interpolate(self.ask[n])[1])
        self.tr[:,0] = self.smooth_values(self.tr[:,0])
        self.tr[:,1] = self.smooth_values(self.tr[:,1])

        for n in range(0, self.N + 1):
            if (self.tr[n,0] < 0):
                self.mu[n,0] = self.mu[n,0] - (np.abs(self.tr[n,0]) + 1)/10.0
            if (self.tr[n,1] < 0):
                self.mu[n,1] = self.mu[n,1] + (np.abs(self.tr[n,1]) + 1)/10.0


    def update_w(self):

        for n in range(0, self.N+1):
            self.d[n,0] = self.dwdp[n,0] * self.mu[n,0] + self.BETA * (self.interpolate(self.ask[n])[0] + self.interpolate(self.bid[n])[0]  - 2 * self.w[n,0]) - self.KAPPA * self.w[n,0]
            self.d[n,1] = self.dwdp[n,1] * self.mu[n,1] + self.BETA * (self.interpolate(self.ask[n])[1] + self.interpolate(self.bid[n])[1]  - 2 * self.w[n,1]) - self.KAPPA * self.w[n,1]
        self.d[:,0] = self.smooth_values(self.d[:,0])
        self.d[:,1] = self.smooth_values(self.d[:,1])

        for n in range(1, self.N):
            self.w[n,0] = self.w[n,0] + self.SIGMA * (self.d[n,0])
            self.w[n,1] = self.w[n,1] + self.SIGMA * (self.d[n,1])
        self.w[0,0]      = (np.exp(2) - 1.0)/10.0
        self.w[self.N,1] = (np.exp(2) - 1.0)/10.0
        self.w[:,0] = self.smooth_values(self.w[:,0])
        self.w[:,1] = self.smooth_values(self.w[:,1])


    def compute_err(self):

        err = 0.0
        for n in range(1, self.N):
            err = err + np.power(self.d[n,0], 2)
            err = err + np.power(self.d[n,1], 2)

        err = err / (2 * (self.N+1))
        self.err = [err] + self.err

        if (len(self.err) >= 10):
            std_err  = np.std(self.err[0:9])
            mean_err = np.mean(self.err[0:9])
            if (std_err/mean_err < self.TOL):
                self.has_converged = True


    def plot_interim(self):
        plt.ion()
        plt.figure(1, figsize=(12,12))
        plt.draw()
        plt.show()

        plt.subplot(321)
        plt.plot(self.p, self.w[:,0], 'r')
        plt.plot(self.p, self.w[:,1], 'b')
        plt.axis([0,1,0,1])
        plt.ylabel('wH(p) + wL(p)')
        plt.xlabel('p')
        plt.grid(True)

        plt.subplot(322)
        plt.plot(self.p, self.ask, 'r')
        plt.plot(self.p, self.bid, 'b')
        plt.plot(self.p, self.p, 'k--')
        plt.axis([0,1,-0.1,1.1])
        plt.ylabel('a(p) + b(p)')
        plt.xlabel('p')
        plt.grid(True)

        plt.subplot(323)
        plt.plot(self.p, self.d[:,0], 'r')
        plt.plot(self.p, self.d[:,1], 'b')
        plt.axis([0,1,-2,2])
        plt.ylabel('dH(p)+ dL(p)')
        plt.xlabel('p')
        plt.grid(True)

        plt.subplot(324)
        plt.plot(self.p, self.dwdp[:,0], 'r')
        plt.plot(self.p, self.dwdp[:,1], 'b')
        plt.axis([0,1,-5,5])
        plt.ylabel("wH'(p) + wL'(p)")
        plt.xlabel('p')
        plt.grid(True)

        plt.subplot(325)
        plt.plot(self.p, self.mu[:,0], 'r')
        plt.plot(self.p, self.mu[:,1], 'b')
        plt.plot(self.p, np.zeros(self.N+1), 'k--')
        plt.axis([0,1,-1,1])
        plt.ylabel('muH(p) + muL(p)')
        plt.xlabel('p')
        plt.grid(True)

        plt.subplot(326)
        plt.plot(self.p, self.tr[:,0], 'r')
        plt.plot(self.p, self.tr[:,1], 'b')
        plt.axis([0,1,-0.25,1])
        plt.ylabel('trH(p) + trL(p)')
        plt.xlabel('p')
        plt.grid(True)

        plt.draw()
        plt.show()
        plt.savefig('plt_interim.png')



    def plot_final(self):

        plt.figure(2, figsize=(12,12))

        plt.subplot(311)
        plt.plot(self.p, self.w[:,0], 'r', linewidth = 3)
        plt.plot(self.p, self.w[:,1], 'b', linewidth = 3)
        plt.axis([0,1,0,1])
        plt.ylabel('wH(p) + wL(p)')
        plt.xlabel('p')
        plt.grid(True)

        plt.subplot(312)
        plt.plot(self.p, self.ask, 'r', linewidth = 3)
        plt.plot(self.p, self.bid, 'b', linewidth = 3)
        plt.plot(self.p, self.p, 'k--', linewidth = 2)
        plt.axis([0,1,-0.1,1.1])
        plt.ylabel('a(p) + b(p)')
        plt.xlabel('p')
        plt.grid(True)

        plt.subplot(313)
        plt.plot(self.p, self.mu[:,0], 'r', linewidth = 3)
        plt.plot(self.p, self.mu[:,1], 'b', linewidth = 3)
        plt.plot(self.p, np.zeros(self.N+1), 'k--', linewidth = 2)
        plt.axis([0,1,-1,1])
        plt.ylabel('muH(p) + muL(p)')
        plt.xlabel('p')
        plt.grid(True)

        plt.show()
        plt.savefig('plt_final.png')


if __name__ == "__main__":

    test = Glosten_And_Milgrom_1985()