import numpy as np
from math import e
from numba import njit, prange

def asset_or_nothing_call(s, n, k, T, K, F, SIGMA, r):
    """
    :param s: initial price
    :param n: partition expiry into n branches (n>=0 for good estimate)
    :param z: size of partitions
    :param T: time to expiry (years)
    :param K: strike price
    :param F: payoff if S(t)>K, t <= T
    :param SIGMA: annual vol
    :param r: interest rate
    :return: price of the asset or nothing
    """
    u = e ** (SIGMA * (T / k))
    d = 1 / u
    BETA = e ** (- (r * T) / k)
    p = (1 + ((r * T) / k) - d) / (u - d)
    asset_option = np.zeros([n + 1, n + 1])
    for moves in prange(n + 1):
        for downfactors in prange(moves + 1):
            asset_option[downfactors, moves] = s * (u ** (moves - downfactors)) * (d ** downfactors)
    for moves in prange(n - 1, -1, -1):
        for downfactors in prange(0, moves + 1):
            if asset_option[downfactors, moves] - K <= 0:
                asset_option[downfactors, moves] = (BETA * ((p * asset_option[downfactors, moves + 1]) + ((1 - p) * asset_option[downfactors + 1, moves + 1])))
            else:
                asset_option[downfactors, moves] = F
    print(asset_option)
    return asset_option

