import numpy as np
import pandas as pd
import math
from statsmodels.tsa.stattools import coint
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt

def init_asset(stock):
    '''
    :param stock: matlab file of price info
    :return: time (t), change in time (dt), bid, bidvol, ask, askvol, market_orders, midprice, microprice, spread in a class
    '''
    t = (np.array((stock['EventTime'][0][0][:, 0])) - 3600000 * 9.5) * 1e-3
    bid = np.array(stock['BuyPrice'][0][0] * 1e-4)
    bidvol = np.array(stock['BuyVolume'][0][0] * 1.0)
    ask = np.array(stock['SellPrice'][0][0] * 1e-4)
    askvol = np.array(stock['SellVolume'][0][0] * 1.0)
    market_order = np.array(stock['MO'][0][0] * 1.0)
    dt = t[1]-t[0]
    midprice = 0.5 * (bid[:, 0] + ask[:, 0])
    microprice = (bid[:, 0] * askvol[:, 0] + ask[:, 0] * bidvol[:, 0]) / (bidvol[:, 0] + askvol[:, 0])
    spread = ask[:, 0] - bid[:, 0]
    return {'time': t, 'bid': bid, 'bidvol': bidvol, 'ask': ask, 'askvol': askvol, 'market_order': market_order, 'dt': dt, 'midprice': midprice, 'microprice': microprice, 'spread': spread}


def pairs_stats(s1, s2):
    '''
    :param s1: asset 1 as a numpy array
    :param s2: asset 2 as a numpy array
    :return: correlation, stationarity (just summary stats), cointegration (p-value)
    '''
    corr = np.corrcoef(s1, s2)
    cointegration = coint(s1, s2)
    #
    # just print summary stats for deciding stationarity
    #
    s1_0, s1_1, s1_2 = np.split(s1, 3)
    s2_0, s2_1, s2_2 = np.split(s2, 3)
    s1_0_mean, s1_1_mean, s1_2_mean = s1_0.mean(), s1_1.mean(), s1_2.mean()
    s2_0_mean, s2_1_mean, s2_2_mean = s2_0.mean(), s2_1.mean(), s2_2.mean()
    s1_0_var, s1_1_var, s1_2_var = s1_0.var(), s1_1.var(), s1_2.var()
    s2_0_var, s2_1_var, s2_2_var = s2_0.var(), s2_1.var(), s2_2.var()

    print("Correlation: " + str(corr) + "\n")
    print("s1 Split Means: " + str(s1_0_mean) + ', ' + str(s1_1_mean) + ', ' + str(s1_2_mean) + "\n")
    print("s1 Split Vars: " + str(s1_0_var) + ', ' + str(s1_1_var) + ', ' + str(s1_2_var) + "\n")
    print("s2 Split Means: " + str(s2_0_mean) + ', ' + str(s2_1_mean) + ', ' + str(s2_2_mean) + "\n")
    print("s2 Split Vars: " + str(s2_0_var) + ', ' + str(s2_1_var) + ', ' + str(s2_2_var) + "\n")
    print("p Value for s1-x*s2: " + str(cointegration))
    
    #
    # check spread
    #
    dif = s1 - s2
    dif.plot()
    plt.axhline(dif.mean(), color = 'black')
    

# example (amzn isn't stationary, two are cointegrated if choose p=.015, spread ~ normal)
amzn = init_asset(sio.loadmat(Path('Nov-2014/AMZN_20141103.mat').absolute())['LOB'])
ebay = init_asset(sio.loadmat(Path('Nov-2014/EBAY_20141103.mat').absolute())['LOB'])
pairs_stats(amzn['midprice'], ebay['midprice'])

