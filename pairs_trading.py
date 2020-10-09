import numpy as np
import pandas as pd
import math
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import coint
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt

def init_asset(stock):
    """
    :param stock: matlab file of price info
    :return: time (t), change in time (dt), bid, bidvol, ask, askvol, market_orders, midprice, microprice, spread in a class
    """
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
    return {'t': t, 'bid': bid, 'bidvol': bidvol, 'ask': ask, 'askvol': askvol, 'market_order': market_order, 'dt': dt, 'midprice': midprice, 'microprice': microprice, 'spread': spread}


def pairs_stats(s1, s2):
    """
    :param s1: asset 1 as a numpy array
    :param s2: asset 2 as a numpy array
    :return: correlation, stationarity (just summary stats), cointegration (p-value)
    """
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
    diff = s1['midprice'] - s2['midprice']
    plt.plot(s1['t'], diff)
    plt.title('S1-S2 Spread')
    plt.ylabel('Midprice Diff')
    plt.axhline(diff.mean(), color='black')
    plt.show()
    

# example (amzn isn't stationary, two are cointegrated if choose p=.015, spread ~ normal)
amzn = init_asset(sio.loadmat(Path('Nov-2014/AMZN_20141103.mat').absolute())['LOB'])
ebay = init_asset(sio.loadmat(Path('Nov-2014/EBAY_20141103.mat').absolute())['LOB'])
pairs_stats(amzn['midprice'], ebay['midprice'])


#
# Intraday Market Activity 
#
def volume_pred(v_i, v_n, spy, vix, r, of):
    """
    :param v_i: lagged volume of shares
    :param v_n: present volume want to predict
    :param spy: SPY ETF intraday returns
    :param vix: VIX intraday 'returns'
    :param r: asset's intraday returns
    :param of: day's net order flow
    :return: regression for log(1+v_n) and basic stats for it
    
    Intuition: basic factors that may affect volume (market+endogenous), but typically, this leads to a poor prediction, adding HL-volatility demonstrates some interesting correlation with volume
    """
    df = pd.DataFrame({'v_i': np.log(v_i + 1), 'spy': spy, 'vix': vix, 'r': r, 'of': of, 'v_n': np.log(v_n + 1)})
    model = ols("v_n = v_i + spy + vix + r + of", df).fit()
    print(model.summary())
    return model._results.params


#
# Volatility Measures
#
def vol(s, dv, t):
    """
    :param s: stock returns over some time t
    :param dv: window size, realized dv-min volatility (i.e. 60 - realized vol/hr)
    :param t: time period you want to look at - in minutes
    :return: dictionary of volatility measures over t: realized vol, HL-vol, Bid/ask change, and ...
    """
    dt = (1440 - t) * 600
    mu = (2 / np.pi) ** 2
    s = s[0:s.size - dt]
    r_vol = pd.DataFrame(s)[::10].rolling(dv).std(ddof=0)
    bp_vol = pd.DataFrame(s)[::10].rolling(dv).apply(lambda x: mu * (x.abs() * x.shift(1).abs()).sum())
    hl_vol = max(s) - min(s)
    # need to add bid/ask count version
    return {'realized': r_vol.dropna(how='all'), 'bipower': bp_vol.dropna(how='all'), 'hl_vol': hl_vol}
