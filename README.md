This semester, the repo will serve to compile a collection of fundamental market making models, as well as explorations of my own. 

**glosten-milgrom-simple.py**
This is the simplest version - given some order book at each time, it computes the expected bid and ask. Here we can see the math & intuition blend nicely - spreads begin to converge after n trades as they gain more info on the "true" price
![ScreenShot](https://drive.google.com/uc?export=view&id=18s2g7-ETOgQ_dNAwFM2qGXg58gop8TzW)

**Kyle Model.py**
This features the single period and multiperiod versions of the discretized Kyle model. It computes params for determining agents order flow at each time period as well as the informed trader expected profit (MM prices fairly). The model is very interesting, but actually implementing the model requires guessing SIGMA_N (future vol at end of trading) and recursively solving for initial vol and the other params. The original paper did not provide a way to get the convergence of SIGMA- True_Sigma at t=0, so here's a I guess toy version as I'm not sure best way to converge
![ScreenShot](https://drive.google.com/uc?export=view&id=1BVKIPqujWb2vA3L-4r8hETevFOpv2omM)

Here is a graph of the order sizes of the participants with V_0=5, SIGMA_T = 0.2, MAX_ITER = 100, SIGMA_noise = 2. We see after several demonstrations that order size of noise and informed are relatively inversely correlated.
![ScreenShot](https://drive.google.com/uc?export=view&id=1Uriq0TB-LOCUhvgGEJJZJz8RLyUYmY5v)

**pairs_trading.py**
Notes: 
- more data to test can be found here - http://sebastian.statistics.utoronto.ca/books/algo-and-hf-trading/data/
  or here - https://lobsterdata.com/info/DataSamples.php
- Correlation returns a 2x2 - the stat you're after is the 12 or 21 entry
- Likewise, for the p-value from cointegration - the second value in the tuple
- I eyeball means and variances for the individual processes and split into three parts, but that can easily be modified to your liking


