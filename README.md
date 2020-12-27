This semester/year, the repo will serve to compile a collection of fundamental market making models, as well as explorations of my own. I apologize for the docs and structure of this - I just wanted to make functional & readable model code available for people to use - most of the original papers do not include code and this helps for reproducability/verifying results. 

**glosten-milgrom-simple.py**
This is the simplest version - given some order book at each time, it computes the expected bid and ask. Here we can see the math & intuition blend nicely - spreads begin to converge after n trades as they gain more info on the "true" price
![ScreenShot](https://drive.google.com/uc?export=view&id=18s2g7-ETOgQ_dNAwFM2qGXg58gop8TzW)

**Kyle Model.py**
This features the single period and multiperiod versions of the discretized Kyle model. It computes params for determining agents order flow at each time period as well as the informed trader expected profit (MM prices fairly). The model is very interesting, but actually implementing the model requires guessing SIGMA_N (future vol at end of trading) and recursively solving for initial vol and the other params. The original paper did not provide a way to get the convergence of SIGMA- True_Sigma at t=0, so here's a I guess toy version as I'm not sure best way to converge
![ScreenShot](https://drive.google.com/uc?export=view&id=1BVKIPqujWb2vA3L-4r8hETevFOpv2omM)

Here is a graph of the order sizes of the participants with V_0=5, SIGMA_T = 0.2, MAX_ITER = 100, SIGMA_noise = 2. We see after several demonstrations that order size of noise and informed are relatively inversely correlated.
![ScreenShot](https://drive.google.com/uc?export=view&id=1Uriq0TB-LOCUhvgGEJJZJz8RLyUYmY5v)

**optimal_execution.py**
This features a collection of models deviating from the seminal work of Almgren & Chriss in 2000. For now, it has optimal execution from the very basic version with linear impact costs as well as the one with stochastic optimal control & dynamic programming. Here is a graph of optimal execution from the stochastic approach with the following parameters: ALPHA = 1, ETA = 0.05, GAMMA = 0.5, BETA = 1, LAMBDA = 0.0003, SIGMA = 0.3, EPSILON = 0.0625, N = 50, T = 1, X = 100:
![ScreenShot](https://drive.google.com/uc?export=view&id=1bO2KBGDsW7c738PQ4fOyMp7GFvv6fYqy)

**Stochastic-Vol-Optimal-Exec.py**
This attempts to replicate the results of Criscuolo & Waehlbroek (2014) when uncovering the effects of stochastic volatility (a deviant of the Heston with temporal averaging) on participation rate schedules for institutions. The code is buggy/doesn't work at the moment - the minimization looks atrocious - the paper simplifies results for only n=4 - not enough/realistic as trades can happen much more frequently. Additionally they use some "patented/private" simulated annealing software that supposedly makes this code work - I don't want to point fingers, but the result isn't practical even if it is solvable using some fancy numerical technique because of runtime issues. 

**pairs_trading.py**
Notes: 
- more data to test can be found here - http://sebastian.statistics.utoronto.ca/books/algo-and-hf-trading/data/
  or here - https://lobsterdata.com/info/DataSamples.php
- Correlation returns a 2x2 - the stat you're after is the 12 or 21 entry
- Likewise, for the p-value from cointegration - the second value in the tuple
- I eyeball means and variances for the individual processes and split into three parts, but that can easily be modified to your liking


