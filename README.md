This semester, the repo will serve to compile a collection of fundamental market making models, as well as explorations of my own. 

**glosten-milgrom-simple.py**
This is the simplest version - given some order book at each time, it computes the expected bid and ask. Here we can see the math & intuition blend nicely - spreads begin to converge after n trades as they gain more info on the "true" price
![ScreenShot](https://drive.google.com/uc?export=view&id=18s2g7-ETOgQ_dNAwFM2qGXg58gop8TzW)
**pairs_trading.py**
Notes: 
- more data to test can be found here - http://sebastian.statistics.utoronto.ca/books/algo-and-hf-trading/data/
  or here - https://lobsterdata.com/info/DataSamples.php
- Correlation returns a 2x2 - the stat you're after is the 12 or 21 entry
- Likewise, for the p-value from cointegration - the second value in the tuple
- I eyeball means and variances for the individual processes and split into three parts, but that can easily be modified to your liking


