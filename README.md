This semester, the repo will serve to compile a collection of fundamental market making models, as well as explorations of my own. 

**glosten_milgrom.py** 
Notes:
- Credit for the original model code goes to Alex Chinco of UIUC
- I will be refactoring and testing modifications to the model like the results of this paper 
https://web.stanford.edu/~milgrom/publishedarticles/Bid%20Ask%20and%20Transaction%20Prices.pdf


**pairs_trading.py**
Notes: 
- more data to test can be found here - http://sebastian.statistics.utoronto.ca/books/algo-and-hf-trading/data/
  or here - https://lobsterdata.com/info/DataSamples.php
- Correlation returns a 2x2 - the stat you're after is the 12 or 21 entry
- Likewise, for the p-value from cointegration - the second value in the tuple
- I eyeball means and variances for the individual processes and split into three parts, but that can easily be modified to your liking
