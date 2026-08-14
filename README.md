# tr


# methodology

1. gather clean data
2. analyze exploratory statistics
3. develop testable tradeable hypothesis
4. out-of-sample and in sample (test, train) backtest
5. paper trade
6. live!


# notes

volatility - a function of price change over time

half life - how long it takes to decay halfway back if a price divated from the mean

u (mew/mean) - average spread over the lookback window the "fair" price between the two venues

sigma (std dev) - how much the spread typically deviates from mew

mew + 1 - normal distribution
mew + 2 - unusual statstically deviation

machine learning models that use a trading objective perform better than models that estimate a signla with a prediction objective


# statstical arbitrage

stat arb takes into account 2 pairs across different venues that may have a price discrepancy

returns are culled from going buying or selling when the prices deviate 2 std or 1std based on the stategy 

since the prices will always revert back to the mean, profiting from the spread

the elements composed of stat arb are three-fold:
1. identification of similar assets to generate arbitrage portfolios
2. the extraction of time-series signals for the temporary deviations of the similarity between assets
3. a trading policy in the arbitrage portfolios based on the time-series signals

some risk that are involved include:
- execution risk due to latency
- fee accummulation and slippage
- qoute inaccuracy
- 
however this infamous quantitative trading strategy has been known to profit

one guy on youtube shared his strategy for mulitple pairs including jpy225 which is the japense yen, an FX currency
across hyperliquid and meta5 (one DEX) and another CEX.

*action states*

spread  >=+ u  + 2std: sell/short (bet on reversal)

spread < u : exit cover short

spread <= u - 2std : buy/long the spread (bet it goes up) 

spread >= u

fade at 2sigma, exit at the mean symetrically
wherew 1 std bands are used as early warning zones or a tighter more agressive variant

a spread can open up between the two acssets because one is crupto wrapper for a real-world FX pair but the other asset is from a real-referenced price
