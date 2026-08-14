import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# 1. Fetch and clean data
tickers = ["BTC-USD", "GLD"]
data = yf.download(tickers, start="2022-01-01", end="2024-01-01")['Close'].dropna()

X = data['GLD']
y = data['BTC-USD']

# statsmodels requires you to explicitly add a constant for the y-intercept
X_with_const = sm.add_constant(X)

# 2. Step 1 of Engle-Granger: Run OLS (Ordinary Least Squares) Regression
model = sm.OLS(y, X_with_const).fit()

# Print the statistical summary (This contains your slope and P-VALUE)
print(model.summary())

# 3. Step 2 of Engle-Granger: Extract the residuals (errors)
residuals = model.resid

# 4. Test for Stationarity using the Augmented Dickey-Fuller (ADF) test
print("\n--- Augmented Dickey-Fuller Test ---")
adf_result = adfuller(residuals)
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"P-Value: {adf_result[1]:.4f}")

if adf_result[1] < 0.05:
    print("Conclusion: Residuals are stationary (mean-reverting). The assets ARE cointegrated.")
else:
    print("Conclusion: Residuals are NOT stationary. The assets are NOT cointegrated.")
'''
# 5. Plot the residuals to visually check for mean-reversion
plt.figure(figsize=(10, 5))
plt.plot(residuals, color='purple', linewidth=1.5, label='Residuals (Spread)')
plt.axhline(0, color='red', linestyle='--', label='Mean (0)')

plt.title('Regression Residuals: Testing for Mean Reversion')
plt.xlabel('Date')
plt.ylabel('Residual Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 2. Calculate cumulative returns (normalizing to start at 0)
cumulative_returns = (data / data.iloc[0]) - 1
'''

# 2. Calculate cumulative returns (normalizing to start at 0)
cumulative_returns = (data / data.iloc[0]) - 1


# 3. Plot the comparison
plt.figure(figsize=(10, 5))
plt.plot(cumulative_returns['BTC-USD'], color='orange', label='Bitcoin (BTC)')
plt.plot(cumulative_returns['GLD'], color='gold', label='Gold (GLD)')

plt.title('Cumulative Returns: Bitcoin vs. Gold')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (0.5 = +50%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()