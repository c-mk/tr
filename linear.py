import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Fetch historical data for Bitcoin and Gold (GLD)
tickers = ["BTC-USD", "GLD"]
data = yf.download(tickers, start="2022-01-01", end="2024-01-01")['Close']

# 2. Clean the data 
# Crypto trades 24/7, Gold trades on weekdays. Drop rows where Gold data is missing (weekends/holidays).
data = data.dropna()

# 3. Define dependent (y) and independent (x) variables
X = data[['GLD']].values      # Independent variable: Gold
y = data['BTC-USD'].values    # Dependent variable: Bitcoin

# 4. Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 5. Plot the results
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', alpha=0.4, label='Actual Daily Prices')
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression Line')

plt.title('Linear Regression: Gold (GLD) vs Bitcoin (BTC-USD)')
plt.xlabel('Gold (GLD) Price in USD')
plt.ylabel('Bitcoin (BTC) Price in USD')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Print the relationship metrics
print(f"Coefficient (Slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")