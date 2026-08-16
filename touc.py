"""
SPY: Price Chart + Linear Regression + Exploratory Stats
===========================================================
Run this LOCALLY (needs internet access to Yahoo Finance).

Setup (one time):
    pip install yfinance pandas numpy matplotlib scipy statsmodels

Run:
    python spy_analysis.py

What this does:
    1. Pulls historical daily SPY price data
    2. Plots the price chart
    3. Fits a linear regression (price vs. time) and overlays the trend line
    4. Runs exploratory statistics: daily returns distribution, volatility,
       skew/kurtosis, rolling volatility, autocorrelation, drawdown
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TICKER = "SPY"
START_DATE = "2015-01-01"
END_DATE = None  # None = today


# ---------------------------------------------------------------------------
# 1. FETCH DATA
# ---------------------------------------------------------------------------
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# 2. PRICE CHART + LINEAR REGRESSION
# ---------------------------------------------------------------------------
def plot_price_with_regression(df):
    df = df.copy()
    df["t"] = np.arange(len(df))  # time index as integer for regression

    slope, intercept, r_value, p_value, std_err = stats.linregress(df["t"], df["Close"])
    df["trend"] = intercept + slope * df["t"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label=f"{TICKER} Close", linewidth=1.2)
    ax.plot(df.index, df["trend"], label="Linear Regression Trend", linewidth=1.8,
            linestyle="--", color="firebrick")
    ax.set_title(f"{TICKER} Price with Linear Regression Trend")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("spy_price_regression.png", dpi=150)
    print("Saved chart: spy_price_regression.png")
    plt.show()

    print("\n--- Linear Regression: Price vs. Time ---")
    print(f"Slope (avg $ change/day): {slope:.4f}")
    print(f"R-squared:                {r_value**2:.4f}")
    print(f"P-value:                  {p_value:.2e}")
    print(f"Std error of slope:       {std_err:.4f}")

    return df


# ---------------------------------------------------------------------------
# 3. EXPLORATORY STATISTICS
# ---------------------------------------------------------------------------
def exploratory_stats(df):
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    returns = df["daily_return"].dropna()

    print("\n--- Exploratory Statistics: Daily Returns ---")
    print(f"Mean daily return:       {returns.mean():.4%}")
    print(f"Std dev (volatility):    {returns.std():.4%}")
    print(f"Annualized volatility:   {returns.std() * np.sqrt(252):.2%}")
    print(f"Skewness:                {stats.skew(returns):.4f}")
    print(f"Kurtosis (excess):       {stats.kurtosis(returns):.4f}")
    print(f"Min daily return:        {returns.min():.2%}")
    print(f"Max daily return:        {returns.max():.2%}")
    print(f"Sharpe (naive, rf=0):    {(returns.mean() / returns.std()) * np.sqrt(252):.2f}")

    # autocorrelation of returns (lag-1) — tests if yesterday's move predicts today's
    autocorr_1 = returns.autocorr(lag=1)
    print(f"Lag-1 autocorrelation:   {autocorr_1:.4f}")

    # max drawdown
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    print(f"Max drawdown:            {drawdown.min():.2%}")

    # histogram of returns
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(returns, bins=80, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_title(f"{TICKER} Daily Return Distribution")
    axes[0].set_xlabel("Daily Return")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3)

    # rolling 30-day volatility
    rolling_vol = returns.rolling(30).std() * np.sqrt(252)
    axes[1].plot(rolling_vol.index, rolling_vol, color="darkorange", linewidth=1.2)
    axes[1].set_title(f"{TICKER} Rolling 30-Day Annualized Volatility")
    axes[1].set_ylabel("Annualized Vol")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("spy_exploratory_stats.png", dpi=150)
    print("\nSaved chart: spy_exploratory_stats.png")
    plt.show()

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Fetching {TICKER} data from {START_DATE}...")
    raw = fetch_data(TICKER, START_DATE, END_DATE)

    reg_df = plot_price_with_regression(raw)
    exploratory_stats(raw)