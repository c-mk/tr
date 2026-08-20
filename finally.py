"""
Two approaches to building a probability distribution for SPY's future price.

APPROACH 1: Historical Monte Carlo (backward-looking)
    Uses past returns to simulate many possible future paths.

APPROACH 2: Risk-Neutral Density from option prices (forward-looking)
    Uses the Breeden-Litzenberger result.

Requires: numpy, scipy, pandas, matplotlib, yfinance
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# ---------------------------------------------------------------------------
# APPROACH 1: Historical Monte Carlo (Geometric Brownian Motion)
# ---------------------------------------------------------------------------

def monte_carlo_distribution(S0, daily_returns, days_ahead, n_sims=50_000):
    mu = daily_returns.mean()
    sigma = daily_returns.std()

    rand_shocks = np.random.normal(mu, sigma, size=(n_sims, days_ahead))
    cumulative_log_return = rand_shocks.sum(axis=1)
    final_prices = S0 * np.exp(cumulative_log_return)

    return final_prices 

# ---------------------------------------------------------------------------
# APPROACH 2: Risk-neutral density from option prices (Breeden-Litzenberger)
# ---------------------------------------------------------------------------

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def risk_neutral_density(S0, strikes, implied_vols, T, r, oi=None, min_oi=50):
    strikes = np.asarray(strikes, dtype=float)
    implied_vols = np.asarray(implied_vols, dtype=float)

    if oi is not None:
        oi = np.asarray(oi)
        keep = oi >= min_oi
        strikes, implied_vols = strikes[keep], implied_vols[keep]

    order = np.argsort(strikes)
    strikes, implied_vols = strikes[order], implied_vols[order]

    # Fit a smooth curve through the implied vol smile
    iv_fit = np.poly1d(np.polyfit(strikes, implied_vols, deg=3))
    K_grid = np.linspace(strikes.min(), strikes.max(), 500)
    sigma_grid = iv_fit(K_grid)

    call_prices = black_scholes_call(S0, K_grid, T, r, sigma_grid)

    dK = K_grid[1] - K_grid[0]
    d2C_dK2 = np.gradient(np.gradient(call_prices, dK), dK)
    density = np.exp(r * T) * d2C_dK2

    density = np.clip(density, 0, None)
    trapz = getattr(np, "trapezoid", None) or np.trapz 
    density /= trapz(density, K_grid) 

    return K_grid, density

# ---------------------------------------------------------------------------
# Integration with yfinance
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ticker = "SPY"
    print(f"Fetching data for {ticker}...")
    spy = yf.Ticker(ticker)

    # 1. Fetch historical data for Monte Carlo
    hist = spy.history(period="1y")
    S0 = hist['Close'].iloc[-1]
    
    # Calculate daily log returns
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna().values

    # 2. Fetch options data
    expirations = spy.options
    today = datetime.today()
    
    # Find an expiration roughly 30 days out
    target_exp = expirations[0]
    for exp in expirations:
        delta_days = (datetime.strptime(exp, '%Y-%m-%d') - today).days
        if delta_days >= 30:
            target_exp = exp
            break
            
    print(f"Current Price: ${S0:.2f}")
    print(f"Selected Expiration: {target_exp}")

    T_calendar_days = (datetime.strptime(target_exp, '%Y-%m-%d') - today).days
    T_years = T_calendar_days / 365.0
    trading_days_ahead = int(T_calendar_days * (252 / 365))

    chain = spy.option_chain(target_exp)
    calls = chain.calls

    # Clean options data: filter out 0 IV and restrict to +/- 15% of spot to keep the polynomial fit stable
    calls = calls[(calls['impliedVolatility'] > 0) & 
                  (calls['strike'] > S0 * 0.85) & 
                  (calls['strike'] < S0 * 1.15)]

    strikes = calls['strike'].values
    implied_vols = calls['impliedVolatility'].values
    open_interest = calls['openInterest'].values

    # --- Run Approach 1 ---
    np.random.seed(0)
    mc_prices = monte_carlo_distribution(S0, log_returns, days_ahead=trading_days_ahead)

    # --- Run Approach 2 ---
    # Using 5% as a standard risk-free rate assumption (r=0.05)
    K_grid, rnd = risk_neutral_density(
        S0, strikes, implied_vols, T=T_years, r=0.047, oi=open_interest, min_oi=100
    )

    # --- Plot both for comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(mc_prices, bins=80, density=True, color="steelblue")
    axes[0].set_title(f"Historical MC ({trading_days_ahead} Trading Days)")
    axes[0].set_xlabel("SPY Price")

    axes[1].plot(K_grid, rnd, color="darkorange")
    axes[1].fill_between(K_grid, rnd, alpha=0.3, color="darkorange")
    axes[1].set_title(f"Options Implied Density (Exp: {target_exp})")
    axes[1].set_xlabel("SPY Price at Expiration")

    plt.tight_layout()
    plt.savefig("y.png", dpi=150)
    print("Saved comparison chart as 'y.png'.")
    # ---------------------------------------------------------------------------
# EXACT DISCRETE PROBABILITY DISTRIBUTION (Append to the end of the script)
# ---------------------------------------------------------------------------

# 1. Convert probability density to discrete probability weights
# by multiplying the density by the step size (the area of each bin).
dK = K_grid[1] - K_grid[0]
discrete_probs = rnd * dK

# 2. Normalize to ensure the weights sum to exactly 1.0 (100%)
# This cleans up any microscopic floating-point rounding errors.
discrete_probs /= discrete_probs.sum()

# 3. Create a DataFrame to map exact price points to their probabilities
prob_distribution = pd.DataFrame({
    "SPY_Price": K_grid,
    "Prob_Weight": discrete_probs,
    "Prob_Percent": discrete_probs * 100
})

print("\n--- Options-Implied Probability Distribution ---")
# Display a slice of the grid for terminal readability (e.g., every 25th price point)
print(prob_distribution.iloc[::25].round(4).to_string(index=False))

# 4. Example: Calculate the cumulative area (probability) of a specific event
target_price = S0 * 0.95  # Example: A 5% drop from the current spot price
prob_below_target = prob_distribution[prob_distribution["SPY_Price"] <= target_price]["Prob_Weight"].sum()
print(f"\nProbability of SPY closing at or below ${target_price:.2f}: {prob_below_target:.2%}")