"""
SPY: Volatility-Adjusted Trend Strategy + Rolling VWAP Bands + Volume-at-Extension Analysis
===============================================================================================
Run this LOCALLY (needs internet access to Yahoo Finance).

Setup (one time):
    pip install yfinance pandas numpy matplotlib scipy

Run:
    python spy_vwap_vol_strategy.py

WHAT THIS BUILDS ON:
    From our earlier exploratory stats we found:
      - Kurtosis ~14 (fat tails)          -> size positions off volatility, not a fixed %
      - Skew ~ -0.3 (sharper down moves)  -> asymmetric risk, tighter on downside
      - Weak autocorrelation (-0.12)      -> momentum/volume filter is a supporting signal only

WHAT'S IN THIS SCRIPT:
    1. Rolling volatility (used to size positions AND set stops)
    2. Rolling VWAP with +/-1 and +/-2 standard deviation bands
       (NOTE: true VWAP is an INTRADAY measure using minute-level volume, resetting daily.
       Free daily data can't replicate that. This is a "rolling VWAP" — a multi-day
       volume-weighted average, which is the standard adaptation when you only have
       daily bars. It answers a similar question ("is price extended vs. where volume
       has concentrated") but is NOT the same tool an intraday trader means by VWAP.)
    3. A volume-confirmed moving-average trend signal
    4. Volatility-adjusted position sizing + stop-loss
    5. THE ANALYSIS YOU ASKED FOR: does average volume tend to be higher when price is
       at/beyond +1std, +2std, -1std, -2std from the rolling VWAP? Broken out and printed
       + charted so you can see it directly.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TICKER = "SPY"
START_DATE = "2015-01-01"
END_DATE = None

VWAP_WINDOW = 20          # rolling window (days) for VWAP + bands
VOL_WINDOW = 20           # rolling window (days) for volatility used in sizing/stops
MA_SHORT = 20
MA_LONG = 100

STARTING_CAPITAL = 10_000
RISK_PER_TRADE = 0.01     # 1% of capital risked per trade (from our kurtosis discussion)
STOP_ATR_MULTIPLE = 2.0   # stop = entry -/+ (this multiple * rolling volatility in $ terms)


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# 2. ROLLING VWAP + STD BANDS
# ---------------------------------------------------------------------------
def compute_rolling_vwap(df, window=VWAP_WINDOW):
    df = df.copy()
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical_price * df["Volume"]

    df["vwap"] = pv.rolling(window).sum() / df["Volume"].rolling(window).sum()

    # std of price around the VWAP, over the same rolling window
    price_dev = typical_price - df["vwap"]
    df["vwap_std"] = price_dev.rolling(window).std()

    df["vwap_plus_1std"] = df["vwap"] + df["vwap_std"]
    df["vwap_plus_2std"] = df["vwap"] + 2 * df["vwap_std"]
    df["vwap_minus_1std"] = df["vwap"] - df["vwap_std"]
    df["vwap_minus_2std"] = df["vwap"] - 2 * df["vwap_std"]

    df["typical_price"] = typical_price
    df["price_z"] = price_dev / df["vwap_std"]  # how many std devs price is from VWAP today
    return df


# ---------------------------------------------------------------------------
# 3. VOLUME-AT-EXTENSION ANALYSIS  <-- the specific question you asked
# ---------------------------------------------------------------------------
def analyze_volume_at_extension(df):
    df = df.copy()
    avg_volume = df["Volume"].mean()

    bins = {
        "Beyond -2std": df["price_z"] < -2,
        "-2std to -1std": (df["price_z"] >= -2) & (df["price_z"] < -1),
        "-1std to 0 (below VWAP)": (df["price_z"] >= -1) & (df["price_z"] < 0),
        "0 to +1std (above VWAP)": (df["price_z"] >= 0) & (df["price_z"] < 1),
        "+1std to +2std": (df["price_z"] >= 1) & (df["price_z"] < 2),
        "Beyond +2std": df["price_z"] >= 2,
    }

    print("\n--- Volume at Price Extension from Rolling VWAP ---")
    print(f"Overall average volume: {avg_volume:,.0f}\n")
    print(f"{'Zone':<28}{'Days':>7}{'Avg Volume':>16}{'Vol vs Overall':>17}")
    results = {}
    for label, mask in bins.items():
        n_days = mask.sum()
        if n_days == 0:
            continue
        zone_avg_vol = df.loc[mask, "Volume"].mean()
        pct_diff = (zone_avg_vol / avg_volume - 1) * 100
        results[label] = {"n_days": n_days, "avg_volume": zone_avg_vol, "pct_vs_overall": pct_diff}
        print(f"{label:<28}{n_days:>7}{zone_avg_vol:>16,.0f}{pct_diff:>15.1f}%")

    return results


def plot_volume_at_extension(results, df):
    labels = list(results.keys())
    pct_diffs = [results[l]["pct_vs_overall"] for l in labels]
    colors = ["firebrick" if v < 0 else "steelblue" for v in pct_diffs]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(labels, pct_diffs, color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Volume vs. Overall Average (%)")
    ax.set_title(f"{TICKER}: Does Volume Expand When Price is Extended from Rolling VWAP?")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("spy_volume_at_extension.png", dpi=150)
    print("\nSaved chart: spy_volume_at_extension.png")
    plt.show()


# ---------------------------------------------------------------------------
# 3b. LEAD/LAG: does volume rise BEFORE a drop into extreme negative territory,
#     or only AT THE SAME TIME as the drop? This distinguishes a genuinely
#     predictive signal from a purely descriptive/coincident one.
# ---------------------------------------------------------------------------
def analyze_volume_lead_lag(df, extension_threshold=-2.0, max_lag=5):
    df = df.copy()
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(63).mean()

    # baseline: what volume_ratio looks like on a random/typical day
    baseline = df["volume_ratio"].mean()

    # find the FIRST day of each distinct move beyond the threshold
    # (avoid counting the same multi-day crash event over and over)
    is_extreme = df["price_z"] < extension_threshold
    event_start = is_extreme & (~is_extreme.shift(1).fillna(False))
    event_dates = df.index[event_start]

    print(f"\n--- Volume Lead/Lag Around Moves Beyond {extension_threshold}std ---")
    print(f"Number of distinct events: {len(event_dates)}")
    print(f"Baseline (typical day) volume ratio: {baseline:.3f}\n")

    lag_results = {}
    for lag in range(max_lag, -1, -1):  # from 5 days BEFORE down to the event day itself
        vals = []
        for d in event_dates:
            loc = df.index.get_loc(d)
            target_loc = loc - lag
            if target_loc >= 0:
                vals.append(df["volume_ratio"].iloc[target_loc])
        if vals:
            avg_ratio = np.mean(vals)
            lag_results[lag] = avg_ratio
            tag = "(event day)" if lag == 0 else f"({lag} day(s) before)"
            print(f"Lag -{lag:<2} {tag:<20} avg volume ratio: {avg_ratio:.3f}  "
                  f"({(avg_ratio/baseline - 1)*100:+.1f}% vs typical day)")

    return lag_results, baseline


def plot_volume_lead_lag(lag_results, baseline, threshold):
    lags = sorted(lag_results.keys(), reverse=True)
    vals = [lag_results[l] for l in lags]
    x_labels = [f"-{l}d" if l > 0 else "Event day" for l in lags]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x_labels, vals, marker="o", linewidth=2, color="firebrick")
    ax.axhline(baseline, color="gray", linestyle="--", label="Typical day (baseline)")
    ax.set_title(f"Volume Ratio Before/During Moves Beyond {threshold}std from VWAP")
    ax.set_ylabel("Volume vs. 63-day average (ratio)")
    ax.set_xlabel("Days relative to the extreme move")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("spy_volume_lead_lag.png", dpi=150)
    print("\nSaved chart: spy_volume_lead_lag.png")
    plt.show()


# ---------------------------------------------------------------------------
# 4. VOLUME-CONFIRMED TREND SIGNAL
# ---------------------------------------------------------------------------
def generate_signals(df, ma_short=MA_SHORT, ma_long=MA_LONG):
    df = df.copy()
    df["ma_short"] = df["Close"].rolling(ma_short).mean()
    df["ma_long"] = df["Close"].rolling(ma_long).mean()
    df["avg_volume_63d"] = df["Volume"].rolling(63).mean()  # ~ 1 quarter

    trend_up = df["ma_short"] > df["ma_long"]
    volume_confirmed = df["Volume"] > df["avg_volume_63d"]  # above-average conviction

    df["signal"] = np.where(trend_up & volume_confirmed, 1, 0)
    return df


# ---------------------------------------------------------------------------
# 5. ROLLING VOLATILITY FOR SIZING + STOPS
# ---------------------------------------------------------------------------
def compute_volatility(df, window=VOL_WINDOW):
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    df["rolling_vol_pct"] = df["daily_return"].rolling(window).std()          # as a % of price
    df["rolling_vol_dollars"] = df["rolling_vol_pct"] * df["Close"]           # in $ terms
    return df


# ---------------------------------------------------------------------------
# 6. BACKTEST WITH VOLATILITY-ADJUSTED SIZE + STOP
# ---------------------------------------------------------------------------
def run_backtest(df, starting_capital=STARTING_CAPITAL, risk_per_trade=RISK_PER_TRADE,
                  stop_atr_mult=STOP_ATR_MULTIPLE):
    df = df.copy()
    df["position"] = df["signal"].shift(1).fillna(0)

    equity = starting_capital
    equity_curve = []
    trade_log = []

    in_trade = False
    entry_price = None
    stop_price = None
    shares = 0

    for i in range(len(df)):
        row = df.iloc[i]
        price = row["Close"]
        vol_dollars = row["rolling_vol_dollars"]

        if not in_trade and row["position"] == 1 and not np.isnan(vol_dollars) and vol_dollars > 0:
            # volatility-adjusted stop distance
            stop_distance = stop_atr_mult * vol_dollars
            # risk-based position size: risk_per_trade % of equity / stop distance per share
            dollar_risk = equity * risk_per_trade
            shares = dollar_risk / stop_distance if stop_distance > 0 else 0
            entry_price = price
            stop_price = price - stop_distance
            in_trade = True

        elif in_trade:
            # check stop
            if price <= stop_price or row["position"] == 0:
                pnl = (price - entry_price) * shares
                equity += pnl
                trade_log.append({
                    "entry_date": df.index[i], "exit_price": price,
                    "entry_price": entry_price, "pnl": pnl,
                    "pnl_pct": (price / entry_price - 1)
                })
                in_trade = False
                shares = 0

        # mark-to-market for equity curve
        if in_trade:
            unrealized = (price - entry_price) * shares
            equity_curve.append(equity + unrealized)
        else:
            equity_curve.append(equity)

    df["equity_curve"] = equity_curve
    df["buy_hold_curve"] = starting_capital * (df["Close"] / df["Close"].iloc[0])
    trades_df = pd.DataFrame(trade_log)
    return df, trades_df


# ---------------------------------------------------------------------------
# 7. STATS + PLOTS
# ---------------------------------------------------------------------------
def report_and_plot(df, trades_df):
    print("\n--- Strategy Performance (Volatility-Adjusted Size + Stop) ---")
    if len(trades_df) == 0:
        print("No trades were generated in this period.")
    else:
        win_rate = (trades_df["pnl"] > 0).mean()
        avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl_pct"].mean()
        avg_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl_pct"].mean()
        total_return = df["equity_curve"].iloc[-1] / STARTING_CAPITAL - 1
        bh_return = df["buy_hold_curve"].iloc[-1] / STARTING_CAPITAL - 1

        running_max = pd.Series(df["equity_curve"]).cummax()
        drawdown = (pd.Series(df["equity_curve"]) - running_max) / running_max

        print(f"Number of trades:   {len(trades_df)}")
        print(f"Win rate:           {win_rate:.1%}")
        print(f"Avg win:            {avg_win:.2%}" if not np.isnan(avg_win) else "Avg win: n/a")
        print(f"Avg loss:           {avg_loss:.2%}" if not np.isnan(avg_loss) else "Avg loss: n/a")
        print(f"Strategy return:    {total_return:.2%}")
        print(f"Buy & hold return:  {bh_return:.2%}")
        print(f"Max drawdown:       {drawdown.min():.2%}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["equity_curve"], label="Vol-Adjusted Strategy", linewidth=1.6)
    ax.plot(df.index, df["buy_hold_curve"], label="Buy & Hold", linewidth=1.2, alpha=0.7)
    ax.set_title(f"{TICKER}: Volatility-Adjusted Strategy vs. Buy & Hold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("spy_vol_strategy_equity.png", dpi=150)
    print("\nSaved chart: spy_vol_strategy_equity.png")
    plt.show()


def plot_price_vwap_bands(df, lookback_days=500):
    plot_df = df.iloc[-lookback_days:]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(plot_df.index, plot_df["Close"], label="Close", linewidth=1.2, color="black")
    ax.plot(plot_df.index, plot_df["vwap"], label="Rolling VWAP", linewidth=1.4, color="blue")
    ax.fill_between(plot_df.index, plot_df["vwap_minus_1std"], plot_df["vwap_plus_1std"],
                     alpha=0.15, color="blue", label="+/-1 std")
    ax.fill_between(plot_df.index, plot_df["vwap_minus_2std"], plot_df["vwap_plus_2std"],
                     alpha=0.08, color="blue", label="+/-2 std")
    ax.set_title(f"{TICKER}: Price vs. Rolling VWAP Bands (last {lookback_days} trading days)")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("spy_vwap_bands.png", dpi=150)
    print("Saved chart: spy_vwap_bands.png")
    plt.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Fetching {TICKER} data from {START_DATE}...")
    raw = fetch_data(TICKER, START_DATE, END_DATE)

    vwap_df = compute_rolling_vwap(raw)
    vol_results = analyze_volume_at_extension(vwap_df)
    plot_volume_at_extension(vol_results, vwap_df)
    plot_price_vwap_bands(vwap_df)

    lag_results, baseline = analyze_volume_lead_lag(vwap_df, extension_threshold=-2.0, max_lag=5)
    plot_volume_lead_lag(lag_results, baseline, threshold=-2.0)

    signaled = generate_signals(vwap_df)
    with_vol = compute_volatility(signaled)
    backtested, trades = run_backtest(with_vol)
    report_and_plot(backtested, trades)