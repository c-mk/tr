"""
SPY Backtest Starter — yfinance edition
=========================================
Run this LOCALLY on your own machine (not in a sandboxed environment) since
it needs to reach Yahoo Finance's servers.

Setup (one time):
    pip install yfinance pandas numpy matplotlib

Run:
    python spy_backtest.py

What this does:
    1. Pulls historical daily SPY price data (free, no login needed)
    2. Runs a simple example strategy (moving average crossover) so you can
       see the whole pipeline work end to end
    3. Reports basic performance stats: win rate, avg win/loss, max
       drawdown, total return
    4. Plots an equity curve

Swap out the `generate_signals()` function with your own trade idea once
you're ready — everything downstream (backtest engine, stats, plotting)
stays the same.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CONFIG — edit these
# ---------------------------------------------------------------------------
TICKER = "SPY"
START_DATE = "2015-01-01"
END_DATE = None  # None = up to today
STARTING_CAPITAL = 10_000
SHORT_WINDOW = 20   # fast moving average (days)
LONG_WINDOW = 100    # slow moving average (days)
COMMISSION_PER_TRADE = 0.0  # set to e.g. 0.65 per contract if trading options


# ---------------------------------------------------------------------------
# 2. DATA
# ---------------------------------------------------------------------------
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check ticker/date range.")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]  # flatten if multiindex
    return df


# ---------------------------------------------------------------------------
# 3. STRATEGY — replace this with your own idea
# ---------------------------------------------------------------------------
def generate_signals(df, short_window=SHORT_WINDOW, long_window=LONG_WINDOW):
    """
    Example: simple moving average crossover.
    Signal = 1 (long) when short MA > long MA, else 0 (flat).

    Replace this entire function with your own rules. Whatever you return
    just needs a 'signal' column of 0s and 1s (or -1 for short, if you want
    to extend it later).
    """
    df = df.copy()
    df["ma_short"] = df["Close"].rolling(short_window).mean()
    df["ma_long"] = df["Close"].rolling(long_window).mean()
    df["signal"] = np.where(df["ma_short"] > df["ma_long"], 1, 0)
    return df


# ---------------------------------------------------------------------------
# 4. BACKTEST ENGINE — generally don't need to touch this part
# ---------------------------------------------------------------------------
def run_backtest(df, starting_capital=STARTING_CAPITAL, commission=COMMISSION_PER_TRADE):
    df = df.copy()
    df["position"] = df["signal"].shift(1).fillna(0)  # act on yesterday's signal (avoid lookahead)
    df["daily_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["position"] * df["daily_return"]

    # crude commission drag: charge once per position change
    df["trade"] = df["position"].diff().abs()
    df["strategy_return"] -= (df["trade"] * commission) / starting_capital

    df["equity_curve"] = starting_capital * (1 + df["strategy_return"]).cumprod()
    df["buy_hold_curve"] = starting_capital * (1 + df["daily_return"]).cumprod()
    return df


# ---------------------------------------------------------------------------
# 5. STATS
# ---------------------------------------------------------------------------
def compute_stats(df):
    trades = df[df["trade"] > 0]
    n_trades = len(trades)

    # identify individual trade P&L by tracking entries/exits
    entries = df.index[(df["position"] == 1) & (df["position"].shift(1) == 0)]
    exits = df.index[(df["position"] == 0) & (df["position"].shift(1) == 1)]

    trade_returns = []
    for entry in entries:
        later_exits = [e for e in exits if e > entry]
        exit_date = later_exits[0] if later_exits else df.index[-1]
        ret = (df.loc[exit_date, "Close"] / df.loc[entry, "Close"]) - 1
        trade_returns.append(ret)

    trade_returns = np.array(trade_returns)
    win_rate = (trade_returns > 0).mean() if len(trade_returns) else float("nan")
    avg_win = trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else float("nan")
    avg_loss = trade_returns[trade_returns < 0].mean() if (trade_returns < 0).any() else float("nan")

    total_return = df["equity_curve"].iloc[-1] / df["equity_curve"].iloc[0] - 1
    buy_hold_return = df["buy_hold_curve"].iloc[-1] / df["buy_hold_curve"].iloc[0] - 1

    running_max = df["equity_curve"].cummax()
    drawdown = (df["equity_curve"] - running_max) / running_max
    max_drawdown = drawdown.min()

    print("=" * 50)
    print(f"BACKTEST RESULTS: {TICKER}")
    print("=" * 50)
    print(f"Period:              {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Number of trades:    {len(trade_returns)}")
    print(f"Win rate:            {win_rate:.1%}" if not np.isnan(win_rate) else "Win rate: n/a")
    print(f"Avg win:             {avg_win:.2%}" if not np.isnan(avg_win) else "Avg win: n/a")
    print(f"Avg loss:            {avg_loss:.2%}" if not np.isnan(avg_loss) else "Avg loss: n/a")
    print(f"Strategy return:     {total_return:.2%}")
    print(f"Buy & hold return:   {buy_hold_return:.2%}")
    print(f"Max drawdown:        {max_drawdown:.2%}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# 6. PLOT
# ---------------------------------------------------------------------------
def plot_results(df):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df.index, df["equity_curve"], label="Strategy", linewidth=1.8)
    ax.plot(df.index, df["buy_hold_curve"], label="Buy & Hold", linewidth=1.2, alpha=0.7)
    ax.set_title(f"{TICKER} Strategy vs Buy & Hold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("backtest_equity_curve.png", dpi=150)
    print("\nChart saved to backtest_equity_curve.png")
    plt.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Fetching {TICKER} data from {START_DATE}...")
    raw = fetch_data(TICKER, START_DATE, END_DATE)

    print("Generating signals...")
    signaled = generate_signals(raw)

    print("Running backtest...")
    results = run_backtest(signaled)

    compute_stats(results)
    plot_results(results)   
    """
SPY Backtest Starter — yfinance edition
=========================================
Run this LOCALLY on your own machine (not in a sandboxed environment) since
it needs to reach Yahoo Finance's servers.

Setup (one time):
    pip install yfinance pandas numpy matplotlib

Run:
    python spy_backtest.py

What this does:
    1. Pulls historical daily SPY price data (free, no login needed)
    2. Runs a simple example strategy (moving average crossover) so you can
       see the whole pipeline work end to end
    3. Reports basic performance stats: win rate, avg win/loss, max
       drawdown, total return
    4. Plots an equity curve

Swap out the `generate_signals()` function with your own trade idea once
you're ready — everything downstream (backtest engine, stats, plotting)
stays the same.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CONFIG — edit these
# ---------------------------------------------------------------------------
TICKER = "SPY"
START_DATE = "2015-01-01"
END_DATE = None  # None = up to today
STARTING_CAPITAL = 10_000
SHORT_WINDOW = 20   # fast moving average (days)
LONG_WINDOW = 100    # slow moving average (days)
COMMISSION_PER_TRADE = 0.0  # set to e.g. 0.65 per contract if trading options


# ---------------------------------------------------------------------------
# 2. DATA
# ---------------------------------------------------------------------------
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check ticker/date range.")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]  # flatten if multiindex
    return df


# ---------------------------------------------------------------------------
# 3. STRATEGY — replace this with your own idea
# ---------------------------------------------------------------------------
def generate_signals(df, short_window=SHORT_WINDOW, long_window=LONG_WINDOW):
    """
    Example: simple moving average crossover.
    Signal = 1 (long) when short MA > long MA, else 0 (flat).

    Replace this entire function with your own rules. Whatever you return
    just needs a 'signal' column of 0s and 1s (or -1 for short, if you want
    to extend it later).
    """
    df = df.copy()
    df["ma_short"] = df["Close"].rolling(short_window).mean()
    df["ma_long"] = df["Close"].rolling(long_window).mean()
    df["signal"] = np.where(df["ma_short"] > df["ma_long"], 1, 0)
    return df


# ---------------------------------------------------------------------------
# 4. BACKTEST ENGINE — generally don't need to touch this part
# ---------------------------------------------------------------------------
def run_backtest(df, starting_capital=STARTING_CAPITAL, commission=COMMISSION_PER_TRADE):
    df = df.copy()
    df["position"] = df["signal"].shift(1).fillna(0)  # act on yesterday's signal (avoid lookahead)
    df["daily_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["position"] * df["daily_return"]

    # crude commission drag: charge once per position change
    df["trade"] = df["position"].diff().abs()
    df["strategy_return"] -= (df["trade"] * commission) / starting_capital

    df["equity_curve"] = starting_capital * (1 + df["strategy_return"]).cumprod()
    df["buy_hold_curve"] = starting_capital * (1 + df["daily_return"]).cumprod()
    return df


# ---------------------------------------------------------------------------
# 5. STATS
# ---------------------------------------------------------------------------
def compute_stats(df):
    trades = df[df["trade"] > 0]
    n_trades = len(trades)

    # identify individual trade P&L by tracking entries/exits
    entries = df.index[(df["position"] == 1) & (df["position"].shift(1) == 0)]
    exits = df.index[(df["position"] == 0) & (df["position"].shift(1) == 1)]

    trade_returns = []
    for entry in entries:
        later_exits = [e for e in exits if e > entry]
        exit_date = later_exits[0] if later_exits else df.index[-1]
        ret = (df.loc[exit_date, "Close"] / df.loc[entry, "Close"]) - 1
        trade_returns.append(ret)

    trade_returns = np.array(trade_returns)
    win_rate = (trade_returns > 0).mean() if len(trade_returns) else float("nan")
    avg_win = trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else float("nan")
    avg_loss = trade_returns[trade_returns < 0].mean() if (trade_returns < 0).any() else float("nan")

    total_return = df["equity_curve"].iloc[-1] / df["equity_curve"].iloc[0] - 1
    buy_hold_return = df["buy_hold_curve"].iloc[-1] / df["buy_hold_curve"].iloc[0] - 1

    running_max = df["equity_curve"].cummax()
    drawdown = (df["equity_curve"] - running_max) / running_max
    max_drawdown = drawdown.min()

    print("=" * 50)
    print(f"BACKTEST RESULTS: {TICKER}")
    print("=" * 50)
    print(f"Period:              {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Number of trades:    {len(trade_returns)}")
    print(f"Win rate:            {win_rate:.1%}" if not np.isnan(win_rate) else "Win rate: n/a")
    print(f"Avg win:             {avg_win:.2%}" if not np.isnan(avg_win) else "Avg win: n/a")
    print(f"Avg loss:            {avg_loss:.2%}" if not np.isnan(avg_loss) else "Avg loss: n/a")
    print(f"Strategy return:     {total_return:.2%}")
    print(f"Buy & hold return:   {buy_hold_return:.2%}")
    print(f"Max drawdown:        {max_drawdown:.2%}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# 6. PLOT
# ---------------------------------------------------------------------------
def plot_results(df):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df.index, df["equity_curve"], label="Strategy", linewidth=1.8)
    ax.plot(df.index, df["buy_hold_curve"], label="Buy & Hold", linewidth=1.2, alpha=0.7)
    ax.set_title(f"{TICKER} Strategy vs Buy & Hold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("backtest_equity_curve.png", dpi=150)
    print("\nChart saved to backtest_equity_curve.png")
    plt.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Fetching {TICKER} data from {START_DATE}...")
    raw = fetch_data(TICKER, START_DATE, END_DATE)

    print("Generating signals...")
    signaled = generate_signals(raw)

    print("Running backtest...")
    results = run_backtest(signaled)

    compute_stats(results)
    plot_results(results)