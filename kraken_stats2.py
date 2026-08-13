"""
Kraken empirical statistics toolkit
------------------------------------
Pulls historical OHLC data from Kraken's public API and computes the
foundational statistics you need BEFORE building a strategy:

  1. Realized volatility (hourly + annualized)
  2. Lag-1 autocorrelation of returns (momentum vs. mean-reversion tendency)
  3. Hurst exponent (trending vs. mean-reverting regime, >0.5 vs <0.5)
  4. Half-life of mean reversion (Ornstein-Uhlenbeck fit)
  5. Empirical frequency of moves >= a given dollar/percent threshold

Run this on your VPS (or any machine with internet access) — Kraken's
public API needs no authentication for OHLC data.

Usage:
    python3 kraken_stats2.py --pair XBTUSD --interval 60 --threshold 1000 --days 730 --save-csv btc_2yr.csv
"""

import argparse
import numpy as np
import pandas as pd
import requests


def fetch_ohlc(pair: str, interval: int = 60, since: int = None) -> tuple:
    """
    Pull one page of OHLC candles from Kraken's public API.
    interval is in minutes: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
    Kraken returns roughly the last 720 candles per call, starting from
    `since` (a unix timestamp) if provided.

    Returns (df, last_ts) where last_ts is Kraken's pagination cursor —
    pass it back in as `since` to fetch the next page.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": pair, "interval": interval}
    if since is not None:
        params["since"] = since

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("error"):
        raise RuntimeError(f"Kraken API error: {data['error']}")

    result_key = [k for k in data["result"].keys() if k != "last"][0]
    rows = data["result"][result_key]
    last_ts = data["result"]["last"]

    df = pd.DataFrame(
        rows,
        columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
    )
    df["time"] = pd.to_datetime(df["time"], unit="s")
    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("time")
    return df, last_ts


def fetch_ohlc_range(
    pair: str,
    interval: int = 60,
    days: int = 730,
    request_delay: float = 1.5,
) -> pd.DataFrame:
    """
    Paginate through Kraken's OHLC endpoint to build a long history.
    Kraken caps each response at ~720 candles, so pulling 2 years of
    hourly data (~17,500 candles) takes roughly 25 sequential calls.

    request_delay adds a pause between calls to stay well within
    Kraken's public API rate limits — don't set this to 0.
    """
    import time as time_module

    now_ts = int(time_module.time())
    since = now_ts - days * 86400

    all_frames = []
    call_count = 0

    while True:
        df, last_ts = fetch_ohlc(pair, interval, since=since)
        call_count += 1

        if df.empty:
            break

        all_frames.append(df)
        newest_ts = int(df.index.max().timestamp())

        print(
            f"  call {call_count}: {len(df)} candles, "
            f"up to {df.index.max()} ({newest_ts})"
        )

        # Stop once we've reached (or nearly reached) the current time,
        # or if Kraken's cursor stops advancing (safety against infinite loop)
        if last_ts <= since or newest_ts >= now_ts - interval * 60:
            break

        since = last_ts
        time_module.sleep(request_delay)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    return combined


def realized_volatility(df: pd.DataFrame, interval_minutes: int) -> dict:
    """Compute realized volatility from log returns, scaled to hourly + annualized."""
    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()

    periods_per_hour = 60 / interval_minutes
    periods_per_year = periods_per_hour * 24 * 365

    per_period_std = log_returns.std()
    hourly_std = per_period_std * np.sqrt(periods_per_hour)
    annualized_std = per_period_std * np.sqrt(periods_per_year)

    return {
        "log_returns": log_returns,
        "per_period_std_pct": per_period_std * 100,
        "hourly_std_pct": hourly_std * 100,
        "annualized_std_pct": annualized_std * 100,
    }


def autocorrelation(log_returns: pd.Series, lag: int = 1) -> float:
    """
    Lag-N autocorrelation of returns.
    Positive  -> momentum (moves tend to continue in the same direction)
    Negative  -> mean reversion (moves tend to reverse)
    Near zero -> close to a random walk, no obvious edge from this alone
    """
    return log_returns.autocorr(lag=lag)


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """
    Rescaled range estimate of the Hurst exponent.
    H ~ 0.5 -> random walk
    H  > 0.5 -> trending / momentum
    H  < 0.5 -> mean-reverting
    """
    prices = series.dropna().values
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
    tau = [t if t > 0 else 1e-8 for t in tau]
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0]


def half_life_mean_reversion(series: pd.Series) -> float:
    """
    Fit an Ornstein-Uhlenbeck process to estimate the half-life of mean
    reversion, in units of candles (multiply by your interval to get time).

    delta_y_t = lambda * (y_{t-1} - mean) + noise
    half_life = -ln(2) / lambda
    """
    y = series.dropna()
    y_lag = y.shift(1).dropna()
    y = y.loc[y_lag.index]

    delta_y = y - y_lag
    y_lag_centered = y_lag - y_lag.mean()

    # OLS slope: delta_y = lambda * y_lag_centered
    lam = np.polyfit(y_lag_centered, delta_y, 1)[0]

    if lam >= 0:
        return float("inf")  # not mean-reverting under this simple fit
    return -np.log(2) / lam


def threshold_move_frequency(df: pd.DataFrame, threshold_dollars: float) -> dict:
    """
    Empirical (not modeled) frequency of a >= threshold_dollars move
    within a single candle, split by direction.
    """
    dollar_change = df["close"] - df["close"].shift(1)
    n = len(dollar_change.dropna())

    up_moves = (dollar_change >= threshold_dollars).sum()
    down_moves = (dollar_change <= -threshold_dollars).sum()

    return {
        "n_candles": n,
        "up_move_freq_pct": 100 * up_moves / n,
        "down_move_freq_pct": 100 * down_moves / n,
        "either_direction_freq_pct": 100 * (up_moves + down_moves) / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Kraken empirical stats")
    parser.add_argument("--pair", default="XBTUSD", help="Kraken pair, e.g. XBTUSD, ETHUSD")
    parser.add_argument("--interval", type=int, default=60, help="Candle interval in minutes")
    parser.add_argument("--threshold", type=float, default=1000, help="Dollar move threshold")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of history to pull (paginates automatically if >~30 days at hourly interval)",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional path to save the pulled OHLC data as CSV, so you don't have to re-fetch it",
    )
    args = parser.parse_args()

    print(f"Fetching {args.pair} at {args.interval}-min candles from Kraken ({args.days} days)...")
    df = fetch_ohlc_range(args.pair, args.interval, days=args.days)

    if df.empty:
        print("No data returned — check your pair code and interval.")
        return

    print(f"\nPulled {len(df)} candles total, {df.index.min()} to {df.index.max()}\n")

    if args.save_csv:
        df.to_csv(args.save_csv)
        print(f"Saved raw OHLC data to {args.save_csv}\n")

    vol = realized_volatility(df, args.interval)
    print("=== Volatility ===")
    print(f"Per-candle std dev:     {vol['per_period_std_pct']:.3f}%")
    print(f"Hourly std dev:         {vol['hourly_std_pct']:.3f}%")
    print(f"Annualized std dev:     {vol['annualized_std_pct']:.1f}%\n")

    ac1 = autocorrelation(vol["log_returns"], lag=1)
    print("=== Autocorrelation (lag-1 returns) ===")
    print(f"Lag-1 autocorrelation:  {ac1:.4f}")
    if ac1 > 0.05:
        print("  -> Positive: some momentum tendency at this interval")
    elif ac1 < -0.05:
        print("  -> Negative: some mean-reversion tendency at this interval")
    else:
        print("  -> Near zero: close to random walk, no obvious edge here alone")
    print()

    h = hurst_exponent(df["close"])
    print("=== Hurst Exponent ===")
    print(f"H = {h:.3f}")
    if h > 0.55:
        print("  -> Trending regime")
    elif h < 0.45:
        print("  -> Mean-reverting regime")
    else:
        print("  -> Close to random walk")
    print()

    hl = half_life_mean_reversion(df["close"])
    print("=== Mean-Reversion Half-Life ===")
    if hl == float("inf"):
        print("  Series is not mean-reverting under this simple linear fit")
    else:
        print(f"  Half-life: {hl:.1f} candles ({hl * args.interval:.0f} minutes)")
    print()

    freq = threshold_move_frequency(df, args.threshold)
    print(f"=== Frequency of >= ${args.threshold:.0f} move per candle ===")
    print(f"Candles analyzed:       {freq['n_candles']}")
    print(f"Up moves:               {freq['up_move_freq_pct']:.2f}%")
    print(f"Down moves:             {freq['down_move_freq_pct']:.2f}%")
    print(f"Either direction:       {freq['either_direction_freq_pct']:.2f}%")


if __name__ == "__main__":
    main()