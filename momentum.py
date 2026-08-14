"""
Spread Momentum Breakout Backtester (HyperLiquid vs MT5)
=======================================================

Adapted to trade a momentum/breakout logic with trailing stops, 
partial scale-outs (TPs), and a UTC time-window filter.

  spread = HL_close - MT5_close
  mu, sigma computed over a lookback window
  LONG the spread when it breaks out above upper band
  SHORT the spread when it breaks down below lower band

BACKTEST ONLY. No live orders are placed. No API keys required to run this
file as-is (it ships with a synthetic sample-data generator).
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import json


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    # --- Signal parameters ---
    lookback_bars: int = 720           # default to 12 hours
    rolling: bool = True               
    entry_sigma: float = 1.5           # enter when spread breaks this many sigma from mu
    exit_sigma: float = 0.0            # (Legacy) kept for backwards compatibility in bands
    stop_sigma_extra: float = 1.0      # (Legacy) kept for backwards compatibility in bands

    # --- Momentum Exits ---
    tp_levels_sigma: list = field(default_factory=lambda: [1.0, 2.0, 3.0]) # TP levels from entry
    tp_allocations: list = field(default_factory=lambda: [0.33, 0.33, 0.34]) # % to close
    trailing_stop_sigma: float = 0.5   # Distance to trail behind peak spread

    # --- Cost model ---
    order_type: str = "market"         
    hl_taker_fee_pct: float = 0.0090
    hl_maker_fee_pct: float = 0.0030
    market_order_slippage_pct: float = 0.01   
    slippage_curve: dict = field(default_factory=dict)  
    mt5_spread: float = 0.001          
    mt5_fee: float = 0.0               
    reference_price: float = 159.308   

    # --- Risk management ---
    risk_pct_per_trade: float = 0.01   
    point_value: float = field(default=0.0)  
    max_drawdown_pct: float = 0.15     
    starting_equity: float = 10_000.0

    def __post_init__(self):
        if self.point_value == 0.0:
            self.point_value = 1.0 / self.reference_price

    @property
    def hl_fee_pct(self) -> float:
        return self.hl_taker_fee_pct if self.order_type == "market" else self.hl_maker_fee_pct

    def _slippage_pct_for_notional(self, notional_usd: float) -> float:
        if self.order_type != "market":
            return 0.0
        if not self.slippage_curve:
            return self.market_order_slippage_pct
        sizes = sorted(self.slippage_curve.keys())
        pcts = [self.slippage_curve[s] for s in sizes]
        return float(np.interp(notional_usd, sizes, pcts))

    def cost_per_side(self, notional_usd: Optional[float] = None) -> float:
        hl_fee = self.hl_fee_pct / 100.0 * self.reference_price
        slip_pct = self._slippage_pct_for_notional(notional_usd) if notional_usd is not None \
            else (self.market_order_slippage_pct if self.order_type == "market" else 0.0)
        slippage = slip_pct / 100.0 * self.reference_price
        return hl_fee + slippage + 0.5 * self.mt5_spread + self.mt5_fee


@dataclass
class Trade:
    side: str               
    entry_time: pd.Timestamp
    entry_price: float      
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None   
    gross_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mt5_csv(path: str, tz: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=["datetime", "open", "high", "low", "close", "tick_volume", "real_volume"],
        encoding="utf-16",
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M")
    if tz:
        df["datetime"] = df["datetime"].dt.tz_localize(tz).dt.tz_convert("UTC")
    else:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")  
    df = df.set_index("datetime").sort_index()
    return df.rename(columns={"close": "mt5_close"})[["mt5_close", "tick_volume"]]


HL_INFO_URL = "https://api.hyperliquid.xyz/info"
HL_WS_URL = "wss://api.hyperliquid.xyz/ws"


def fetch_hl_candles_rest(coin: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows = []
    cur_start = start_ms

    while cur_start < end_ms:
        payload = {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval, "startTime": cur_start, "endTime": end_ms},
        }
        resp = requests.post(HL_INFO_URL, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        all_rows.extend(batch)
        last_t = batch[-1]["t"]
        if last_t <= cur_start:
            break  
        cur_start = last_t + 1

        if len(batch) < 5000:
            break  

        time.sleep(0.2)  

    if not all_rows:
        return pd.DataFrame(columns=["hl_close"]).set_index(pd.DatetimeIndex([], tz="UTC", name="datetime"))

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["hl_close"] = df["c"].astype(float)
    return df.set_index("datetime")[["hl_close"]].sort_index()


def load_price_data(
    mt5_csv_path: Optional[str] = None,
    hl_coin: str = "xyz:JPY",
    hl_interval: str = "1m",
    mt5_tz: Optional[str] = None,
) -> pd.DataFrame:
    if not mt5_csv_path:
        return _generate_sample_data()

    mt5_df = load_mt5_csv(mt5_csv_path, tz=mt5_tz)

    start_ms = int(mt5_df.index.min().timestamp() * 1000)
    end_ms = int(mt5_df.index.max().timestamp() * 1000)
    hl_df = fetch_hl_candles_rest(hl_coin, hl_interval, start_ms, end_ms)

    merged = mt5_df.join(hl_df, how="inner")[["hl_close", "mt5_close"]].dropna()
    if merged.empty:
        raise ValueError(
            "No overlapping timestamps between MT5 CSV and HyperLiquid candles. "
            "Check mt5_tz alignment and hl_coin symbol."
        )
    return merged


def _generate_sample_data(n_bars: int = 3441, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-08-11 07:26", periods=n_bars, freq="1min", tz="UTC")

    base = 159.0 + np.cumsum(rng.normal(0, 0.01, n_bars))
    spread = np.zeros(n_bars)
    mu, theta, sigma_shock = -0.001, 0.05, 0.02
    for i in range(1, n_bars):
        spread[i] = spread[i - 1] + theta * (mu - spread[i - 1]) + rng.normal(0, sigma_shock)

    hl_close = base + spread
    mt5_close = base

    return pd.DataFrame({"hl_close": hl_close, "mt5_close": mt5_close}, index=t)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class SpreadBacktester:
    def __init__(self, data: pd.DataFrame, cfg: StrategyConfig):
        self.data = data.copy()
        self.cfg = cfg
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple] = []
        self.halted_at: Optional[pd.Timestamp] = None

    def _compute_bands(self) -> pd.DataFrame:
        d = self.data
        d["spread"] = d["hl_close"] - d["mt5_close"]

        if self.cfg.rolling:
            mu = d["spread"].rolling(self.cfg.lookback_bars, min_periods=self.cfg.lookback_bars // 4).mean()
            sigma = d["spread"].rolling(self.cfg.lookback_bars, min_periods=self.cfg.lookback_bars // 4).std()
        else:
            mu_static = d["spread"].iloc[: self.cfg.lookback_bars].mean()
            sigma_static = d["spread"].iloc[: self.cfg.lookback_bars].std()
            mu = pd.Series(mu_static, index=d.index)
            sigma = pd.Series(sigma_static, index=d.index)

        d["mu"] = mu
        d["sigma"] = sigma
        d["buy_entry"] = mu - self.cfg.entry_sigma * sigma
        d["sell_entry"] = mu + self.cfg.entry_sigma * sigma
        return d

    def _position_size(self, equity: float, stop_distance: float) -> float:
        if stop_distance <= 0 or self.cfg.point_value <= 0:
            return 0.0
        risk_amount = equity * self.cfg.risk_pct_per_trade
        return risk_amount / (stop_distance * self.cfg.point_value)

    def run(self):
        d = self._compute_bands().dropna(subset=["mu", "sigma"])
        cfg = self.cfg
        equity = cfg.starting_equity
        peak_equity = equity
        open_trade: Optional[Trade] = None
        halted = False

        current_stop = None
        current_tps = []
        current_tp_allocs = []

        for ts, row in d.iterrows():
            spread_open = row["spread"]
            spread_close = row["spread"] 
            sigma = row["sigma"]

            # --- manage open trade first ---
            if open_trade is not None:
                exit_reason = None
                is_partial = False
                pct_to_close = 1.0

                if open_trade.side == "long":
                    # 1. Update Trailing Stop
                    new_stop = spread_close - (sigma * cfg.trailing_stop_sigma)
                    if current_stop is None or new_stop > current_stop:
                        current_stop = new_stop
                    
                    # 2. Check Exits
                    if spread_close <= current_stop:
                        exit_reason = "trailing_stop"
                    elif current_tps and spread_close >= current_tps[0]:
                        exit_reason = "TP_hit"
                        is_partial = True
                        pct_to_close = current_tp_allocs[0]
                        current_tps.pop(0)
                        current_tp_allocs.pop(0)

                else:  # short
                    # 1. Update Trailing Stop
                    new_stop = spread_close + (sigma * cfg.trailing_stop_sigma)
                    if current_stop is None or new_stop < current_stop:
                        current_stop = new_stop
                    
                    # 2. Check Exits
                    if spread_close >= current_stop:
                        exit_reason = "trailing_stop"
                    elif current_tps and spread_close <= current_tps[0]:
                        exit_reason = "TP_hit"
                        is_partial = True
                        pct_to_close = current_tp_allocs[0]
                        current_tps.pop(0)
                        current_tp_allocs.pop(0)

                if exit_reason:
                    exit_price = spread_close
                    direction = 1 if open_trade.side == "long" else -1
                    
                    # Compute exit size based on partial or full closure
                    close_size = open_trade.size if not is_partial else (open_trade.size * pct_to_close)
                    
                    notional_usd = close_size 
                    gross = direction * (exit_price - open_trade.entry_price) * close_size * cfg.point_value
                    commission = 2 * cfg.cost_per_side(notional_usd) * close_size * cfg.point_value
                    net = gross - commission

                    equity += net
                    
                    # Log the closed fraction as a finalized trade
                    closed_trade = Trade(
                        side=open_trade.side,
                        entry_time=open_trade.entry_time,
                        entry_price=open_trade.entry_price,
                        size=close_size,
                        exit_time=ts,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        gross_pnl=gross,
                        commission=commission,
                        net_pnl=net
                    )
                    self.trades.append(closed_trade)
                    
                    if is_partial:
                        open_trade.size -= close_size
                    else:
                        open_trade = None
                        current_stop = None
                        current_tps = []
                        current_tp_allocs = []

            # --- kill switch check ---
            peak_equity = max(peak_equity, equity)
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            if drawdown >= cfg.max_drawdown_pct and not halted:
                halted = True
                self.halted_at = ts

            # --- consider new entry (Time Filter: 12:00 to 16:00 UTC) ---
            trade_window_open = 12 <= ts.hour <= 16

            if open_trade is None and not halted and trade_window_open:
                # Momentum Logic: Buy upper breakout, Short lower breakout
                if spread_open >= row["sell_entry"]:  # Breakout UP
                    stop_dist = sigma * cfg.trailing_stop_sigma
                    size = self._position_size(equity, stop_dist)
                    if size > 0:
                        open_trade = Trade("long", ts, spread_open, size)
                        current_stop = spread_open - stop_dist
                        current_tps = [spread_open + (sigma * tp) for tp in cfg.tp_levels_sigma]
                        current_tp_allocs = list(cfg.tp_allocations)
                        
                elif spread_open <= row["buy_entry"]: # Breakout DOWN
                    stop_dist = sigma * cfg.trailing_stop_sigma
                    size = self._position_size(equity, stop_dist)
                    if size > 0:
                        open_trade = Trade("short", ts, spread_open, size)
                        current_stop = spread_open + stop_dist
                        current_tps = [spread_open - (sigma * tp) for tp in cfg.tp_levels_sigma]
                        current_tp_allocs = list(cfg.tp_allocations)

            self.equity_curve.append((ts, equity))

        self.equity_df = pd.DataFrame(self.equity_curve, columns=["time", "equity"]).set_index("time")
        self.bands_df = d
        return self.summary()

    def performance_stats(self, periods_per_year: int = 252) -> dict:
        if self.equity_df.empty:
            return {"sharpe": None, "sortino": None}

        daily_equity = self.equity_df["equity"].resample("1D").last().dropna()
        daily_returns = daily_equity.pct_change().dropna()

        if len(daily_returns) < 2 or daily_returns.std() == 0:
            return {"sharpe": None, "sortino": None, "n_daily_obs": len(daily_returns)}

        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(periods_per_year)

        downside = daily_returns[daily_returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            sortino = None  
        else:
            sortino = daily_returns.mean() / downside.std() * np.sqrt(periods_per_year)

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "n_daily_obs": len(daily_returns),
        }

    def summary(self) -> dict:
        if not self.trades:
            return {"trades": 0, "message": "No trades triggered."}
        pnl = [t.net_pnl for t in self.trades]
        wins = [p for p in pnl if p > 0]
        eq = self.equity_df["equity"]
        running_max = eq.cummax()
        dd = (running_max - eq) / running_max
        stats = {
            "trades": len(self.trades),
            "win_rate_pct": 100 * len(wins) / len(pnl),
            "total_net_pnl": sum(pnl),
            "avg_trade": sum(pnl) / len(pnl),
            "max_drawdown_pct": dd.max() * 100,
            "ending_equity": eq.iloc[-1],
            "halted_by_kill_switch": self.halted_at is not None,
            "halted_at": self.halted_at,
        }
        stats.update(self.performance_stats())
        return stats

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def plot(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        d = self.bands_df
        axes[0].plot(d.index, d["spread"], label="spread", color="orange", linewidth=0.8)
        axes[0].plot(d.index, d["mu"], label="mu", color="black", linewidth=0.8)
        axes[0].plot(d.index, d["buy_entry"], "--", color="green", linewidth=0.7, label="lower band (-σ)")
        axes[0].plot(d.index, d["sell_entry"], "--", color="red", linewidth=0.7, label="upper band (+σ)")
        
        for t in self.trades:
            marker = "^" if t.side == "long" else "v"
            color = "green" if t.side == "long" else "red"
            axes[0].scatter(t.entry_time, t.entry_price, marker=marker, color=color, s=40, zorder=5)
            if t.exit_time:
                axes[0].scatter(t.exit_time, t.exit_price, marker="x", color="black", s=30, zorder=5)
        
        axes[0].set_title("Spread Breakout with Trailing Exits")
        axes[0].legend(fontsize=8)

        axes[1].plot(self.equity_df.index, self.equity_df["equity"], color="blue")
        axes[1].set_title("Equity curve")
        if self.halted_at:
            axes[1].axvline(self.halted_at, color="red", linestyle="--", label="kill switch triggered")
            axes[1].legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
        return fig


# ---------------------------------------------------------------------------
# Out-of-sample / walk-forward testing
# ---------------------------------------------------------------------------

def train_test_split_test(data: pd.DataFrame, cfg: StrategyConfig, train_frac: float = 0.5) -> dict:
    split_idx = int(len(data) * train_frac)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    bt_train = SpreadBacktester(train_data, cfg)
    stats_train = bt_train.run()

    bt_test = SpreadBacktester(test_data, cfg)
    stats_test = bt_test.run()

    return {"train": (bt_train, stats_train), "test": (bt_test, stats_test)}


def walk_forward_test(data: pd.DataFrame, cfg: StrategyConfig, n_folds: int = 6, compound: bool = True) -> dict:
    fold_size = len(data) // n_folds
    fold_stats = []
    backtesters = []
    running_equity = cfg.starting_equity

    for i in range(n_folds):
        start = i * fold_size
        end = len(data) if i == n_folds - 1 else (i + 1) * fold_size
        fold_data = data.iloc[start:end]
        if len(fold_data) < cfg.lookback_bars // 4:
            continue  

        fold_cfg = StrategyConfig(**{**cfg.__dict__, "starting_equity": running_equity if compound else cfg.starting_equity})
        bt = SpreadBacktester(fold_data, fold_cfg)
        stats = bt.run()
        stats["fold"] = i + 1
        stats["fold_start"] = fold_data.index.min()
        stats["fold_end"] = fold_data.index.max()
        fold_stats.append(stats)
        backtesters.append(bt)

        if compound:
            running_equity = stats.get("ending_equity", running_equity)

    return {"fold_stats": pd.DataFrame(fold_stats), "backtesters": backtesters}


if __name__ == "__main__":
    cfg = StrategyConfig(
        entry_sigma=1.5,
        risk_pct_per_trade=0.01,     
        max_drawdown_pct=0.15,       
        order_type="market",         
        rolling=True,               
        lookback_bars=720,
        starting_equity=10_000.0,
        tp_levels_sigma=[1.0, 2.0, 3.0], 
        tp_allocations=[0.33, 0.33, 0.34], 
        trailing_stop_sigma=0.5 
    )

    data = load_price_data(mt5_csv_path=None)
    bt = SpreadBacktester(data, cfg)
    result = bt.run()

    print("=== Backtest Summary ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    print("\n=== Trades (first 10) ===")
    print(bt.trades_frame().head(10))