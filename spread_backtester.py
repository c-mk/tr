"""
Spread Mean-Reversion Backtester (HyperLiquid vs MT5)
=======================================================

Replicates the logic seen in the ArbitrageLab "Pair Analysis" backtester:
  spread = HL_close - MT5_close
  mu, sigma computed over a lookback window
  SHORT the spread when it opens >= mu + entry_sigma * sigma, cover when close <= mu
  LONG  the spread when it opens <= mu - entry_sigma * sigma, cover when close >= mu

This version adds two things the original report did NOT have, per your
requirements:
  1. Risk-based position sizing (risk a fixed % of equity per trade)
  2. A hard max-drawdown kill switch that halts new entries

BACKTEST ONLY. No live orders are placed. No API keys required to run this
file as-is (it ships with a synthetic sample-data generator). To use real
data, replace `load_price_data()` with a loader that reads your own
historical export from HyperLiquid + MT5 (see notes at the bottom).

IMPORTANT CALIBRATION NOTE ON POSITION SIZING
-----------------------------------------------
This backtest trades a *spread* between two different instruments on two
different venues (a HyperLiquid perp and an MT5 forex position). "1 unit of
size" here means 1 unit of notional on EACH leg, scaled by `point_value`
(how much P&L, in your account currency, one unit of spread-price-movement
is worth for the position size you'd actually place on both legs). This
number depends on your MT5 lot size / pip value and your HyperLiquid
contract size and leverage -- it is NOT automatically derived from the
data. You must calibrate `point_value` yourself before trusting the dollar
P&L figures. Until you do, treat PnL as being in "spread units" (like the
original report), not dollars.

This is not financial advice. Backtested results (especially small-sample,
single-window backtests like this) do not guarantee live performance.
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
    lookback_bars: int = 3441          # matches the 3-working-day / 1-min window in the report
    rolling: bool = True               # True: recompute mu/sigma on a rolling window each bar
                                        # False: compute mu/sigma once over the whole lookback (matches PDF exactly)
    entry_sigma: float = 2.0           # enter when spread is this many sigma from mu
    exit_sigma: float = 0.0            # exit when spread returns to mu (+ this many sigma)
    stop_sigma_extra: float = 1.0      # stop-loss sits (entry_sigma + this) sigma from mu
                                        # e.g. entry at 2sigma, stop at 3sigma -> 1 extra sigma of room

    # --- Cost model (mirrors the "Estimate cost/side" box in the report) ---
    # HyperLiquid fee schedule (your account's actual current tier):
    #   Taker: 0.0090%   Maker: 0.0030%
    # order_type determines which one applies to entries/exits below.
    order_type: str = "market"         # "market" (taker, guarantees fill) or "limit" (maker, fill not guaranteed)
    hl_taker_fee_pct: float = 0.0090
    hl_maker_fee_pct: float = 0.0030
    market_order_slippage_pct: float = 0.01   # flat fallback if no slippage_curve is provided
    slippage_curve: dict = field(default_factory=dict)  # {notional_usd: slippage_pct}, from build_slippage_curve()
    mt5_spread: float = 0.001          # MT5 bid/ask spread, in price units (charged as half-spread/side)
    mt5_fee: float = 0.0               # MT5 flat fee per side, in price units
    reference_price: float = 159.308   # used to convert HL fee % into price units

    # --- Risk management ---
    risk_pct_per_trade: float = 0.01   # fraction of current equity risked per trade (0.01 = 1%)
    point_value: float = field(default=0.0)  # 0 => auto-derived as 1/reference_price (see __post_init__)
    max_drawdown_pct: float = 0.10     # kill switch: halt new entries if drawdown from equity peak exceeds this
    starting_equity: float = 10_000.0

    def __post_init__(self):
        if self.point_value == 0.0:
            # point_value = $ P&L per unit size (= $ notional) per unit of spread movement,
            # assuming a linear USD-denominated contract on both legs, notional-matched.
            # Verify HL's actual contract multiplier for this specific asset before trusting
            # dollar P&L for real position sizing -- HIP-3 synthetic markets can differ.
            self.point_value = 1.0 / self.reference_price

    @property
    def hl_fee_pct(self) -> float:
        return self.hl_taker_fee_pct if self.order_type == "market" else self.hl_maker_fee_pct

    def _slippage_pct_for_notional(self, notional_usd: float) -> float:
        """Interpolates the calibrated slippage curve if available, else falls
        back to the flat market_order_slippage_pct estimate."""
        if self.order_type != "market":
            return 0.0
        if not self.slippage_curve:
            return self.market_order_slippage_pct
        sizes = sorted(self.slippage_curve.keys())
        pcts = [self.slippage_curve[s] for s in sizes]
        return float(np.interp(notional_usd, sizes, pcts))

    def cost_per_side(self, notional_usd: Optional[float] = None) -> float:
        """Cost in spread price units for one side (entry OR exit) of a trade
        with the given dollar notional. If notional_usd is None, uses the
        flat slippage estimate (no size-dependent lookup)."""
        hl_fee = self.hl_fee_pct / 100.0 * self.reference_price
        slip_pct = self._slippage_pct_for_notional(notional_usd) if notional_usd is not None \
            else (self.market_order_slippage_pct if self.order_type == "market" else 0.0)
        slippage = slip_pct / 100.0 * self.reference_price
        return hl_fee + slippage + 0.5 * self.mt5_spread + self.mt5_fee


@dataclass
class Trade:
    side: str               # "long" or "short"
    entry_time: pd.Timestamp
    entry_price: float      # spread value at entry
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None   # "target", "stop"
    gross_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Data loading — MT5 (your CSV export)
# ---------------------------------------------------------------------------

def load_mt5_csv(path: str, tz: Optional[str] = None) -> pd.DataFrame:
    """
    Parses a raw MT5 "Export Bars" CSV, no header, UTF-16 encoded, with
    7 fields per row:
        DateTime, Open, High, Low, Close, TickVolume, RealVolume

    RealVolume is always 0 for retail forex/CFD symbols (OTC market, no
    consolidated tape) -- it's kept only for schema completeness.

    tz: if your broker's server time has a known fixed UTC offset, pass it
    (e.g. "Etc/GMT-3") to localize+convert to UTC. If None, timestamps are
    left as-is (naive) -- fine for testing the mechanics, but you MUST
    align this to real UTC before trusting live/paper results, since HL
    timestamps are UTC and a few hours of offset will silently corrupt
    the spread calculation.
    """
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
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")  # ASSUMES server time == UTC, verify with your broker
    df = df.set_index("datetime").sort_index()
    return df.rename(columns={"close": "mt5_close"})[["mt5_close", "tick_volume"]]


# ---------------------------------------------------------------------------
# Data loading — HyperLiquid (REST historical + WS live stub)
# ---------------------------------------------------------------------------

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
HL_WS_URL = "wss://api.hyperliquid.xyz/ws"


def fetch_hl_candles_rest(coin: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Bulk HISTORICAL candles via HyperLiquid's public REST info endpoint.
    No API key needed -- this is public market data.

    Only the most recent 5000 candles are returned per call, so this pages
    forward automatically until start_ms..end_ms is fully covered.

    `coin` for a HIP-3 synthetic asset needs the dex prefix, e.g. "xyz:JPY".
    """
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
            break  # safety: avoid infinite loop if server returns same page
        cur_start = last_t + 1

        if len(batch) < 5000:
            break  # fewer than the page cap means we reached the end

        time.sleep(0.2)  # be polite to the public endpoint / stay under rate limits

    if not all_rows:
        return pd.DataFrame(columns=["hl_close"]).set_index(pd.DatetimeIndex([], tz="UTC", name="datetime"))

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["hl_close"] = df["c"].astype(float)
    return df.set_index("datetime")[["hl_close"]].sort_index()


def stream_hl_candles_ws(coin: str, interval: str, on_candle: Callable[[dict], None]):
    """
    LIVE candle stream via HyperLiquid's websocket -- for later, once you
    move from backtest into paper/live trading. Not used by the backtest
    engine below. Requires the `websocket-client` package
    (`pip install websocket-client`).

    Note: the websocket only pushes candles going forward from the moment
    you subscribe -- it has no historical backfill. Use
    fetch_hl_candles_rest() for anything historical.
    """
    import websocket  # local import: only needed if you actually call this

    def _on_open(ws):
        sub = {"method": "subscribe", "subscription": {"type": "candle", "coin": coin, "interval": interval}}
        ws.send(json.dumps(sub))

    def _on_message(ws, message):
        msg = json.loads(message)
        if msg.get("channel") == "candle":
            on_candle(msg["data"])

    ws_app = websocket.WebSocketApp(HL_WS_URL, on_open=_on_open, on_message=_on_message)
    ws_app.run_forever()


def fetch_hl_l2_book(coin: str) -> dict:
    """
    Live L2 order book snapshot from HyperLiquid. No key needed (public data).
    Returns {"bids": [{"px": float, "sz": float}, ...], "asks": [...]}
    sorted best-to-worst, i.e. bids[0] = best bid, asks[0] = best ask.

    NOTE: this is the CURRENT book only -- HyperLiquid's public API does not
    expose historical order book depth, so this cannot reconstruct what
    liquidity looked like on the specific days in your backtest window. It's
    a best-effort proxy: "if the book looks like this today, entries of this
    size would move price by about this much."
    """
    resp = requests.post(HL_INFO_URL, json={"type": "l2Book", "coin": coin},
                          headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    data = resp.json()
    bid_levels, ask_levels = data["levels"]
    return {
        "bids": [{"px": float(l["px"]), "sz": float(l["sz"])} for l in bid_levels],
        "asks": [{"px": float(l["px"]), "sz": float(l["sz"])} for l in ask_levels],
        "time": data.get("time"),
    }


def estimate_market_impact_pct(levels: list, notional_usd: float, mid_price: float) -> float:
    """
    Walks one side of the book (asks for a buy, bids for a sell), accumulating
    size until notional_usd is filled, and returns the volume-weighted average
    fill price's slippage from mid, as a percent of mid_price.

    If the book doesn't have enough depth to fill the requested notional,
    returns the impact of sweeping the entire visible book (a lower bound --
    real impact would be worse).
    """
    remaining = notional_usd
    sz_filled, cost_filled = 0.0, 0.0

    for level in levels:
        level_notional = level["px"] * level["sz"]
        take_notional = min(remaining, level_notional)
        take_sz = take_notional / level["px"]
        sz_filled += take_sz
        cost_filled += take_sz * level["px"]
        remaining -= take_notional
        if remaining <= 0:
            break

    if sz_filled == 0:
        return 0.0

    vwap = cost_filled / sz_filled
    return abs(vwap - mid_price) / mid_price * 100.0


def build_slippage_curve(coin: str, notional_points: list) -> dict:
    """
    Fetches the current book once and estimates % slippage at several
    notional sizes, e.g. build_slippage_curve("xyz:JPY", [1000, 5000, 10000, 50000]).
    Returns {notional: slippage_pct}. Use this to eyeball how fast slippage
    grows with your intended position size, then pick a
    market_order_slippage_pct that matches YOUR typical trade size (or wire
    in interpolation via numpy.interp for a size-varying cost model).
    """
    book = fetch_hl_l2_book(coin)
    mid = (book["bids"][0]["px"] + book["asks"][0]["px"]) / 2.0
    curve = {}
    for notional in notional_points:
        buy_impact = estimate_market_impact_pct(book["asks"], notional, mid)
        sell_impact = estimate_market_impact_pct(book["bids"], notional, mid)
        curve[notional] = max(buy_impact, sell_impact)  # worst side, conservative
    return curve


# ---------------------------------------------------------------------------
# Combined loader used by the backtester
# ---------------------------------------------------------------------------

def load_price_data(
    mt5_csv_path: Optional[str] = None,
    hl_coin: str = "xyz:JPY",
    hl_interval: str = "5m",
    mt5_tz: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by UTC timestamp with columns: hl_close, mt5_close.

    If mt5_csv_path is given: loads your real MT5 export, then pulls the
    matching HyperLiquid historical window via REST and inner-joins the two
    on timestamp (this is the "3441 aligned 1m bars" step from the report).

    If mt5_csv_path is None: falls back to synthetic data so the engine is
    runnable standalone with zero external dependencies / network calls.
    """
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
    """Synthetic mean-reverting spread on top of a random-walk base price,
    loosely shaped like the JPY example in the report, purely for testing
    the engine's mechanics."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-08-11 07:26", periods=n_bars, freq="5min", tz="UTC")

    base = 159.0 + np.cumsum(rng.normal(0, 0.01, n_bars))

    # Ornstein-Uhlenbeck-ish mean-reverting spread around mu ~ -0.001
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
        d["buy_exit"] = mu + self.cfg.exit_sigma * sigma
        d["buy_stop"] = mu - (self.cfg.entry_sigma + self.cfg.stop_sigma_extra) * sigma
        d["sell_entry"] = mu + self.cfg.entry_sigma * sigma
        d["sell_exit"] = mu - self.cfg.exit_sigma * sigma
        d["sell_stop"] = mu + (self.cfg.entry_sigma + self.cfg.stop_sigma_extra) * sigma
        return d

    def _position_size(self, equity: float, stop_distance: float) -> float:
        """Risk-% sizing: size such that hitting the stop loses
        risk_pct_per_trade of current equity."""
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

        for ts, row in d.iterrows():
            o, c = row["hl_close"] - row["mt5_close"], row["spread"]  # open proxy == prior bar's spread; simplified to use bar's own spread
            spread_open = row["spread"]   # NOTE: using close-as-open approximation for 1-bar signals;
            spread_close = row["spread"]  # for tighter fidelity, shift by 1 bar to use true bar-open.

            # --- manage open trade first ---
            if open_trade is not None:
                exit_reason = None
                if open_trade.side == "long":
                    if spread_close >= row["buy_exit"]:
                        exit_reason = "target"
                    elif spread_close <= row["buy_stop"]:
                        exit_reason = "stop"
                else:  # short
                    if spread_close <= row["sell_exit"]:
                        exit_reason = "target"
                    elif spread_close >= row["sell_stop"]:
                        exit_reason = "stop"

                if exit_reason:
                    exit_price = spread_close
                    direction = 1 if open_trade.side == "long" else -1
                    notional_usd = open_trade.size  # size IS dollar notional under this point_value convention
                    gross = direction * (exit_price - open_trade.entry_price) * open_trade.size * cfg.point_value
                    commission = 2 * cfg.cost_per_side(notional_usd) * open_trade.size * cfg.point_value  # entry+exit
                    net = gross - commission

                    open_trade.exit_time = ts
                    open_trade.exit_price = exit_price
                    open_trade.exit_reason = exit_reason
                    open_trade.gross_pnl = gross
                    open_trade.commission = commission
                    open_trade.net_pnl = net

                    equity += net
                    self.trades.append(open_trade)
                    open_trade = None

            # --- kill switch check ---
            peak_equity = max(peak_equity, equity)
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            if drawdown >= cfg.max_drawdown_pct and not halted:
                halted = True
                self.halted_at = ts

            # --- consider new entry (only if flat and not halted) ---
            if open_trade is None and not halted:
                if spread_open <= row["buy_entry"]:
                    stop_dist = row["buy_entry"] - row["buy_stop"]
                    size = self._position_size(equity, stop_dist)
                    if size > 0:
                        open_trade = Trade("long", ts, spread_open, size)
                elif spread_open >= row["sell_entry"]:
                    stop_dist = row["sell_stop"] - row["sell_entry"]
                    size = self._position_size(equity, stop_dist)
                    if size > 0:
                        open_trade = Trade("short", ts, spread_open, size)

            self.equity_curve.append((ts, equity))

        self.equity_df = pd.DataFrame(self.equity_curve, columns=["time", "equity"]).set_index("time")
        self.bands_df = d
        return self.summary()

    # -----------------------------------------------------------------
    def performance_stats(self, periods_per_year: int = 252) -> dict:
        """
        Sharpe and Sortino computed on DAILY-resampled equity returns
        (not per-trade or per-minute returns -- those overstate significance
        for a strategy trading dozens of times a day). Annualized assuming
        ~252 trading days/year.

        Sharpe: mean(daily_return) / std(daily_return) * sqrt(252)
        Sortino: same, but denominator only uses the std of NEGATIVE daily
        returns (downside deviation) -- doesn't penalize upside volatility.
        """
        if self.equity_df.empty:
            return {"sharpe": None, "sortino": None}

        daily_equity = self.equity_df["equity"].resample("1D").last().dropna()
        daily_returns = daily_equity.pct_change().dropna()

        if len(daily_returns) < 2 or daily_returns.std() == 0:
            return {"sharpe": None, "sortino": None, "note": "not enough daily variation to compute"}

        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(periods_per_year)

        downside = daily_returns[daily_returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            sortino = None  # no downside days in this window -- can't compute
        else:
            sortino = daily_returns.mean() / downside.std() * np.sqrt(periods_per_year)

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "n_daily_obs": len(daily_returns),
        }

    def summary(self) -> dict:
        eq = self.equity_df["equity"] if not self.equity_df.empty else pd.Series([self.cfg.starting_equity])
        running_max = eq.cummax()
        dd = (running_max - eq) / running_max

        if not self.trades:
            stats = {
                "trades": 0,
                "win_rate_pct": 0.0,
                "total_gross_pnl": 0.0,
                "total_net_pnl": 0.0,
                "total_commission": 0.0,
                "avg_trade": 0.0,
                "max_drawdown_pct": dd.max() * 100 if len(dd) else 0.0,
                "ending_equity": eq.iloc[-1],
                "halted_by_kill_switch": self.halted_at is not None,
                "halted_at": self.halted_at,
            }
            stats.update(self.performance_stats())
            return stats

        pnl = [t.net_pnl for t in self.trades]
        gross = [t.gross_pnl for t in self.trades]
        commission = [t.commission for t in self.trades]
        wins = [p for p in pnl if p > 0]
        stats = {
            "trades": len(self.trades),
            "win_rate_pct": 100 * len(wins) / len(pnl),
            "total_gross_pnl": sum(gross),
            "total_net_pnl": sum(pnl),
            "total_commission": sum(commission),
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

    # -----------------------------------------------------------------
    def plot(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        d = self.bands_df
        axes[0].plot(d.index, d["spread"], label="spread", color="orange", linewidth=0.8)
        axes[0].plot(d.index, d["mu"], label="mu", color="black", linewidth=0.8)
        axes[0].plot(d.index, d["buy_entry"], "--", color="green", linewidth=0.7, label="buy entry (-2σ)")
        axes[0].plot(d.index, d["sell_entry"], "--", color="red", linewidth=0.7, label="sell entry (+2σ)")
        for t in self.trades:
            marker = "^" if t.side == "long" else "v"
            color = "green" if t.side == "long" else "red"
            axes[0].scatter(t.entry_time, t.entry_price, marker=marker, color=color, s=40, zorder=5)
            if t.exit_time:
                axes[0].scatter(t.exit_time, t.exit_price, marker="x", color="black", s=30, zorder=5)
        axes[0].set_title("Spread with entries/exits")
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
# Example run
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Out-of-sample / walk-forward testing
# ---------------------------------------------------------------------------

def train_test_split_test(data: pd.DataFrame, cfg: StrategyConfig, train_frac: float = 0.5) -> dict:
    """
    Splits the data chronologically into an in-sample ("train") window and
    an out-of-sample ("test") window, then runs the SAME frozen cfg
    (no re-tuning) on each independently. This is the direct check for the
    trap from before: does a threshold/config that looks good on one window
    still look good on a DIFFERENT window it wasn't picked to flatter?

    Returns {"train": (backtester, stats), "test": (backtester, stats)}.
    """
    split_idx = int(len(data) * train_frac)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    bt_train = SpreadBacktester(train_data, cfg)
    stats_train = bt_train.run()

    bt_test = SpreadBacktester(test_data, cfg)
    stats_test = bt_test.run()

    return {"train": (bt_train, stats_train), "test": (bt_test, stats_test)}


def walk_forward_test(data: pd.DataFrame, cfg: StrategyConfig, n_folds: int = 6, compound: bool = True) -> dict:
    """
    Splits data into n_folds sequential, non-overlapping, chronological
    windows and runs the SAME frozen cfg (no re-tuning between folds) on
    each one, in order. This is the real test of robustness: a strategy
    with genuine edge should perform reasonably (even if not identically)
    across most folds, not just the one window you happened to tune on.

    compound=True: each fold's starting equity = previous fold's ending
    equity (mimics actually trading straight through time).
    compound=False: every fold restarts at cfg.starting_equity (cleaner
    for comparing per-fold performance without earlier folds' luck bleeding in).

    Returns {"fold_stats": DataFrame, "backtesters": [SpreadBacktester, ...]}
    """
    fold_size = len(data) // n_folds
    fold_stats = []
    backtesters = []
    running_equity = cfg.starting_equity

    for i in range(n_folds):
        start = i * fold_size
        end = len(data) if i == n_folds - 1 else (i + 1) * fold_size
        fold_data = data.iloc[start:end]
        if len(fold_data) < cfg.lookback_bars // 4:
            continue  # skip folds too short to compute meaningful mu/sigma

        fold_cfg = StrategyConfig(**{**cfg.__dict__, "starting_equity": running_equity if compound else cfg.starting_equity})
        bt = SpreadBacktester(fold_data, fold_cfg)
        stats = bt.run()
        stats["fold"] = i + 1
        stats["fold_start"] = fold_data.index.min()
        stats["fold_end"] = fold_data.index.max()
        fold_stats.append(stats)
        backtesters.append(bt)

        if compound:
            running_equity = stats["ending_equity"]

    return {"fold_stats": pd.DataFrame(fold_stats), "backtesters": backtesters}


if __name__ == "__main__":
    # Optional: calibrate slippage against HyperLiquid's LIVE order book for
    # the notional sizes you'd actually trade. Requires network access to
    # api.hyperliquid.xyz (this call is commented out by default since it
    # can't run in a sandboxed/offline environment):
    #
    #   curve = build_slippage_curve("xyz:JPY", [1000, 5000, 10000, 25000, 50000])
    #   print(curve)  # e.g. {1000: 0.004, 5000: 0.011, 10000: 0.023, ...}
    #
    # Then pass it in: StrategyConfig(..., slippage_curve=curve)
    # The backtester will interpolate slippage based on EACH trade's actual
    # notional (which varies over time as equity/risk sizing changes),
    # rather than using one flat guess for every trade.

    cfg = StrategyConfig(
        entry_sigma=2.0,
        exit_sigma=0.0,
        stop_sigma_extra=1.0,
        risk_pct_per_trade=0.01,     # risk 1% of equity per trade
        max_drawdown_pct=0.10,       # halt new entries after 10% drawdown from peak
        order_type="market",         # market/taker: guarantees fill, costs more per side (recommended -- see notes above)
        # point_value left at 0 -> auto-derived as 1/reference_price (dollar-notional-matched legs)
        rolling=False,               # False = static mu/sigma over the whole window, matching the PDF exactly
    )

    # Real run against your MT5 export + live HyperLiquid REST pull:
    #   data = load_price_data(
    #       mt5_csv_path="/mnt/user-data/uploads/USDJPYM1.csv",
    #       hl_coin="xyz:JPY",
    #       hl_interval="1m",
    #       mt5_tz=None,   # set your broker's server timezone if known, e.g. "Etc/GMT-3"
    #   )
    # Falls back to synthetic data when no CSV path is given, so this file
    # still runs standalone with no network access:
    data = load_price_data(mt5_csv_path=None)
    bt = SpreadBacktester(data, cfg)
    result = bt.run()

    print("=== Backtest Summary ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    print("\n=== Trades (first 10) ===")
    print(bt.trades_frame().head(10))

    bt.plot(save_path=" spread_backtest_plot.png")
    print("spread_backtest_plot.png")