"""
BTC multi-timeframe divergence entries (rules 4-5 of zihao's system)
====================================================================

Layers, all causal (a signal is used only after the bar that confirms it
has closed):

  Daily   : MACD(9/34) + RSI divergence over >= 3 swing peaks (i.e. at least
            two consecutive divergences) -> "alert" in that direction.
  4 hour  : within ``alert_days`` after the daily alert, the same kind of
            divergence on 4h bars -> "zone".
  15 min  : within ``zone_hours`` after the 4h zone, price breaks the last
            15m swing high (bearish case) and then closes back below it ->
            enter short at that close. Mirror image for bullish/long.

Each trade has a stop just beyond the breakout extreme, a target of
``target_r`` x risk and a time exit after ``max_hold_days``; exits are
simulated on 15m bars (stop assumed first when both are touched in one bar).

For comparison the same daily alerts are also traded directly at the daily
close (stop beyond the last 5 daily bars' extreme), and a looser "4h only"
setup (no daily alert needed) gives a larger sample.

Every trade is tagged with-trend / counter-trend relative to the daily
200-day MA, since the user's rule 1 sets direction by that line.

Data: OKX BTC/USDT via ccxt (1d, 4h, 15m), cached under data/crypto/.
    python src/strategies/btc_mtf_divergence_study.py --proxy http://127.0.0.1:8800
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from strategies.btc_ma_divergence_strategy import DivergenceConfig, divergence_events  # noqa: E402

TF = {"1d": pd.Timedelta(days=1), "4h": pd.Timedelta(hours=4), "15m": pd.Timedelta(minutes=15)}


@dataclass
class MTFConfig:
    n_peaks: int = 3              # >= 2 consecutive divergences
    pivot_k_daily: int = 5
    pivot_k_4h: int = 5
    pivot_k_15m: int = 3
    alert_days: float = 10        # how long a daily alert stays live
    zone_hours: float = 48        # how long a 4h zone stays live
    target_r: float = 2.0
    max_hold_days: float = 10
    min_risk: float = 0.003       # stop at least 0.3% away
    cost: float = 0.001           # per side


@dataclass
class Trade:
    setup: str
    direction: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    exit_time: pd.Timestamp
    exit: float
    outcome: str
    pnl_pct: float
    r_multiple: float
    with_trend: bool


def known_events(close: pd.Series, tf: str, n_peaks: int, k: int) -> pd.Series:
    """Divergence events re-indexed to the time they become known (bar close)."""
    ev = divergence_events(close, DivergenceConfig(n_peaks=n_peaks, pivot_k=k))
    ev.index = ev.index + TF[tf]
    return ev


def last_swing(series: pd.Series, before: pd.Timestamp, k: int, kind: str) -> Optional[float]:
    """Most recent swing high/low of 15m bars confirmed (k bars later) before ``before``."""
    window = series.loc[before - pd.Timedelta(days=5):before - TF["15m"]]
    if len(window) < 2 * k + 1:
        return None
    roll = window.rolling(2 * k + 1, center=True)
    ext = roll.max() if kind == "high" else roll.min()
    piv = window[(window == ext)]
    # confirmed only once k further bars have closed before `before`
    piv = piv[piv.index <= window.index[-1] - k * TF["15m"]]
    return None if piv.empty else float(piv.iloc[-1])


def find_15m_entry(m15: pd.DataFrame, direction: int, start: pd.Timestamp,
                   end: pd.Timestamp, cfg: MTFConfig):
    """Break of the last swing then close back inside -> (entry_time, entry, stop)."""
    kind = "high" if direction < 0 else "low"
    ref = last_swing(m15[kind], start, cfg.pivot_k_15m, kind)
    if ref is None:
        return None
    bars = m15.loc[start:end]
    broke, ext = False, None
    for t, b in bars.iterrows():
        if direction < 0:
            if b.high > ref:
                broke, ext = True, max(ext or b.high, b.high)
            if broke and b.close < ref:
                return t + TF["15m"], b.close, ext
        else:
            if b.low < ref:
                broke, ext = True, min(ext or b.low, b.low)
            if broke and b.close > ref:
                return t + TF["15m"], b.close, ext
    return None


def simulate(m15: pd.DataFrame, setup: str, direction: int, signal_time, entry_time,
             entry: float, stop: float, regime: pd.Series, cfg: MTFConfig) -> Optional[Trade]:
    risk = max(abs(entry - stop) / entry, cfg.min_risk)
    stop = entry * (1 - direction * risk)
    target = entry * (1 + direction * risk * cfg.target_r)
    path = m15.loc[entry_time:entry_time + pd.Timedelta(days=cfg.max_hold_days)]
    if path.empty:
        return None
    outcome, exit_t, exit_px = "time", path.index[-1] + TF["15m"], path.close.iloc[-1]
    for t, b in path.iterrows():
        hit_stop = b.high >= stop if direction < 0 else b.low <= stop
        hit_tgt = b.low <= target if direction < 0 else b.high >= target
        if hit_stop:
            outcome, exit_t, exit_px = "stop", t + TF["15m"], stop
            break
        if hit_tgt:
            outcome, exit_t, exit_px = "target", t + TF["15m"], target
            break
    pnl = direction * (exit_px / entry - 1) - 2 * cfg.cost
    reg = regime.asof(signal_time)
    return Trade(setup, direction, signal_time, entry_time, entry, stop, target, exit_t, exit_px,
                 outcome, pnl, pnl / risk, bool(reg == direction))


def run_study(d1: pd.DataFrame, h4: pd.DataFrame, m15: pd.DataFrame,
              cfg: MTFConfig = MTFConfig()) -> pd.DataFrame:
    ma = d1.close.rolling(200).mean()
    regime = np.sign(d1.close - ma).where(ma.notna())
    regime.index = regime.index + TF["1d"]           # known at the daily close

    ev_d = known_events(d1.close, "1d", cfg.n_peaks, cfg.pivot_k_daily)
    ev_h = known_events(h4.close, "4h", cfg.n_peaks, cfg.pivot_k_4h)
    start_ok = m15.index[0] + pd.Timedelta(days=5)
    trades: List[Trade] = []

    for t_d, s in ev_d.items():
        s = int(s)
        if t_d < start_ok or t_d > m15.index[-1]:
            continue
        # (a) daily only: enter on the daily close, stop beyond last 5 daily bars
        recent = d1.loc[:t_d - TF["1d"]].tail(5)
        stop = recent.high.max() if s < 0 else recent.low.min()
        entry = m15.close.asof(t_d - TF["15m"])
        tr = simulate(m15, "daily only", s, t_d, t_d, entry, stop, regime, cfg)
        if tr:
            trades.append(tr)
        # (b) daily -> 4h -> 15m
        zone = ev_h[(ev_h.index >= t_d) & (ev_h.index <= t_d + pd.Timedelta(days=cfg.alert_days))]
        zone = zone[zone == s]
        if not zone.empty:
            t_h = zone.index[0]
            hit = find_15m_entry(m15, s, t_h, t_h + pd.Timedelta(hours=cfg.zone_hours), cfg)
            if hit:
                tr = simulate(m15, "daily+4h+15m", s, t_d, hit[0], hit[1], hit[2], regime, cfg)
                if tr:
                    trades.append(tr)

    # (c) 4h zone + 15m trigger without the daily alert (bigger sample)
    for t_h, s in ev_h.items():
        s = int(s)
        if t_h < start_ok or t_h > m15.index[-1]:
            continue
        hit = find_15m_entry(m15, s, t_h, t_h + pd.Timedelta(hours=cfg.zone_hours), cfg)
        if hit:
            tr = simulate(m15, "4h+15m", s, t_h, hit[0], hit[1], hit[2], regime, cfg)
            if tr:
                trades.append(tr)

    return pd.DataFrame([asdict(t) for t in trades])


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    trades = trades.copy()
    trades["trend"] = np.where(trades.with_trend, "with trend", "counter trend")
    rows = []
    for keys, g in list(trades.groupby("setup")) + list(trades.groupby(["setup", "trend"])):
        name = keys if isinstance(keys, str) else " / ".join(keys)
        rows.append({
            "setup": name,
            "trades": len(g),
            "win_rate": (g.pnl_pct > 0).mean(),
            "avg_pnl": g.pnl_pct.mean(),
            "avg_R": g.r_multiple.mean(),
            "total_R": g.r_multiple.sum(),
            "compound_1x": (1 + g.pnl_pct).prod() - 1,
        })
    return pd.DataFrame(rows).set_index("setup").sort_index()


def main():
    from src.data.crypto_data_fetcher import fetch_ohlcv_ccxt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exchange", default="okx")
    ap.add_argument("--pair", default="BTC/USDT")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--proxy", default=None, help="e.g. http://127.0.0.1:8800")
    ap.add_argument("--n-peaks", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "crypto", "results"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.simplefilter("ignore", FutureWarning)

    data = {tf: fetch_ohlcv_ccxt(args.pair, tf, args.start, exchange=args.exchange, proxy=args.proxy)
            for tf in ("1d", "4h", "15m")}
    for tf, df in data.items():
        print(f"{tf}: {len(df)} bars {df.index[0]} -> {df.index[-1]}")

    cfg = MTFConfig(n_peaks=args.n_peaks)
    trades = run_study(data["1d"], data["4h"], data["15m"], cfg)
    os.makedirs(args.out, exist_ok=True)
    trades.to_csv(os.path.join(args.out, "btc_mtf_trades.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"\nConfig: {cfg}\n")
    print(summarize(trades).round(3).to_string())


if __name__ == "__main__":
    main()
