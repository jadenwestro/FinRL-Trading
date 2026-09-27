"""
BTC 200-day MA regime + MACD/RSI divergence (daily layer of zihao's rules)
=========================================================================

The trader's discretionary rules, daily-timeframe part only:
  1. Regime: close above the 200-day MA -> long bias, below -> short bias.
  2. MACD (fast 9 / slow 34, signal 9) histogram peaks falling while price
     makes higher highs (N consecutive peaks) -> bearish divergence; mirror
     image for bullish divergence.
  3. RSI(14) peak lower while price peak higher -> bearish divergence
     (mirror for bullish).
  When 2 (and optionally 3) fire against the regime -> take profit / reverse.
  The 4h / 15m entry refinement (rules 4-5) needs intraday data and is not
  modelled here.

All signals are causal: a swing high/low at day i is only known at day i+k
(k = pivot half-window), and positions change on the close after a signal.

Variants compared by ``main()``:
  - long/short vs long/cash on the 200-day regime
  - a hysteresis band around the MA (fewer whipsaw trades)
  - divergence overlay: exit fully, or cut to half size, against the trend

Usage:
    python src/strategies/btc_ma_divergence_strategy.py [--cost 0.001]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from strategies.btc_momentum_strategy import crypto_metrics, vectorized_check  # noqa: E402


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def macd_hist(close: pd.Series, fast: int = 9, slow: int = 34, signal: int = 9) -> pd.Series:
    macd = ema(close, fast) - ema(close, slow)
    return macd - ema(macd, signal)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def ma_regime(close: pd.Series, window: int = 200, band: float = 0.0) -> pd.Series:
    """+1 above MA*(1+band), -1 below MA*(1-band), previous state inside the band."""
    ma = close.rolling(window).mean()
    reg = pd.Series(np.nan, index=close.index)
    reg[close > ma * (1 + band)] = 1.0
    reg[close < ma * (1 - band)] = -1.0
    return reg.ffill().fillna(0.0)


# ---------------------------------------------------------------------------
# Divergence detection (causal)
# ---------------------------------------------------------------------------

@dataclass
class DivergenceConfig:
    pivot_k: int = 5          # swing point = extreme of a 2k+1 day window
    n_peaks: int = 2          # 3 = the "三连背离" (three falling MACD peaks)
    use_macd: bool = True
    use_rsi: bool = True
    macd_fast: int = 9
    macd_slow: int = 34
    macd_signal: int = 9
    rsi_period: int = 14


def divergence_events(close: pd.Series, cfg: DivergenceConfig) -> pd.Series:
    """Series indexed by the date a divergence becomes known: -1 bearish, +1 bullish."""
    hist = macd_hist(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    r = rsi(close, cfg.rsi_period)
    k = cfg.pivot_k
    events: Dict[pd.Timestamp, float] = {}

    for kind, sign in (("high", -1.0), ("low", 1.0)):
        roll = close.rolling(2 * k + 1, center=True)
        ext = roll.max() if kind == "high" else roll.min()
        pivots = np.flatnonzero((close == ext) & ext.notna())
        agg = np.max if kind == "high" else np.min
        rows = []
        for i in pivots:
            w = slice(max(i - k, 0), i + k + 1)
            known = close.index[min(i + k, len(close) - 1)]
            rows.append((close.iloc[i], agg(hist.iloc[w].values), agg(r.iloc[w].values), known))

        for j in range(cfg.n_peaks - 1, len(rows)):
            seq = rows[j - cfg.n_peaks + 1:j + 1]
            p, h, rr = [x[0] for x in seq], [x[1] for x in seq], [x[2] for x in seq]
            pairs = lambda v: list(zip(v, v[1:]))  # noqa: E731
            if kind == "high":
                price_ok = all(b > a for a, b in pairs(p))
                macd_ok = all(b < a for a, b in pairs(h)) and h[0] > 0
                rsi_ok = rr[-1] < rr[-2]
            else:
                price_ok = all(b < a for a, b in pairs(p))
                macd_ok = all(b > a for a, b in pairs(h)) and h[0] < 0
                rsi_ok = rr[-1] > rr[-2]
            if price_ok and (macd_ok or not cfg.use_macd) and (rsi_ok or not cfg.use_rsi):
                events[seq[-1][3]] = sign

    return pd.Series(events, dtype=float).sort_index()


# ---------------------------------------------------------------------------
# Position rules
# ---------------------------------------------------------------------------

def divergence_overlay(regime: pd.Series, events: pd.Series,
                       against: str = "flat", allow_short: bool = False) -> pd.Series:
    """Target position given the MA regime and divergence events.

    A divergence against the regime (bearish in an uptrend, bullish in a
    downtrend) cuts exposure: ``against="flat"`` -> 0, ``"half"`` -> 0.5x,
    ``"reverse"`` -> flip. A divergence with the regime restores full size.
    A regime change always resets to the regime direction.
    """
    ev = events.reindex(regime.index).fillna(0.0).values
    reg = regime.values
    scale = {"flat": 0.0, "half": 0.5, "reverse": -1.0}[against]
    out = np.zeros(len(reg))
    cur, last = 0.0, 0.0
    for t in range(len(reg)):
        g = reg[t]
        if g != last:
            cur, last = g, g
        e = ev[t]
        if e != 0 and g != 0:
            cur = g * scale if e == -g else g
        out[t] = cur
    pos = pd.Series(out, index=regime.index)
    return pos if allow_short else pos.clip(lower=0.0)


def backtest_positions(close: pd.Series, pos: pd.Series, cost: float = 0.001,
                       short_fee: float = 0.05) -> pd.Series:
    """Equity curve: position set at close t earns t+1; cost per unit turnover;
    shorts pay ``short_fee`` a year (borrow / funding)."""
    eq = vectorized_check(close, pos, cost)
    fee = pos.shift(1).fillna(0.0).clip(upper=0).abs() * short_fee / 365
    return eq * (1 - fee).cumprod()


def build_variants(close: pd.Series) -> Dict[str, pd.Series]:
    reg = ma_regime(close)
    reg5 = ma_regime(close, band=0.05)
    div2 = divergence_events(close, DivergenceConfig(n_peaks=2))
    div3 = divergence_events(close, DivergenceConfig(n_peaks=3))
    return {
        "Buy & Hold": pd.Series(1.0, index=close.index),
        "MA200 long/short (rule 1 as is)": reg,
        "MA200 long/short + div3 reverse (rules 1-3)": divergence_overlay(reg, div3, "reverse", allow_short=True),
        "MA200 long/cash": reg.clip(lower=0),
        "MA200 ±5% band, long/cash": reg5.clip(lower=0),
        "±5% band + div3 exit": divergence_overlay(reg5, div3, "flat"),
        "±5% band + div2 half size": divergence_overlay(reg5, div2, "half"),
    }


def main():
    from src.data.crypto_data_fetcher import fetch_crypto_daily

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-start", default="2013-01-01")
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--source", default="auto")
    args = ap.parse_args()
    warnings.simplefilter("ignore", FutureWarning)

    close = fetch_crypto_daily("BTC", start=args.data_start, source=args.source)["close"]
    variants = build_variants(close)
    periods = [("2015-01-01", None), ("2015-01-01", "2020-12-31"), ("2021-01-01", None)]
    pd.set_option("display.width", 200)
    for start, end in periods:
        rows = {}
        for name, pos in variants.items():
            c = close.loc[start:end]
            eq = backtest_positions(c, pos.reindex(c.index).fillna(0.0), args.cost)
            m = crypto_metrics(eq)
            m["in_market"] = float((pos.reindex(c.index) != 0).mean())
            m["trades"] = int((pos.reindex(c.index).diff().fillna(0) != 0).sum())
            rows[name] = m
        df = pd.DataFrame(rows).T[["cagr", "sharpe", "max_drawdown", "calmar", "in_market", "trades"]]
        print(f"\n== {start} -> {end or close.index[-1].date()}  (cost {args.cost:.2%}, 365-day annualisation)")
        print(df.round(3).to_string())


if __name__ == "__main__":
    main()
