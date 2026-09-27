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
from typing import Dict, List, Optional

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
    # "hump": compare MACD-histogram humps (runs between zero crossings), the
    #         way the user marks divergences on a chart. "pivot": compare
    #         price swing points found with a 2k+1 bar window.
    method: str = "hump"
    pivot_k: int = 5          # pivot method: swing = extreme of a 2k+1 bar window
    min_hump_bars: int = 3    # hump method: zero-line flickers shorter than this are ignored
    n_peaks: int = 3          # peaks compared; 3 = at least two divergences in a row
    use_macd: bool = True
    use_rsi: bool = True
    macd_fast: int = 9
    macd_slow: int = 34
    macd_signal: int = 9
    rsi_period: int = 14


def divergence_events(close: pd.Series, cfg: DivergenceConfig,
                      high: Optional[pd.Series] = None,
                      low: Optional[pd.Series] = None) -> pd.Series:
    """Series indexed by the bar a divergence becomes known: -1 bearish, +1 bullish."""
    if cfg.method == "hump":
        return _hump_divergence_events(close, cfg, high, low)
    return _pivot_divergence_events(close, cfg)


def _smoothed_sign(hist: pd.Series, min_bars: int) -> np.ndarray:
    """Sign of the histogram with runs shorter than ``min_bars`` folded into
    the previous run, so a one-bar dip through zero doesn't split a hump.
    Causal: a short run is only folded while it is still short."""
    raw = np.sign(hist.fillna(0.0).values)
    out = raw.copy()
    run_start = 0
    for t in range(1, len(raw)):
        if raw[t] != raw[t - 1]:
            run_start = t
        if t - run_start + 1 < min_bars and run_start > 0:
            out[t] = out[run_start - 1]
    return out


def _hump_divergence_events(close: pd.Series, cfg: DivergenceConfig,
                            high: Optional[pd.Series], low: Optional[pd.Series]) -> pd.Series:
    """Divergence across consecutive same-sign MACD-histogram humps.

    Bearish, checked on every bar of a positive hump: over the last
    ``n_peaks`` positive humps (the current one included) price highs keep
    rising while the histogram peaks keep falling, the current RSI peak is
    below the first hump's RSI peak, and the histogram has just turned down
    (momentum fading). Fires at most once per hump. Bullish is the mirror.
    """
    hist = macd_hist(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    r = rsi(close, cfg.rsi_period).values
    hv = hist.values
    hi = (high if high is not None else close).values
    lo = (low if low is not None else close).values
    sign = _smoothed_sign(hist, cfg.min_hump_bars)
    idx = close.index
    events: Dict[pd.Timestamp, float] = {}

    for hump_sign, direction in ((1.0, -1.0), (-1.0, 1.0)):
        more = (lambda a, b: b > a) if hump_sign > 0 else (lambda a, b: b < a)
        ext = max if hump_sign > 0 else min
        done: List[dict] = []
        cur: Optional[dict] = None
        for t in range(1, len(hv)):
            if np.isnan(hv[t]) or np.isnan(r[t]):
                continue
            if sign[t] == hump_sign:
                px_t = hi[t] if hump_sign > 0 else lo[t]
                if cur is None:
                    cur = {"p": px_t, "h": hv[t], "r": r[t], "fired": False}
                else:
                    cur["p"], cur["h"], cur["r"] = ext(cur["p"], px_t), ext(cur["h"], hv[t]), ext(cur["r"], r[t])
                if cur["fired"] or len(done) < cfg.n_peaks - 1:
                    continue
                chain = done[-(cfg.n_peaks - 1):] + [cur]
                p = [c["p"] for c in chain]
                h = [abs(c["h"]) for c in chain]
                price_ok = all(more(a, b) for a, b in zip(p, p[1:]))
                macd_ok = all(b < a for a, b in zip(h, h[1:]))
                rsi_ok = not more(chain[0]["r"], cur["r"]) and cur["r"] != chain[0]["r"]
                turning = abs(hv[t]) < abs(hv[t - 1])
                if price_ok and turning and (macd_ok or not cfg.use_macd) and (rsi_ok or not cfg.use_rsi):
                    events[idx[t]] = direction
                    cur["fired"] = True
            elif cur is not None:
                done.append(cur)
                cur = None

    return pd.Series(events, dtype=float).sort_index()


def _pivot_divergence_events(close: pd.Series, cfg: DivergenceConfig) -> pd.Series:
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


def long_cash_rules(close: pd.Series, events: pd.Series, slow: int = 150, band: float = 0.05,
                    fast: Optional[int] = None) -> pd.Series:
    """Long-or-cash rules found to hold up best in the 2015-2026 study.

    - Trend: long only while the slow-MA regime (with ±band hysteresis) is up.
    - Bearish divergence -> sell; bullish divergence -> buy back.
    - Optional fast MA: inside an uptrend, step aside while close < fast MA and
      buy back when it closes above again. A divergence exit is only undone by a
      bullish divergence or by a fresh dip under / reclaim of the fast MA.
    """
    reg = ma_regime(close, slow, band).values
    ev = events.reindex(close.index).fillna(0.0).values
    px = close.values
    maf = close.rolling(fast).mean().values if fast else None
    out = np.zeros(len(px))
    cur, last, armed = 0.0, 0.0, True
    for t in range(len(px)):
        g = reg[t]
        if g != last:
            cur, last, armed = (1.0 if g > 0 else 0.0), g, True
        if g > 0:
            fast_ok = maf is None or px[t] > maf[t]
            if maf is not None and not fast_ok:
                armed = True
            if ev[t] < 0:
                cur, armed = 0.0, False
            elif ev[t] > 0:
                cur, armed = 1.0, True
            elif maf is not None:
                if cur > 0 and not fast_ok:
                    cur = 0.0
                elif cur == 0 and fast_ok and armed:
                    cur = 1.0
        else:
            cur = 0.0
        out[t] = cur
    return pd.Series(out, index=close.index)


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
    div = divergence_events(close, DivergenceConfig())   # >= 2 divergences in a row
    return {
        "Buy & Hold": pd.Series(1.0, index=close.index),
        "MA200 long/short (rule 1 as is)": reg,
        "MA200 long/short + divergence reverse (rules 1-3)": divergence_overlay(reg, div, "reverse", allow_short=True),
        "MA200 long/cash": reg.clip(lower=0),
        "MA200 ±5% band, long/cash": reg5.clip(lower=0),
        "±5% band + divergence exit": divergence_overlay(reg5, div, "flat"),
        "±5% band + divergence half size": divergence_overlay(reg5, div, "half"),
        "MA150 ±5% + divergence (recommended)": long_cash_rules(close, div, slow=150),
        "MA150 ±5% + divergence + MA50 step-aside": long_cash_rules(close, div, slow=150, fast=50),
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
