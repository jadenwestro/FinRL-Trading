"""
Daily trend rules plus a 4-hour layer, for BTC / ETH / SOL
==========================================================

zihao asked whether adding the 4h chart to the daily long/cash rules can
raise the win rate and give more signals. The daily rules stay the base
(see crypto_trend_signals.py): buy above the 150-day MA x 1.05, sell all
below MA x 0.95, sell all on a bearish 3-hump divergence, and in the
current version sell half below the 50-day MA and buy it back above it x 1.02.

4h layers tested on top (all long or cash, never short):
  4h trim N      : replace the 50-day trim with a 4h one: while the daily
                   rules hold, a 4h close below the 4h MA(N) sells half and a
                   close above MA(N) x 1.01 buys it back.
  4h 7/34 half   : daily rules as now, but hold only half while the 4h
                   MA7 is below the 4h MA34.
  4h early half  : also buy half before the daily buy signal, once the
                   daily close is above the 150-day MA and the 4h MA7 is
                   above MA34; it is sold if either condition fails.
  4h divergence  : while the daily rules hold, a bearish 3-hump MACD/RSI
                   divergence on 4h bars sells half, a bullish one buys it back.

All signals are causal: a daily value is used after its day closes, a 4h
value after its 4h bar closes, and a position set at a 4h close earns from
the next bar. Costs are charged per unit of turnover.

"Win rate" counts sell actions: a sale wins when its price, after costs,
is above the average cost of the coins held at that moment.

Data: OKX 15m candles (resampled to 4h and daily) via ccxt, cached under
data/crypto/.
    python src/strategies/crypto_daily_4h_study.py --proxy http://127.0.0.1:8800
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from strategies.btc_ma_divergence_strategy import DivergenceConfig, divergence_events  # noqa: E402
from strategies.crypto_trend_signals import Rules, run_rules  # noqa: E402

H4 = pd.Timedelta(hours=4)
DAY = pd.Timedelta(days=1)
BARS_PER_YEAR = 365 * 6


def to_4h_grid(daily: pd.Series, h4_index: pd.DatetimeIndex) -> np.ndarray:
    """Daily values (indexed by day open) as known at each 4h bar's close."""
    known = pd.Series(daily.values, index=daily.index + DAY)
    return known.reindex(h4_index + H4, method="ffill").fillna(0.0).values


def daily_state(c1d: pd.Series, h4_index: pd.DatetimeIndex, half_ma: Optional[int]):
    ev = divergence_events(c1d, DivergenceConfig())
    pos, _ = run_rules(c1d, Rules(half_ma=half_ma), ev)
    ma150 = c1d.rolling(150).mean()
    above = (c1d > ma150).astype(float)
    return to_4h_grid(pos, h4_index), to_4h_grid(above, h4_index)


def trim_layer(base: np.ndarray, c4: np.ndarray, ma: np.ndarray, band: float = 0.01) -> np.ndarray:
    """Hold half of ``base`` after a close below ``ma`` until a close above ma x (1+band)."""
    out = base.copy()
    trimmed = False
    for i in range(len(base)):
        if base[i] <= 0 or np.isnan(ma[i]):
            trimmed = False
            continue
        if not trimmed and c4[i] < ma[i]:
            trimmed = True
        elif trimmed and c4[i] > ma[i] * (1 + band):
            trimmed = False
        if trimmed:
            out[i] = base[i] * 0.5
    return out


def divergence_layer(base: np.ndarray, ev4: np.ndarray) -> np.ndarray:
    out = base.copy()
    trimmed = False
    for i in range(len(base)):
        if base[i] <= 0:
            trimmed = False
            continue
        if ev4[i] < 0:
            trimmed = True
        elif ev4[i] > 0:
            trimmed = False
        if trimmed:
            out[i] = base[i] * 0.5
    return out


def build_variants(c15: pd.Series) -> Dict[str, pd.Series]:
    c4s = c15.resample("4h").last().dropna()
    c1d = c15.resample("1D").last().dropna()
    idx = c4s.index
    c4 = c4s.values

    base50, above150 = daily_state(c1d, idx, half_ma=50)
    base_core, _ = daily_state(c1d, idx, half_ma=None)
    up4 = (c4s.rolling(7).mean() > c4s.rolling(34).mean()).values

    ev4 = divergence_events(c4s, DivergenceConfig())
    ev4 = pd.Series(ev4.values, index=ev4.index + H4).reindex(idx + H4).fillna(0.0).values

    S = lambda x: pd.Series(x, index=idx)  # noqa: E731
    out = {
        "buy & hold": S(np.ones(len(idx))),
        "daily rule (now)": S(base50),
        "daily, no trim": S(base_core),
    }
    for n in (34, 90, 180):
        ma = c4s.rolling(n).mean().values
        out[f"daily + 4h trim MA{n}"] = S(trim_layer(base_core, c4, ma))
    out["daily rule + 4h 7/34 half"] = S(np.where(up4, base50, base50 * 0.5))
    early = np.where((base50 == 0) & (above150 > 0) & up4, 0.5, base50)
    out["daily rule + 4h early half"] = S(early)
    out["daily rule + 4h divergence"] = S(divergence_layer(base50, ev4))
    ma90 = c4s.rolling(90).mean().values
    combo = trim_layer(base_core, c4, ma90)
    combo = np.where((base_core == 0) & (above150 > 0) & up4, 0.5, combo)
    out["4h trim MA90 + early half"] = S(combo)
    return out, c4s


def equity(close: pd.Series, pos: pd.Series, cost: float) -> pd.Series:
    held = pos.shift(1).fillna(0.0)
    r = close.pct_change().fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    return (1 + held * r - turn * cost).cumprod()


def sell_stats(close: pd.Series, pos: pd.Series, cost: float) -> Dict[str, float]:
    """Actions per year and the share of sells made above the average cost."""
    p, c = pos.values, close.values
    qty, avg = 0.0, 0.0
    wins = sells = buys = 0
    for i in range(1, len(p)):
        d = p[i] - p[i - 1]
        if d > 1e-9:
            buys += 1
            avg = (avg * qty + c[i] * (1 + cost) * d) / (qty + d)
            qty += d
        elif d < -1e-9:
            sells += 1
            wins += c[i] * (1 - cost) > avg
            qty += d
            if qty < 1e-9:
                qty, avg = 0.0, 0.0
    years = (close.index[-1] - close.index[0]).days / 365
    return {"actions_per_year": (buys + sells) / years,
            "sell_win_rate": wins / sells if sells else np.nan}


def metrics(close: pd.Series, pos: pd.Series, cost: float) -> Dict[str, float]:
    eq = equity(close, pos, cost)
    years = (close.index[-1] - close.index[0]).days / 365
    out = {"cagr": eq.iloc[-1] ** (1 / years) - 1,
           "max_drawdown": (eq / eq.cummax() - 1).min()}
    out.update(sell_stats(close, pos, cost))
    return out


def study(c15: pd.Series, cost: float) -> pd.DataFrame:
    variants, c4 = build_variants(c15)
    start = c4.index[0] + pd.Timedelta(days=180)
    c = c4.loc[start:]
    half = c.index[len(c) // 2]
    rows = {}
    for name, pos in variants.items():
        p = pos.loc[start:]
        m = metrics(c, p, cost)
        m["cagr_first_half"] = metrics(c.loc[:half], p.loc[:half], cost)["cagr"]
        m["cagr_second_half"] = metrics(c.loc[half:], p.loc[half:], cost)["cagr"]
        rows[name] = m
    return pd.DataFrame(rows).T


def main():
    from src.data.crypto_data_fetcher import fetch_ohlcv_ccxt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--proxy", default=None, help="e.g. http://127.0.0.1:8800")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "crypto", "results"))
    args = ap.parse_args()
    warnings.simplefilter("ignore", FutureWarning)
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 250)
    for sym in args.symbols:
        c15 = fetch_ohlcv_ccxt(f"{sym}/USDT", "15m", args.start, exchange="okx", proxy=args.proxy)["close"]
        res = study(c15, args.cost)
        res.to_csv(os.path.join(args.out, f"{sym}_daily_4h_cost{args.cost:g}.csv"))
        print(f"\n{sym}: {c15.index[0]} -> {c15.index[-1]}, cost {args.cost:.2%} a side")
        print(res.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
