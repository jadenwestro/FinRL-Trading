"""
Ideas from the "10 strongest trading methods" video, tested on the daily rules
=============================================================================

zihao asked to take the strategies and indicators from a YouTube video
(Speculation Lab, "交易的神们 ... 最强的10个") and use them to improve the
current daily rules for BTC / ETH / SOL. The current rules stay the base
(see crypto_trend_signals.py, long or cash only):

  buy  : close > 150-day MA x 1.05          stop : close < 150-day MA x 0.95
  trim : close < 50-day MA sells half, close > 50-day MA x 1.02 buys it back
  exit : bearish 3-hump MACD(9/34)+RSI divergence sells all

Each video idea that can run on daily spot bars is added as one change:

  minervini filter : buy only while Minervini's trend template holds
                     (close and MA50 above MA150 and MA200, MA200 higher than
                     a month ago, >= 25% above the 52-week low, within 25% of
                     the 52-week high); the stop is unchanged.
  minervini entry  : buy as soon as the template holds, even before
                     close > MA150 x 1.05; the stop is unchanged.
  qullamaggie trim : replace the 50-day trim with EMA10 / EMA20: below EMA10
                     hold 2/3, below EMA20 hold 1/3, back above EMA10 full.
  macd 12/26/9     : the divergence exit uses the video's MACD(12/26/9)
                     three-segment histogram divergence, no RSI.
  quad stoch dip   : after a trim, buy the half back on the video's "bull
                     flag" (close above EMA50, low touches EMA20, stoch(9,3)
                     <= 20 and stoch(60,10) >= 85) instead of waiting for
                     close > MA50 x 1.02.
  crash buy        : after a trim, buy the half back after a one-day drop of
                     >= 10% while the 200-day MA is rising (Flight / "buy the
                     fast drop in a bull market").
  parabolic trim   : Alex's exhaustion: after >= 3 green days in a row adding
                     up to >= 20%, the first close below the previous close
                     sells half; it is bought back on a close above the peak.
  outside bar      : Larry Williams: a daily bar with a higher high and lower
                     low than the previous one that closes below the previous
                     low (body >= 2x the previous body) buys the idle cash at
                     the close; sold at the first profitable close, or at
                     previous low - 0.2 x ATR(14). Only while close > MA200.

Signals use a day's values after it closes; the position earns from the next
day. Costs are charged per unit of turnover (default 0.1% a side).

Data: OKX daily candles via ccxt (cached under data/crypto/), or
--source coinmetrics (daily closes only: high = low = close, so the
outside-bar and stochastic ideas are only approximate there).

    python src/strategies/crypto_video_ideas_study.py --proxy http://127.0.0.1:8800
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

from strategies.btc_ma_divergence_strategy import (  # noqa: E402
    DivergenceConfig, divergence_events, ema, ma_regime)
from strategies.crypto_daily_4h_study import metrics  # noqa: E402


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def stoch_k(df: pd.DataFrame, n: int, smooth: int) -> pd.Series:
    lo, hi = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    raw = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return raw.rolling(smooth).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def minervini_template(c: pd.Series) -> pd.Series:
    m50, m150, m200 = (c.rolling(n).mean() for n in (50, 150, 200))
    lo52, hi52 = c.rolling(365).min(), c.rolling(365).max()
    ok = ((c > m150) & (c > m200) & (m50 > m150) & (m50 > m200) & (m150 > m200)
          & (m200 > m200.shift(30)) & (c >= lo52 * 1.25) & (c >= hi52 * 0.75))
    return ok.fillna(False)


def quad_stoch_dip(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    e20, e50 = ema(c, 20), ema(c, 50)
    k9, k60 = stoch_k(df, 9, 3), stoch_k(df, 60, 10)
    return ((c > e50) & (df["low"] <= e20 * 1.01) & (k9 <= 20) & (k60 >= 85)).fillna(False)


def crash_day(c: pd.Series, drop: float = 0.10) -> pd.Series:
    m200 = c.rolling(200).mean()
    return ((c / c.shift(1) - 1 <= -drop) & (m200 > m200.shift(30))).fillna(False)


def parabolic_top(c: pd.Series, min_days: int = 3, min_gain: float = 0.20) -> pd.Series:
    """First down close after >= min_days up closes adding up to >= min_gain."""
    up = (c > c.shift(1)).values
    cv = c.values
    out = np.zeros(len(cv), dtype=bool)
    run = 0
    for t in range(1, len(cv)):
        if up[t]:
            run += 1
            continue
        if run >= min_days and cv[t - 1] / cv[t - 1 - run] - 1 >= min_gain:
            out[t] = True
        run = 0
    return pd.Series(out, index=c.index)


def outside_bar_long(df: pd.DataFrame, min_ratio: float = 2.0) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body, pbody = (c - o).abs(), (c - o).abs().shift(1)
    return ((h > h.shift(1)) & (l < l.shift(1)) & (c < l.shift(1)) & (body >= min_ratio * pbody)).fillna(False)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    entry: str = "base"           # base | filter | template
    trim: str = "ma50"            # ma50 | qull
    addback: str = "ma50"         # ma50 | stoch | crash  (extra ways to buy the half back)
    div_macd: tuple = (9, 34, 9)
    div_rsi: bool = True
    parabolic: bool = False


def run_variant(df: pd.DataFrame, v: Variant) -> pd.Series:
    c = df["close"]
    px = c.values
    ma150 = c.rolling(150).mean().values
    ma50 = c.rolling(50).mean().values
    reg = ma_regime(c, 150, 0.05).values
    tmpl = minervini_template(c).values
    e10, e20 = ema(c, 10).values, ema(c, 20).values
    f, s, g = v.div_macd
    ev = divergence_events(c, DivergenceConfig(macd_fast=f, macd_slow=s, macd_signal=g, use_rsi=v.div_rsi)
                           ).reindex(c.index).fillna(0.0).values
    extra_add = np.zeros(len(px), dtype=bool)
    if v.addback == "stoch":
        extra_add = quad_stoch_dip(df).values
    elif v.addback == "crash":
        extra_add = crash_day(c).values
    para = parabolic_top(c).values if v.parabolic else np.zeros(len(px), dtype=bool)

    pos = np.zeros(len(px))
    held, div_out, last = False, False, 0.0
    trimmed, para_peak = False, None
    for t in range(len(px)):
        if np.isnan(ma150[t]):
            continue
        g_t = reg[t]
        if g_t != last:
            last = g_t
            div_out, trimmed, para_peak = False, False, None
            if g_t < 0:
                held = False
        # entry
        if not held and not div_out:
            if v.entry == "base":
                go = g_t > 0
            elif v.entry == "filter":
                go = g_t > 0 and tmpl[t]
            else:  # template entry: also allowed before the +5% line, never below the stop
                go = (g_t > 0 or px[t] > ma150[t] * 0.95) and tmpl[t]
            if go:
                held, trimmed, para_peak = True, False, None
        # stop for template entries made inside the band
        if held and px[t] < ma150[t] * 0.95:
            held = False
        if held and ev[t] < 0:
            held, div_out = False, True
        if not held and div_out and ev[t] > 0 and g_t > 0:
            held, div_out, trimmed, para_peak = True, False, False, None
        if not held:
            pos[t] = 0.0
            continue

        size = 1.0
        if v.trim == "ma50":
            if not np.isnan(ma50[t]):
                if not trimmed and px[t] < ma50[t]:
                    trimmed = True
                elif trimmed and (px[t] > ma50[t] * 1.02 or extra_add[t]):
                    trimmed = False
            size = 0.5 if trimmed else 1.0
        else:  # qullamaggie ladder
            size = 1.0 if px[t] >= e10[t] else (2 / 3 if px[t] >= e20[t] else 1 / 3)
        if v.parabolic:
            if para_peak is None and para[t]:
                para_peak = px[t - 1]
            elif para_peak is not None and px[t] > para_peak:
                para_peak = None
            if para_peak is not None:
                size = min(size, 0.5)
        pos[t] = size
    return pd.Series(pos, index=c.index)


def outside_bar_overlay(df: pd.DataFrame, base: pd.Series, ratio: float = 2.0):
    """Use idle cash (1 - base) for Larry Williams outside-bar trades; returns (pos, trade returns)."""
    c, lo = df["close"].values, df["low"].values
    sig = outside_bar_long(df, ratio).values
    m200 = df["close"].rolling(200).mean().values
    a = atr(df).values
    b = base.values
    out = b.copy()
    trades = []
    in_tr, entry, stop = False, 0.0, 0.0
    for t in range(1, len(c)):
        if in_tr:
            if lo[t] <= stop or c[t] > entry:
                exit_px = stop if lo[t] <= stop else c[t]
                trades.append(exit_px / entry - 1)
                in_tr = False
            else:
                out[t] = b[t] + (1 - b[t])
                continue
        if sig[t] and not np.isnan(m200[t]) and c[t] > m200[t] and b[t] < 1:
            in_tr, entry, stop = True, c[t], lo[t - 1] - 0.2 * a[t]
            out[t] = 1.0
    return pd.Series(out, index=base.index), np.array(trades)


VARIANTS: Dict[str, Optional[Variant]] = {
    "buy & hold": None,
    "current rules": Variant(),
    "minervini filter": Variant(entry="filter"),
    "minervini entry": Variant(entry="template"),
    "qullamaggie trim": Variant(trim="qull"),
    "macd 12/26/9 exit": Variant(div_macd=(12, 26, 9), div_rsi=False),
    "quad stoch dip": Variant(addback="stoch"),
    "crash buy": Variant(addback="crash"),
    "parabolic trim": Variant(parabolic=True),
}


def study(df: pd.DataFrame, cost: float, test_start: pd.Timestamp) -> pd.DataFrame:
    c = df["close"]
    pos_map = {n: (pd.Series(1.0, index=c.index) if v is None else run_variant(df, v)) for n, v in VARIANTS.items()}
    ob_pos, ob_trades = outside_bar_overlay(df, pos_map["current rules"])
    pos_map["outside bar"] = ob_pos
    cc = c.loc[test_start:]
    half = cc.index[len(cc) // 2]
    rows = {}
    for name, pos in pos_map.items():
        p = pos.loc[test_start:]
        m = metrics(cc, p, cost)
        m["cagr_first_half"] = metrics(cc.loc[:half], p.loc[:half], cost)["cagr"]
        m["cagr_second_half"] = metrics(cc.loc[half:], p.loc[half:], cost)["cagr"]
        m["mdd_first_half"] = metrics(cc.loc[:half], p.loc[:half], cost)["max_drawdown"]
        m["mdd_second_half"] = metrics(cc.loc[half:], p.loc[half:], cost)["max_drawdown"]
        m["exposure"] = float(p.mean())
        rows[name] = m
    res = pd.DataFrame(rows).T
    res.attrs["outside_bar_trades"] = ob_trades
    res.attrs["half"] = half
    return res


def load(symbol: str, source: str, proxy: Optional[str], start: str) -> pd.DataFrame:
    if source == "okx":
        from src.data.crypto_data_fetcher import fetch_ohlcv_ccxt
        return fetch_ohlcv_ccxt(f"{symbol}/USDT", "1d", start, exchange="okx", proxy=proxy)
    from src.data.crypto_data_fetcher import fetch_crypto_daily
    c = fetch_crypto_daily(symbol, start=start, source=source)["close"]
    return pd.DataFrame({"open": c.shift(1).fillna(c), "high": c, "low": c, "close": c})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--source", default="okx", choices=["okx", "coinmetrics"])
    ap.add_argument("--proxy", default=None, help="e.g. http://127.0.0.1:8800")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "crypto", "results"))
    args = ap.parse_args()
    warnings.simplefilter("ignore", FutureWarning)
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 250)
    for sym in args.symbols:
        df = load(sym, args.source, args.proxy, args.start)
        test_start = df.index[0] + pd.Timedelta(days=180)
        res = study(df, args.cost, test_start)
        res.to_csv(os.path.join(args.out, f"{sym}_video_ideas_{args.source}.csv"))
        tr = res.attrs["outside_bar_trades"]
        print(f"\n{sym}: {df.index[0].date()} -> {df.index[-1].date()}, tested from {test_start.date()}, "
              f"halves split at {res.attrs['half'].date()}, cost {args.cost:.2%} a side")
        print(res.to_string(float_format=lambda v: f"{v:.3f}"))
        if len(tr):
            print(f"outside-bar trades: {len(tr)}, win rate {np.mean(tr > 0):.0%}, "
                  f"avg {tr.mean():+.2%}, avg win {tr[tr > 0].mean():+.2%}, "
                  f"avg loss {tr[tr <= 0].mean() if (tr <= 0).any() else 0:+.2%}")
        else:
            print("outside-bar trades: 0")


if __name__ == "__main__":
    main()
