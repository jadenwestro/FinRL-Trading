"""
Short-term MA(7/34) rules on 4h / 1h / 15m bars for BTC / ETH / SOL
===================================================================

zihao's short-term version of the trend rules:

  4 hour : direction.  MA7 above MA34 = up trend, below = down trend.
  1 hour : confirmation, same test on 1h bars.
  15 min : timing.  Enter when the 15m MA7 crosses above MA34 while the
           4h and 1h trends both point up (mirror image for shorts).

Every higher-timeframe bar is used only after it has closed: 1h and 4h
bars are built from the 15m bars and their values become known at
bar open + bar length. A position decided at a 15m close earns from the
next 15m bar. Costs are charged per unit of turnover (default 0.1% a side,
OKX taker fee).

Variants compared (long/cash unless marked "long/short"):
  4h only          : hold while the 4h MA7 > MA34
  4h + 1h          : hold while both 4h and 1h MA7 > MA34
  4h+1h, 15m entry : enter on a 15m golden cross with 4h and 1h up;
                     exit on a 1h death cross or a 4h death cross
  ... + half       : 1h death cross sells half, 4h death cross sells all,
                     15m golden cross with 1h up again buys the half back
  ... + daily trend: only trade long while the daily 150-day MA rule is long
  long/short       : the same with the mirror-image short side
plus buy & hold and the daily rule (150-day MA +-5%, 50-day trim) for scale.

Data: OKX 15m candles via ccxt, cached under data/crypto/.
    python src/strategies/crypto_short_term_ma_study.py --proxy http://127.0.0.1:8800
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

from strategies.btc_ma_divergence_strategy import ma_regime  # noqa: E402

BAR = pd.Timedelta(minutes=15)
BARS_PER_YEAR = 365 * 96


def ma(x: pd.Series, n: int, kind: str = "sma") -> pd.Series:
    return x.ewm(span=n, adjust=False).mean() if kind == "ema" else x.rolling(n).mean()


def trend_on_15m(close15: pd.Series, rule: str, fast: int, slow: int, kind: str) -> pd.Series:
    """+1 / -1 / 0 (MA7 vs MA34) of ``rule`` bars, as known at each 15m close.

    The returned series is indexed by 15m bar open time like ``close15``; the
    value at bar t is what was known when bar t closed.
    """
    c = close15.resample(rule).last().dropna() if rule != "15min" else close15
    f, s = ma(c, fast, kind), ma(c, slow, kind)
    sig = np.sign(f - s).where(s.notna(), 0.0)
    known_at = sig.index + pd.Timedelta(rule)            # higher bar is known when it closes
    close_times = close15.index + BAR
    out = pd.Series(sig.values, index=known_at).reindex(close_times, method="ffill").fillna(0.0)
    out.index = close15.index
    return out


def daily_trend_on_15m(close15: pd.Series, slow: int = 150, band: float = 0.05) -> pd.Series:
    d = close15.resample("1D").last().dropna()
    reg = ma_regime(d, slow, band)
    close_times = close15.index + BAR
    out = pd.Series(reg.values, index=reg.index + pd.Timedelta("1D")).reindex(close_times, method="ffill")
    out.index = close15.index
    return out.fillna(0.0)


def entry_exit_rules(t4: np.ndarray, t1: np.ndarray, t15: np.ndarray, half: bool = False,
                     allow_short: bool = False, gate: Optional[np.ndarray] = None) -> np.ndarray:
    """15m cross entries inside the 4h + 1h trend; exits on 1h / 4h crosses."""
    n = len(t4)
    pos = np.zeros(n)
    cur = 0.0
    prev15 = 0.0
    for i in range(n):
        cross = t15[i] if t15[i] != prev15 and t15[i] != 0 else 0.0
        prev15 = t15[i]
        side = np.sign(cur)
        # exits first
        if side != 0:
            if t4[i] != side:
                cur = 0.0
            elif t1[i] != side:
                cur = side * 0.5 if half else 0.0
            if gate is not None and side > 0 and gate[i] <= 0:
                cur = 0.0
        # entries / adds
        for d in ((1.0, -1.0) if allow_short else (1.0,)):
            if d > 0 and gate is not None and gate[i] <= 0:
                continue
            if cross == d and t4[i] == d and t1[i] == d and np.sign(cur) != -d and abs(cur) < 1.0:
                cur = d
        pos[i] = cur
    return pos


def equity(close: pd.Series, pos: pd.Series, cost: float, short_fee: float = 0.05) -> pd.Series:
    """Position set at close t earns bar t+1; cost per unit turnover; shorts pay a funding fee."""
    held = pos.shift(1).fillna(0.0)
    r = close.pct_change().fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    fee = held.clip(upper=0).abs() * short_fee / BARS_PER_YEAR
    return (1 + held * r - turn * cost - fee).cumprod()


def trade_stats(close: pd.Series, pos: pd.Series, cost: float) -> Dict[str, float]:
    """Round trips: from flat to a position until flat (or flipped) again."""
    p = pos.values
    c = close.values
    eq = equity(close, pos, cost).values
    trips, start = [], None
    for i in range(1, len(p)):
        if p[i - 1] == 0 and p[i] != 0:
            start = i
        elif start is not None and (p[i] == 0 or np.sign(p[i]) != np.sign(p[i - 1])):
            trips.append((eq[i] / eq[start] - 1, i - start))
            start = i if p[i] != 0 else None
    if not trips:
        return {"trades_per_year": 0.0, "win_rate": np.nan, "avg_trade": np.nan, "avg_hours": np.nan}
    years = (close.index[-1] - close.index[0]).days / 365
    pnl = np.array([t[0] for t in trips])
    return {"trades_per_year": len(trips) / years, "win_rate": float((pnl > 0).mean()),
            "avg_trade": float(pnl.mean()), "avg_hours": float(np.mean([t[1] for t in trips]) / 4)}


def metrics(close: pd.Series, pos: pd.Series, cost: float) -> Dict[str, float]:
    eq = equity(close, pos, cost)
    years = (close.index[-1] - close.index[0]).days / 365
    cagr = eq.iloc[-1] ** (1 / years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    daily = eq.resample("1D").last().pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else np.nan
    out = {"cagr": cagr, "max_drawdown": mdd, "sharpe": sharpe,
           "time_in_market": float((pos != 0).mean())}
    out.update(trade_stats(close, pos, cost))
    return out


def build_variants(close15: pd.Series, fast: int = 7, slow: int = 34, kind: str = "sma"
                   ) -> Dict[str, pd.Series]:
    t4 = trend_on_15m(close15, "4h", fast, slow, kind)
    t1 = trend_on_15m(close15, "1h", fast, slow, kind)
    t15 = trend_on_15m(close15, "15min", fast, slow, kind)
    gate = daily_trend_on_15m(close15)
    a4, a1, a15, g = t4.values, t1.values, t15.values, gate.values
    idx = close15.index
    S = lambda x: pd.Series(x, index=idx)  # noqa: E731
    return {
        "buy & hold": S(np.ones(len(idx))),
        "daily rule (150d MA, 50d trim)": _daily_rule_on_15m(close15),
        "4h only": S((a4 > 0).astype(float)),
        "4h + 1h": S(((a4 > 0) & (a1 > 0)).astype(float)),
        "4h+1h, 15m entry": S(entry_exit_rules(a4, a1, a15)),
        "4h+1h, 15m entry, half": S(entry_exit_rules(a4, a1, a15, half=True)),
        "4h+1h, 15m entry, daily trend": S(entry_exit_rules(a4, a1, a15, gate=g)),
        "4h+1h, 15m entry, half, daily trend": S(entry_exit_rules(a4, a1, a15, half=True, gate=g)),
        "4h only long/short": S(a4.astype(float)),
        "4h+1h, 15m entry long/short": S(entry_exit_rules(a4, a1, a15, allow_short=True)),
    }


def _daily_rule_on_15m(close15: pd.Series) -> pd.Series:
    from strategies.crypto_trend_signals import Rules, run_rules

    d = close15.resample("1D").last().dropna()
    pos, _ = run_rules(d, Rules(half_ma=50))
    close_times = close15.index + BAR
    out = pd.Series(pos.values, index=pos.index + pd.Timedelta("1D")).reindex(close_times, method="ffill")
    out.index = close15.index
    return out.fillna(0.0)


def study(close15: pd.Series, start: str, cost: float, fast: int, slow: int, kind: str) -> pd.DataFrame:
    variants = build_variants(close15, fast, slow, kind)
    c = close15.loc[start:]
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
    ap.add_argument("--fast", type=int, default=7)
    ap.add_argument("--slow", type=int, default=34)
    ap.add_argument("--kind", default="sma", choices=["sma", "ema"])
    ap.add_argument("--cost", type=float, default=0.001, help="per side, fraction of trade value")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "crypto", "results"))
    args = ap.parse_args()
    warnings.simplefilter("ignore", FutureWarning)
    os.makedirs(args.out, exist_ok=True)

    pd.set_option("display.width", 250)
    for sym in args.symbols:
        m15 = fetch_ohlcv_ccxt(f"{sym}/USDT", "15m", args.start, exchange="okx", proxy=args.proxy)
        close = m15["close"]
        # test period starts once the daily 150-day MA exists
        test_start = str((close.index[0] + pd.Timedelta(days=180)).date())
        res = study(close, test_start, args.cost, args.fast, args.slow, args.kind)
        res.to_csv(os.path.join(args.out, f"{sym}_short_term_{args.kind}{args.fast}_{args.slow}_cost{args.cost:g}.csv"))
        print(f"\n{sym}: 15m bars {close.index[0]} -> {close.index[-1]}, tested from {test_start}, "
              f"MA{args.fast}/{args.slow} {args.kind}, cost {args.cost:.2%} a side")
        print(res.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
