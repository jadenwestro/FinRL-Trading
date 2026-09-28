"""
Three-layer (macro / fund flow / market structure) overlays on the daily rules
==============================================================================

Tests whether the weekly "three-layer" macro framework improves the daily
long/cash trend rules that the daily reminder runs (150-day MA +/-5%, sell
half below the 50-day MA, three-hump divergence exit; crypto_trend_signals.py).

Each overlay can only *reduce* the position the base rules want (it caps the
position at 0 or 0.5 while its condition holds), so the base rules still decide
when to buy and sell. Everything is causal: macro data are lagged one extra
day (US data for day t is only published after the crypto daily close).

  L1 macro     10-year Treasury yield (FRED DGS10)
               y10_up20   : 10Y rose > 0.25 pp over 20 trading days -> cap 0.5
               y10_above_ma: 10Y above its 100-day MA              -> cap 0.5
               y10_ge_475 : 10Y >= 4.75%                           -> cap 0.5
  L2 flows     US spot BTC ETF daily net flows (Farside, from 2024-01-11)
               etf_4w_out : 20-day sum of flows < 0                 -> cap 0.5
               y10_up_and_etf_out: both 10Y rising and ETF outflow -> cap 0.5
               crypto Fear & Greed index (alternative.me, from 2018-02)
               fng_ge_80  : 7-day average >= 80 (extreme greed)    -> cap 0.5
  L3 structure 50-week MA (350-day SMA of the daily close)
               below_50w  : close < 50-week MA                      -> cap 0

ETH and SOL use the BTC ETF flow series (their own funds are younger still).
ETF overlays only matter from 2024; results for them are reported on that
window separately and are a small sample.

Data are cached under data/macro/. On a network that needs a proxy:

    python src/strategies/macro_overlay_study.py --proxy http://127.0.0.1:8800

Optional: --etf-csv path/to/flows.csv (columns date,flow in US$m) when
Farside cannot be read.
"""

from __future__ import annotations

import argparse
import io
import json
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

from strategies.btc_ma_divergence_strategy import (  # noqa: E402
    DivergenceConfig, backtest_positions, divergence_events)
from strategies.btc_momentum_strategy import crypto_metrics  # noqa: E402
from strategies.crypto_trend_signals import Rules, load_daily, run_rules  # noqa: E402

MACRO_DIR = os.path.join(PROJECT_ROOT, "data", "macro")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def _get(url: str, proxy: Optional[str]) -> bytes:
    import requests
    proxies = {"http": proxy, "https": proxy} if proxy else None
    r = requests.get(url, headers=UA, proxies=proxies, timeout=60)
    r.raise_for_status()
    return r.content


def _cached(name: str, fetch, refresh: bool) -> pd.Series:
    path = os.path.join(MACRO_DIR, f"{name}.csv")
    if not refresh and os.path.exists(path) and \
            pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(path), unit="s") < pd.Timedelta(hours=12):
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s = fetch()
    os.makedirs(MACRO_DIR, exist_ok=True)
    s.to_csv(path)
    return s


def load_fred(series_id: str, proxy: Optional[str], refresh: bool = False) -> pd.Series:
    def fetch():
        raw = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=2015-01-01", proxy)
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = ["date", series_id]
        df["date"] = pd.to_datetime(df["date"])
        return pd.to_numeric(df.set_index("date")[series_id], errors="coerce").dropna()
    return _cached(f"fred_{series_id}", fetch, refresh)


def load_fear_greed(proxy: Optional[str], refresh: bool = False) -> pd.Series:
    def fetch():
        data = json.loads(_get("https://api.alternative.me/fng/?limit=0&format=json", proxy))["data"]
        s = pd.Series({pd.Timestamp(int(d["timestamp"]), unit="s").normalize(): float(d["value"])
                       for d in data}).sort_index()
        s.name = "fear_greed"
        return s
    return _cached("fear_greed", fetch, refresh)


def load_btc_etf_flows(proxy: Optional[str], refresh: bool = False,
                       csv: Optional[str] = None) -> pd.Series:
    """Daily total net flow of US spot BTC ETFs in US$m (Farside 'Total' column)."""
    if csv:
        df = pd.read_csv(csv, parse_dates=["date"])
        return df.set_index("date")["flow"].sort_index()

    def fetch():
        html = _get("https://farside.co.uk/bitcoin-etf-flow-all-data/", proxy).decode("utf-8", "ignore")
        best = None
        for t in pd.read_html(io.StringIO(html)):
            t.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
            tot = [c for c in t.columns if "Total" in c]
            if tot and (best is None or len(t) > len(best[0])):
                best = (t, tot[-1])
        if best is None:
            raise RuntimeError("no Total column found on Farside page")
        t, col = best
        dates = pd.to_datetime(t.iloc[:, 0], errors="coerce", format="mixed", dayfirst=True)
        vals = (t[col].astype(str).str.replace(",", "").str.replace("(", "-").str.replace(")", "")
                .str.replace("-$", "", regex=True))
        s = pd.Series(pd.to_numeric(vals, errors="coerce").values, index=dates)
        s = s[s.index.notna()].dropna().sort_index()
        s = s[~s.index.duplicated()]
        s.name = "btc_etf_flow_musd"
        return s
    return _cached("btc_etf_flows", fetch, refresh)


def to_daily(s: pd.Series, index: pd.DatetimeIndex, lag: int = 1) -> pd.Series:
    """Align a (business-day) series to the crypto calendar, forward-filled and
    lagged ``lag`` calendar days so day t only uses data published by then."""
    s = s[~s.index.duplicated()].sort_index()
    return s.reindex(s.index.union(index)).ffill().shift(lag, freq="D").reindex(index)


def overlay_caps(close: pd.Series, y10: Optional[pd.Series], fng: Optional[pd.Series],
                 etf: Optional[pd.Series]) -> Dict[str, pd.Series]:
    """Max allowed position per day for each overlay (1 = no limit)."""
    idx = close.index
    one = pd.Series(1.0, index=idx)
    caps: Dict[str, pd.Series] = {}
    if y10 is not None:
        y = y10.dropna()
        y_d = to_daily(y, idx)
        up20 = to_daily(y.diff(20), idx)
        above = to_daily(y - y.rolling(100).mean(), idx)
        caps["y10_up20"] = one.where(~(up20 > 0.25), 0.5)
        caps["y10_above_ma"] = one.where(~(above > 0), 0.5)
        caps["y10_ge_475"] = one.where(~(y_d >= 4.75), 0.5)
    if fng is not None:
        f7 = fng.rolling(7, min_periods=3).mean()
        caps["fng_ge_80"] = one.where(~(to_daily(f7, idx, lag=0) >= 80), 0.5)
    if etf is not None:
        e = etf.dropna()
        s20 = to_daily(e.rolling(20).sum(), idx)
        out = s20 < 0
        caps["etf_4w_out"] = one.where(~out, 0.5)
        if y10 is not None:
            up = to_daily(y10.dropna().diff(20), idx) > 0.25
            caps["y10_up_and_etf_out"] = one.where(~(up & out), 0.5)
    w50 = close.rolling(350).mean()
    caps["below_50w"] = one.where(~(close < w50), 0.0)
    return caps


def stats(close: pd.Series, pos: pd.Series, start, end, cost: float) -> dict:
    c = close.loc[start:end]
    p = pos.reindex(c.index).fillna(0.0)
    m = crypto_metrics(backtest_positions(c, p, cost))
    return {"cagr": m["cagr"], "mdd": m["max_drawdown"], "sharpe": m["sharpe"],
            "trades_per_yr": float((p.diff().fillna(0) != 0).sum()) / max((c.index[-1] - c.index[0]).days / 365, 1e-9),
            "exposure": float(p.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--source", default="okx")
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--etf-csv", default=None)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "macro", "overlay_results.csv"))
    args = ap.parse_args()
    warnings.simplefilter("ignore", FutureWarning)

    def attempt(name, fn):
        try:
            s = fn()
            print(f"{name}: {len(s)} rows, {s.index[0].date()} .. {s.index[-1].date()}, last {s.iloc[-1]:.2f}")
            return s
        except Exception as e:  # keep going with the layers we can get
            print(f"{name}: NOT AVAILABLE ({type(e).__name__}: {e})")
            return None

    y10 = attempt("10Y yield (FRED DGS10)", lambda: load_fred("DGS10", args.proxy, args.refresh))
    fng = attempt("Fear & Greed", lambda: load_fear_greed(args.proxy, args.refresh))
    etf = attempt("BTC ETF flows", lambda: load_btc_etf_flows(args.proxy, args.refresh, args.etf_csv))

    rules = Rules(slow=150, band=0.05, half_ma=50, half_band=0.02)
    rows, port = [], {}
    for sym in args.symbols:
        close = load_daily(sym, args.source, args.proxy, args.start)
        base = run_rules(close, rules, divergence_events(close, DivergenceConfig()))[0]
        t0 = close.index[0] + pd.Timedelta(days=rules.slow + 30)
        mid = t0 + (close.index[-1] - t0) / 2
        windows = {"full": (t0, None), "1st half": (t0, mid), "2nd half": (mid, None),
                   "ETF era 2024+": (pd.Timestamp("2024-01-11"), None)}
        variants = {"base": base}
        for name, cap in overlay_caps(close, y10, fng, etf).items():
            variants[name] = np.minimum(base, cap.fillna(1.0))
        for vname, pos in variants.items():
            port.setdefault(vname, {})[sym] = (close, pos)
            for wname, (a, b) in windows.items():
                rows.append({"symbol": sym, "variant": vname, "window": wname,
                             **stats(close, pos, a, b, args.cost)})

    # equal-weight 1/3 each, rebalanced daily, on the common window
    for vname, legs in port.items():
        rets = []
        for sym, (close, pos) in legs.items():
            eq = backtest_positions(close, pos.reindex(close.index).fillna(0.0), args.cost)
            rets.append(eq.pct_change().rename(sym))
        r = pd.concat(rets, axis=1).dropna()
        r = r.loc[r.index[0] + pd.Timedelta(days=180):]
        eq = (1 + r.mean(axis=1)).cumprod()
        mid = eq.index[0] + (eq.index[-1] - eq.index[0]) / 2
        for wname, (a, b) in {"full": (None, None), "1st half": (None, mid), "2nd half": (mid, None),
                              "ETF era 2024+": (pd.Timestamp("2024-01-11"), None)}.items():
            m = crypto_metrics(eq.loc[a:b] / eq.loc[a:b].iloc[0])
            rows.append({"symbol": "PORT(1/3 each)", "variant": vname, "window": wname,
                         "cagr": m["cagr"], "mdd": m["max_drawdown"], "sharpe": m["sharpe"],
                         "trades_per_yr": np.nan, "exposure": np.nan})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    for w in ["full", "1st half", "2nd half", "ETF era 2024+"]:
        t = df[df.window == w].pivot_table(index="variant", columns="symbol", values=["cagr", "mdd"])
        print(f"\n=== {w}: CAGR / max drawdown ===")
        print(t.to_string(float_format=lambda v: f"{v:6.1%}"))
    print(f"\nFull table: {args.out}")


if __name__ == "__main__":
    main()
