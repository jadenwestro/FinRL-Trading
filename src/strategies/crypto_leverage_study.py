"""
What if the daily trend rules were traded with leveraged perpetual longs?
=======================================================================

Same signals as crypto_trend_signals.py (150-day MA +/-5%, trim half below
the 50-day MA, three-hump divergence exit; long or cash only). Instead of
buying spot, each position is an isolated-margin long on a USDT perpetual:

  * on every signal change the position is closed and reopened: margin =
    target fraction x account equity, notional = leverage x margin
    (fees charged on the change in notional only);
  * funding is paid by the long every day on the notional (3 x 8h rate),
    taken from the position margin;
  * liquidation: the day's low touches the isolated liquidation price
    (entry, margin and maintenance-margin rate). The whole margin is lost.
    Because positions only change at the daily close, the daily low is an
    exact test (no intraday data needed);
  * after a liquidation the account waits for the next BUY signal
    (``--reenter next`` re-enters at the next close if the rules still say
    long).

Spot prices are used for the perpetual (basis ignored). OKX only serves the
last ~3 months of funding history, so the backtest uses an assumed flat rate
(default 0.01% per 8h, the exchange's base rate) plus a stress case; the
recent real average is printed for reference when OKX is reachable.

    python src/strategies/crypto_leverage_study.py --symbols BTC ETH SOL --proxy http://127.0.0.1:8800
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from strategies.btc_ma_divergence_strategy import (  # noqa: E402
    DivergenceConfig, backtest_positions, divergence_events)
from strategies.btc_momentum_strategy import crypto_metrics  # noqa: E402
from strategies.crypto_trend_signals import Rules, run_rules  # noqa: E402


def simulate_perp(bars: pd.DataFrame, pos: pd.Series, leverage: float, fee: float = 0.001,
                  funding_8h: float = 0.0001, mmr: float = 0.005, reenter: str = "signal",
                  stop_pct: Optional[float] = None, dd_limit: Optional[float] = None
                  ) -> Dict[str, object]:
    """Equity curve (marked at each close) of the rules traded as an isolated long perp.

    Drawdown guards (both wait for the next BUY signal afterwards, like a liquidation):
      stop_pct : a resting stop order ``stop_pct`` below the price the position was
                 (re)opened at; fills at the stop, or at the open if the day gaps below it
      dd_limit : close everything at the close once equity is ``dd_limit`` below its
                 peak since the position was opened
    """
    opn, low, close = bars["open"].values, bars["low"].values, bars["close"].values
    target = pos.reindex(bars.index).fillna(0.0).values
    cash, qty, margin, entry, cur = 1.0, 0.0, 0.0, 0.0, 0.0
    blocked = False
    stop_px, peak = 0.0, 0.0
    eq = np.empty(len(close))
    liqs: List[pd.Timestamp] = []
    stops = 0
    fees_paid = funding_paid = 0.0

    for t in range(len(close)):
        if qty > 0:
            f = qty * close[t - 1] * funding_8h * 3
            margin -= f
            funding_paid += f
            liq_px = (qty * entry - margin) / (qty * (1 - mmr))
            hit_stop = stop_pct is not None and low[t] <= stop_px and max(stop_px, liq_px) == stop_px \
                and opn[t] > liq_px
            if hit_stop:
                px = min(opn[t], stop_px)
                cost = fee * qty * px
                fees_paid += cost
                cash += margin + qty * (px - entry) - cost
                qty, margin, cur, blocked = 0.0, 0.0, 0.0, True
                stops += 1
            elif low[t] <= liq_px:
                liqs.append(bars.index[t])
                qty, margin, cur = 0.0, 0.0, 0.0
                blocked = True
        equity = cash + margin + qty * (close[t] - entry)
        if qty > 0 and dd_limit is not None:
            peak = max(peak, equity)
            if equity < peak * (1 - dd_limit):
                cost = fee * qty * close[t]
                fees_paid += cost
                equity -= cost
                cash, qty, margin, cur, blocked = equity, 0.0, 0.0, 0.0, True
                stops += 1
        want = target[t]
        if blocked:
            prev = target[t - 1] if t > 0 else 0.0
            if want > 0 and (reenter == "next" or want > prev):
                blocked = False
            else:
                want = 0.0
        if abs(want - cur) > 1e-12 and equity > 0:
            old_notional = qty * close[t]
            new_notional = leverage * want * equity
            cost = fee * abs(new_notional - old_notional)
            fees_paid += cost
            equity -= cost
            margin = want * equity
            qty = leverage * margin / close[t]
            if cur == 0.0:
                peak = equity
            entry, cash, cur = close[t], equity - margin, want
            stop_px = close[t] * (1 - stop_pct) if stop_pct is not None else 0.0
        eq[t] = equity

    return {"equity": pd.Series(eq, index=bars.index), "liquidations": liqs, "stops": stops,
            "fees": fees_paid, "funding": funding_paid}


def load_bars(symbol: str, proxy: Optional[str], start: str) -> pd.DataFrame:
    from src.data.crypto_data_fetcher import fetch_ohlcv_ccxt

    return fetch_ohlcv_ccxt(f"{symbol}/USDT", "1d", start, exchange="okx", proxy=proxy)


def recent_funding(symbol: str, proxy: Optional[str]) -> Optional[float]:
    """Average 8h funding rate over what OKX still serves (~3 months), or None."""
    try:
        import ccxt

        opts = {"enableRateLimit": True}
        if proxy:
            opts["proxies"] = {"http": proxy, "https": proxy}
        rows = ccxt.okx(opts).fetch_funding_rate_history(f"{symbol}/USDT:USDT", limit=100)
        return float(np.mean([r["fundingRate"] for r in rows])) if rows else None
    except Exception:
        return None


def protect_study(args, rules: Rules) -> None:
    """Drawdown guards on spot and leveraged longs, per coin and for the 3-coin account."""
    guards = {"none": {}, "stop 8%": {"stop_pct": 0.08}, "stop 12%": {"stop_pct": 0.12},
              "dd 15%": {"dd_limit": 0.15}, "dd 25%": {"dd_limit": 0.25}}
    lev_guards = {"none": {}, "stop 3%": {"stop_pct": 0.03}, "stop 5%": {"stop_pct": 0.05},
                  "stop 8%": {"stop_pct": 0.08}}
    rows, curves = [], {}
    for sym in args.symbols:
        bars = load_bars(sym, args.proxy, args.start)
        close = bars["close"]
        pos, _ = run_rules(close, rules, divergence_events(close, DivergenceConfig()))
        test_start = close.index[0] + pd.Timedelta(days=rules.slow + 30)
        b, p = bars.loc[test_start:], pos.loc[test_start:]
        for lev in [1] + [x for x in args.leverage if x > 1]:
            for gname, g in (guards if lev == 1 else lev_guards).items():
                if lev > 1 and g.get("stop_pct", 0) * lev >= 0.95:
                    continue
                r = simulate_perp(b, p, lev, args.fee, 0.0 if lev == 1 else args.funding, args.mmr, **g)
                m = crypto_metrics(r["equity"].clip(lower=1e-9))
                rows.append({"symbol": sym, "lev": "spot" if lev == 1 else f"{lev:g}x", "guard": gname,
                             "cagr": m["cagr"], "max_dd": m["max_drawdown"], "stops": r["stops"],
                             "liqs": len(r["liquidations"])})
                if lev == 1:
                    curves[(sym, gname)] = r["equity"]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "protect_results.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, formatters={"cagr": "{:.1%}".format, "max_dd": "{:.1%}".format}))

    # whole account (spot, no guard): coin mix and a cash reserve, rebalanced daily
    rets = pd.DataFrame({s: curves[(s, "none")].pct_change() for s in args.symbols}).dropna()
    mixes = {"1/3 each": {"BTC": 1 / 3, "ETH": 1 / 3, "SOL": 1 / 3},
             "BTC 50 ETH 30 SOL 20": {"BTC": .5, "ETH": .3, "SOL": .2},
             "BTC 60 ETH 40": {"BTC": .6, "ETH": .4, "SOL": 0},
             "BTC only": {"BTC": 1, "ETH": 0, "SOL": 0}}
    acc = []
    for name, w in mixes.items():
        for reserve in (0.0, 0.3):
            r = (rets * pd.Series(w)).sum(axis=1) * (1 - reserve)
            m = crypto_metrics((1 + r).cumprod())
            acc.append({"mix": name, "cash reserve": f"{reserve:.0%}", "from": r.index[0].date(),
                        "cagr": m["cagr"], "max_dd": m["max_drawdown"]})
    print("\nWhole account (spot, same rules on each coin):")
    print(pd.DataFrame(acc).to_string(index=False, formatters={"cagr": "{:.1%}".format,
                                                                "max_dd": "{:.1%}".format}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--leverage", nargs="+", type=float, default=[1, 2, 3, 5, 10])
    ap.add_argument("--fee", type=float, default=0.001, help="per unit notional traded (taker + slippage)")
    ap.add_argument("--funding", type=float, default=0.0001, help="assumed 8h funding rate paid by longs")
    ap.add_argument("--stress-funding", type=float, default=0.0003)
    ap.add_argument("--mmr", type=float, default=0.005, help="maintenance margin rate")
    ap.add_argument("--protect", action="store_true", help="run the drawdown-guard study instead")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "crypto", "leverage"))
    args = ap.parse_args()
    warnings.simplefilter("ignore", FutureWarning)
    os.makedirs(args.out, exist_ok=True)

    rules = Rules(150, 0.05, half_ma=50, half_band=0.02)
    if args.protect:
        return protect_study(args, rules)
    rows, liq_rows = [], []
    for sym in args.symbols:
        bars = load_bars(sym, args.proxy, args.start)
        close = bars["close"]
        pos, _ = run_rules(close, rules, divergence_events(close, DivergenceConfig()))
        test_start = close.index[0] + pd.Timedelta(days=rules.slow + 30)
        b, p = bars.loc[test_start:], pos.loc[test_start:]
        spot = crypto_metrics(backtest_positions(b["close"], p, args.fee))
        rows.append({"symbol": sym, "case": "spot", "from": b.index[0].date(), "cagr": spot["cagr"],
                     "max_dd": spot["max_drawdown"], "x_money": spot["total_return"] + 1, "liqs": 0})
        cases = [(lev, args.funding, "signal") for lev in args.leverage]
        cases += [(lev, args.stress_funding, "signal") for lev in args.leverage if lev > 1]
        cases += [(lev, args.funding, "next") for lev in args.leverage if lev >= 5]
        for lev, fund, re in cases:
            r = simulate_perp(b, p, lev, args.fee, fund, args.mmr, re)
            m = crypto_metrics(r["equity"].clip(lower=1e-9))
            name = f"{lev:g}x funding {fund:.2%}/8h" + (" re-enter next day" if re == "next" else "")
            rows.append({"symbol": sym, "case": name, "from": b.index[0].date(), "cagr": m["cagr"],
                         "max_dd": m["max_drawdown"], "x_money": r["equity"].iloc[-1],
                         "liqs": len(r["liquidations"])})
            if fund == args.funding and re == "signal":
                for d in r["liquidations"]:
                    liq_rows.append({"symbol": sym, "leverage": lev, "date": d.date()})
        rf = recent_funding(sym, args.proxy)
        if rf is not None:
            print(f"{sym}: OKX average funding over the last ~100 periods = {rf:.4%} per 8h")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "leverage_results.csv"), index=False)
    pd.DataFrame(liq_rows).to_csv(os.path.join(args.out, "liquidations.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, formatters={"cagr": "{:.1%}".format, "max_dd": "{:.1%}".format,
                                                 "x_money": "{:,.2f}".format}))
    if liq_rows:
        print("\nLiquidation dates (base funding):")
        print(pd.DataFrame(liq_rows).groupby(["symbol", "leverage"])["date"]
              .apply(lambda s: ", ".join(str(x) for x in s)).to_string())
    print(f"\nResults in {args.out}")


if __name__ == "__main__":
    main()
