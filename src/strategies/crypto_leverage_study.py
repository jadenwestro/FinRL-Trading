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
                  funding_8h: float = 0.0001, mmr: float = 0.005, reenter: str = "signal"
                  ) -> Dict[str, object]:
    """Equity curve (marked at each close) of the rules traded as an isolated long perp."""
    low, close = bars["low"].values, bars["close"].values
    target = pos.reindex(bars.index).fillna(0.0).values
    cash, qty, margin, entry, cur = 1.0, 0.0, 0.0, 0.0, 0.0
    blocked = False
    eq = np.empty(len(close))
    liqs: List[pd.Timestamp] = []
    fees_paid = funding_paid = 0.0

    for t in range(len(close)):
        if qty > 0:
            f = qty * close[t - 1] * funding_8h * 3
            margin -= f
            funding_paid += f
            liq_px = (qty * entry - margin) / (qty * (1 - mmr))
            if low[t] <= liq_px:
                liqs.append(bars.index[t])
                qty, margin, cur = 0.0, 0.0, 0.0
                blocked = True
        equity = cash + margin + qty * (close[t] - entry)
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
            entry, cash, cur = close[t], equity - margin, want
        eq[t] = equity

    return {"equity": pd.Series(eq, index=bars.index), "liquidations": liqs,
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
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "crypto", "leverage"))
    args = ap.parse_args()
    warnings.simplefilter("ignore", FutureWarning)
    os.makedirs(args.out, exist_ok=True)

    rules = Rules(150, 0.05, half_ma=50, half_band=0.02)
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
