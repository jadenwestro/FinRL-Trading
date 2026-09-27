"""
Paper trading ledger for the daily BTC / ETH / SOL trend rules
==============================================================

A simulated spot account that follows crypto_trend_signals.py exactly.
Each coin gets its own sleeve (by default 1/3 of the starting cash) and
holds the rule's target fraction of that sleeve: 0, 1/2 or all. Trades
fill at the daily close with a fee on the traded value. Nothing is sent to
an exchange.

Files in --book (created on the first run):
  state.json   cash and coin quantity per sleeve, last processed date
  trades.csv   every simulated buy / sell
  daily.csv    one row per day: price, target, coins, value per sleeve and total

Each run takes the latest daily close and the rule's target position for
each coin, trades the sleeves that changed and appends the day. A date
that was already booked is skipped, so re-running the same day is safe.

    python src/strategies/paper_trader.py --book /mnt/project-files/btc-strategy/paper-trading \\
        --date 2026-09-27 --close BTC=84439.9 ETH=2696.15 SOL=121.41 \\
        --target BTC=1 ETH=1 SOL=1 --reason BTC="trend up" ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

FIELDS_TRADES = ["date", "symbol", "side", "price", "qty", "value", "fee", "target_after", "reason"]


def _kv(items: Optional[List[str]], cast=float) -> Dict[str, object]:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.upper()] = cast(v)
    return out


def load_state(book: str, symbols: List[str], start_cash: float) -> dict:
    path = os.path.join(book, "state.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    per = start_cash / len(symbols)
    return {"start_cash": start_cash, "last_date": None,
            "sleeves": {s: {"cash": per, "qty": 0.0, "target": 0.0} for s in symbols}}


def step(state: dict, date: str, close: Dict[str, float], target: Dict[str, float],
         reason: Dict[str, str], fee: float) -> List[dict]:
    """Move each sleeve to its target fraction at today's close; return the trades."""
    trades = []
    for sym, sl in state["sleeves"].items():
        if sym not in close:
            continue
        px, tgt = close[sym], float(target.get(sym, sl["target"]))
        equity = sl["cash"] + sl["qty"] * px
        want_value = tgt * equity
        diff = want_value - sl["qty"] * px
        if abs(diff) > 1e-6 * max(equity, 1.0) and tgt != sl["target"]:
            if diff > 0:                       # buy: fee comes out of the cash spent
                spend = min(diff, sl["cash"])
                qty = spend * (1 - fee) / px
                sl["cash"] -= spend
                sl["qty"] += qty
                trades.append({"date": date, "symbol": sym, "side": "BUY", "price": px, "qty": qty,
                               "value": spend, "fee": spend * fee, "target_after": tgt,
                               "reason": reason.get(sym, "")})
            else:
                qty = sl["qty"] if tgt == 0 else min(sl["qty"], -diff / px)
                gross = qty * px
                sl["qty"] -= qty
                sl["cash"] += gross * (1 - fee)
                trades.append({"date": date, "symbol": sym, "side": "SELL", "price": px, "qty": qty,
                               "value": gross, "fee": gross * fee, "target_after": tgt,
                               "reason": reason.get(sym, "")})
        sl["target"] = tgt
    state["last_date"] = date
    return trades


def summary(state: dict, close: Dict[str, float]) -> dict:
    rows, total = {}, 0.0
    for sym, sl in state["sleeves"].items():
        px = close.get(sym)
        value = sl["cash"] + sl["qty"] * (px or 0.0)
        total += value
        rows[sym] = {"price": px, "target": sl["target"], "qty": sl["qty"], "cash": sl["cash"], "value": value}
    return {"sleeves": rows, "total": total, "pnl": total - state["start_cash"],
            "pnl_pct": total / state["start_cash"] - 1}


def _append_csv(path: str, rows: List[dict], fields: List[str]):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", required=True)
    ap.add_argument("--date", required=True, help="the daily close being booked, YYYY-MM-DD (UTC)")
    ap.add_argument("--close", nargs="+", required=True, help="SYM=price")
    ap.add_argument("--target", nargs="+", required=True, help="SYM=0|0.5|1")
    ap.add_argument("--reason", nargs="*", default=[], help='SYM="text" for the trade log')
    ap.add_argument("--start-cash", type=float, default=10_000.0)
    ap.add_argument("--fee", type=float, default=0.001)
    args = ap.parse_args()

    close = _kv(args.close)
    target = _kv(args.target)
    reason = _kv(args.reason, str)
    os.makedirs(args.book, exist_ok=True)
    state = load_state(args.book, sorted(close), args.start_cash)

    if state["last_date"] is not None and args.date <= state["last_date"]:
        print(f"{args.date} already booked (last {state['last_date']}); nothing done")
        trades = []
    else:
        trades = step(state, args.date, close, target, reason, args.fee)
        with open(os.path.join(args.book, "state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        _append_csv(os.path.join(args.book, "trades.csv"), trades, FIELDS_TRADES)
        s = summary(state, close)
        day = {"date": args.date, "total": round(s["total"], 2), "pnl_pct": round(s["pnl_pct"], 4)}
        for sym, r in s["sleeves"].items():
            day.update({f"{sym}_price": r["price"], f"{sym}_target": r["target"],
                        f"{sym}_qty": round(r["qty"], 8), f"{sym}_value": round(r["value"], 2)})
        _append_csv(os.path.join(args.book, "daily.csv"), [day], list(day))

    s = summary(state, close)
    print(f"Paper account after {state['last_date']} close:")
    for t in trades:
        print(f"  TRADE {t['symbol']} {t['side']} {t['qty']:.6f} @ {t['price']:,.2f} "
              f"(value {t['value']:,.2f}, fee {t['fee']:.2f}) -> target {t['target_after']:g}  {t['reason']}")
    for sym, r in s["sleeves"].items():
        print(f"  {sym}: target {r['target']:g}, {r['qty']:.6f} coins, cash {r['cash']:,.2f}, "
              f"value {r['value']:,.2f}")
    print(f"  TOTAL {s['total']:,.2f} USDT, P&L {s['pnl']:+,.2f} ({s['pnl_pct']:+.2%}) "
          f"vs start {state['start_cash']:,.2f}")


if __name__ == "__main__":
    main()
