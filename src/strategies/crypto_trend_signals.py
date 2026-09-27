"""
Daily trend signals, trade log and charts for BTC / ETH / SOL
=============================================================

Rules (long or cash, never short) — the version that held up best in the
2015-2026 study (see btc_ma_divergence_strategy.py):

  Buy      : close > 150-day MA x 1.05
  Stop     : close < 150-day MA x 0.95  -> sell everything
             (the stop line moves up with the MA)
  Take profit (optional, --tp): sell a fraction once the gain from the
             entry price reaches a level, e.g. "0.3:0.33" = sell 1/3 at +30%
  Divergence: bearish MACD(9/34)+RSI divergence over >= 3 histogram humps
             -> sell everything; bullish divergence -> buy back to full

Outputs per symbol (under --out):
  <SYM>_trades.csv   every buy / sell with price, reason, stop and target
  <SYM>_chart.png    price, 150-day MA, stop line and buy/sell markers
and prints today's status (hold or cash, stop price, next target).

Data: OKX daily candles via ccxt (use --proxy on a network that needs it),
or --source coinmetrics (daily closes, lags a few months).

    python src/strategies/crypto_trend_signals.py --symbols BTC ETH SOL --tp 0.3:0.33
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from strategies.btc_ma_divergence_strategy import (  # noqa: E402
    DivergenceConfig, backtest_positions, divergence_events, ma_regime)
from strategies.btc_momentum_strategy import crypto_metrics  # noqa: E402


@dataclass
class Rules:
    slow: int = 150
    band: float = 0.05
    tp: Tuple[Tuple[float, float], ...] = ()   # ((gain, fraction sold), ...)
    use_divergence: bool = True


def run_rules(close: pd.Series, rules: Rules, events: Optional[pd.Series] = None
              ) -> Tuple[pd.Series, pd.DataFrame]:
    """Daily target position in [0, 1] plus a log of every position change."""
    ma = close.rolling(rules.slow).mean()
    reg = ma_regime(close, rules.slow, rules.band).values
    if events is None:
        events = divergence_events(close, DivergenceConfig()) if rules.use_divergence \
            else pd.Series(dtype=float)
    ev = events.reindex(close.index).fillna(0.0).values
    px, mav, idx = close.values, ma.values, close.index

    pos = np.zeros(len(px))
    cur, last, entry, taken = 0.0, 0.0, None, set()
    log: List[dict] = []

    def record(t, new, reason):
        nonlocal cur
        if abs(new - cur) < 1e-12:
            return
        stop = mav[t] * (1 - rules.band) if not np.isnan(mav[t]) else np.nan
        nxt = [entry * (1 + g) for g, _ in rules.tp if g not in taken] if entry else []
        log.append({"date": idx[t], "action": "BUY" if new > cur else "SELL",
                    "price": px[t], "position_after": new, "reason": reason,
                    "entry_price": entry, "stop_price": stop,
                    "next_target": min(nxt) if nxt else np.nan})
        cur = new

    for t in range(len(px)):
        g = reg[t]
        if g != last:
            last = g
            if g > 0:
                entry, taken = px[t], set()
                record(t, 1.0, "trend up: close > MA x (1+band)")
            else:
                record(t, 0.0, "stop: close < MA x (1-band)")
                entry = None
        if g > 0:
            if ev[t] < 0:
                record(t, 0.0, "bearish divergence")
            elif ev[t] > 0 and cur < 1.0:
                if cur == 0.0:
                    entry, taken = px[t], set()
                record(t, 1.0, "bullish divergence")
            if entry and cur > 0:
                for gain, frac in rules.tp:
                    if gain not in taken and px[t] >= entry * (1 + gain):
                        taken.add(gain)
                        record(t, max(0.0, cur - frac), f"take profit +{gain:.0%}")
        pos[t] = cur

    return pd.Series(pos, index=idx), pd.DataFrame(log)


def status_today(close: pd.Series, pos: pd.Series, log: pd.DataFrame, rules: Rules) -> dict:
    ma = close.rolling(rules.slow).mean()
    last = close.index[-1]
    out = {"date": last.date(), "close": close.iloc[-1], "position": pos.iloc[-1],
           "ma": ma.iloc[-1], "buy_above": ma.iloc[-1] * (1 + rules.band),
           "stop_below": ma.iloc[-1] * (1 - rules.band)}
    if not log.empty:
        lt = log.iloc[-1]
        out.update(last_action=f"{lt.action} {lt.date.date()} @ {lt.price:,.2f} ({lt.reason})",
                   entry_price=lt.entry_price, next_target=lt.next_target)
    return out


def plot(symbol: str, close: pd.Series, log: pd.DataFrame, rules: Rules, since: str, path: str,
         status: Optional[dict] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    # CJK-capable fonts: Windows, macOS, Linux
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Zen Hei",
                                       "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    c = close.loc[since:]
    ma = close.rolling(rules.slow).mean().loc[since:]
    lg = log[log.date >= pd.Timestamp(since)] if not log.empty else log

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(c.index, c, color="#52514e", lw=1.2, label=f"{symbol} 日收盘价")
    ax.plot(ma.index, ma, color="#2a78d6", lw=2, label=f"{rules.slow} 天均线")
    ax.plot(ma.index, ma * (1 + rules.band), color="#1baf7a", lw=1.0, ls=":",
            label=f"买入线（均线 +{rules.band:.0%}）")
    ax.plot(ma.index, ma * (1 - rules.band), color="#eb6834", lw=1.2, ls="--",
            label=f"止损线（均线 -{rules.band:.0%}）")
    styles = {
        "buy": dict(marker="^", color="#008300", label="买入"),
        "stop": dict(marker="v", color="#e34948", label="卖出：止损"),
        "tp": dict(marker="D", color="#eda100", label="卖出：止盈 / 背离"),
    }
    for kind, st in styles.items():
        if lg.empty:
            continue
        if kind == "buy":
            sel = lg[lg.action == "BUY"]
        elif kind == "stop":
            sel = lg[(lg.action == "SELL") & lg.reason.str.startswith("stop")]
        else:
            sel = lg[(lg.action == "SELL") & ~lg.reason.str.startswith("stop")]
        if len(sel):
            ax.scatter(sel.date, sel.price, s=90, zorder=5, edgecolor="#fcfcfb", linewidth=1.5,
                       marker=st["marker"], color=st["color"], label=st["label"])
            for i, (_, r) in enumerate(sel.iterrows()):
                dy = (12 if kind == "buy" else -18) * (1 + (i % 2) * 0.8)
                ax.annotate(f"{r.price:,.0f}", (r.date, r.price), textcoords="offset points",
                            xytext=(0, dy), ha="center", fontsize=8, color="#52514e")

    # current plan at the right edge
    if status and status["position"] > 0:
        x = c.index[-1]
        levels = [(status["stop_below"], "#eb6834", f"现在的止损 {status['stop_below']:,.0f}")]
        nt = status.get("next_target")
        if nt is not None and not np.isnan(nt):
            levels.append((nt, "#eda100", f"下一个止盈 {nt:,.0f}"))
        for y, col, text in levels:
            ax.axhline(y, color=col, lw=0.8, alpha=0.6, xmin=0.9)
            ax.annotate(text, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=9,
                        color="#0b0b0b", ha="left")
    state = "持有" if status and status["position"] > 0 else "空仓"
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 10 else f"{v:,.2f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_title(f"{symbol}：{rules.slow} 天均线趋势规则的买卖点（{since[:4]} 年至今）   今天：{state}",
                 loc="left", color="#0b0b0b")
    ax.grid(True, color="#e8e7e2", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.margins(x=0.08)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def load_daily(symbol: str, source: str, proxy: Optional[str], start: str) -> pd.Series:
    from src.data.crypto_data_fetcher import fetch_crypto_daily, fetch_ohlcv_ccxt

    if source == "okx":
        return fetch_ohlcv_ccxt(f"{symbol}/USDT", "1d", start, exchange="okx", proxy=proxy)["close"]
    return fetch_crypto_daily(symbol, start=start, source=source)["close"]


def parse_tp(spec: Optional[str]) -> Tuple[Tuple[float, float], ...]:
    if not spec:
        return ()
    return tuple((float(a), float(b)) for a, b in (x.split(":") for x in spec.split(",")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--source", default="okx", choices=["okx", "coinmetrics", "auto", "yfinance"])
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--slow", type=int, default=150)
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--tp", default=None, help='take-profit tranches "gain:fraction,...", e.g. 0.3:0.33')
    ap.add_argument("--chart-since", default="2023-01-01")
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "crypto", "signals"))
    args = ap.parse_args()
    warnings.simplefilter("ignore", FutureWarning)
    os.makedirs(args.out, exist_ok=True)

    variants = {"all in / all out": Rules(args.slow, args.band),
                "with take-profit tranches": Rules(args.slow, args.band, parse_tp(args.tp or "0.3:0.33"))}
    chosen = variants["with take-profit tranches"] if args.tp else variants["all in / all out"]
    rows, status = {}, {}
    for sym in args.symbols:
        close = load_daily(sym, args.source, args.proxy, args.start)
        ev = divergence_events(close, DivergenceConfig())
        test_start = close.index[0] + pd.Timedelta(days=args.slow + 30)
        for name, rules in [("buy & hold", None)] + list(variants.items()):
            pos = pd.Series(1.0, index=close.index) if rules is None else run_rules(close, rules, ev)[0]
            c = close.loc[test_start:]
            m = crypto_metrics(backtest_positions(c, pos.reindex(c.index).fillna(0.0), args.cost))
            rows[(sym, name)] = {"from": c.index[0].date(), "cagr": m["cagr"], "max_drawdown": m["max_drawdown"],
                                 "sharpe": m["sharpe"],
                                 "trades": int((pos.loc[test_start:].diff().fillna(0) != 0).sum())}
        pos, log = run_rules(close, chosen, ev)
        log.to_csv(os.path.join(args.out, f"{sym}_trades.csv"), index=False)
        status[sym] = status_today(close, pos, log, chosen)
        plot(sym, close, log, chosen, args.chart_since, os.path.join(args.out, f"{sym}_chart.png"), status[sym])

    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nToday:")
    for sym, s in status.items():
        state = "HOLD" if s["position"] > 0 else "CASH"
        line = (f"{sym} {s['date']}: {state} ({s['position']:.0%}), close {s['close']:,.2f}, "
                f"buy above {s['buy_above']:,.2f}, stop below {s['stop_below']:,.2f}")
        if s.get("next_target") and not np.isnan(s["next_target"]) and s["position"] > 0:
            line += f", next take-profit {s['next_target']:,.2f}"
        print(line + f"\n    last: {s.get('last_action', '-')}")
    print(f"\nCharts and trade logs in {args.out}")


if __name__ == "__main__":
    main()
