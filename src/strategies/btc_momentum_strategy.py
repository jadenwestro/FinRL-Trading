"""
BTC Time-Series Momentum Timing Baseline
========================================

Single-asset trend timing: hold BTC when its trailing momentum is positive,
otherwise hold cash. The signal is the repo's TS-MOM engine
(``tsmomsignal.TSMOMSignalEngine``, Moskowitz et al. 2012):

    ret_Lm = P(t-1m) / P(t-Lm) - 1      (monthly closes, skip the last month)
    signal = +1 if ret_Lm > band, -1 if ret_Lm < -band, else 0

Long-only mapping: +1 -> 100% BTC, 0 / -1 -> 100% cash.
Signals are formed on month-end closes and executed on the next day's close.

Backtests run through ``backtest.backtest_engine.BacktestEngine`` (bt, with a
flat per-trade cost). Because crypto trades every calendar day, the report
annualises daily statistics with 365 periods per year, not 252.

Usage:
    python src/strategies/btc_momentum_strategy.py \
        --start 2015-01-01 --lookback 12 --cost 0.001
"""

from __future__ import annotations

import argparse
import logging
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

from strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult  # noqa: E402
from strategies.tsmomsignal import TSMOMSignalEngine  # noqa: E402

logger = logging.getLogger(__name__)

CRYPTO_PERIODS_PER_YEAR = 365


class _QuietLogger:
    """Stand-in for StrategyLogger so the TS-MOM engine doesn't spawn a writer thread."""

    def log_error(self, msg):
        logger.debug(msg)


@dataclass
class BTCMomentumConfig(StrategyConfig):
    name: str = "BTC_TSMOM"
    ticker: str = "BTC"
    lookback_months: int = 12
    neutral_band: float = 0.0
    execution_lag_days: int = 1  # trade on the close after the month-end signal


class BTCMomentumStrategy(BaseStrategy):
    """Hold BTC or cash based on TS-MOM sign."""

    def __init__(self, config: BTCMomentumConfig):
        super().__init__(config)
        self.engine = TSMOMSignalEngine(
            strategy_name=config.name,
            logger=_QuietLogger(),
            lookback_months=config.lookback_months,
            neutral_band=config.neutral_band,
        )

    def generate_weights(self, data: Dict[str, pd.DataFrame],
                         target_date: Optional[str] = None) -> StrategyResult:
        """
        Args:
            data: {"prices": DataFrame indexed by date with a ``close`` column}
            target_date: ignore prices after this date (no look-ahead)

        Returns:
            StrategyResult whose ``weights`` is a DataFrame indexed by the
            effective trade date, one column (the ticker), values in {0, 1}.
        """
        prices = data["prices"]
        if target_date is not None:
            prices = prices.loc[:pd.Timestamp(target_date)]
        df = prices[["close"]].rename_axis("date").reset_index()

        with warnings.catch_warnings():
            # pandas >= 2.2 deprecates resample("M") used inside tsmomsignal.
            warnings.simplefilter("ignore", FutureWarning)
            signal = self.engine.generate_signal_one_ticker(df)

        weight = (signal > 0).astype(float)
        weight.index = weight.index + pd.Timedelta(days=self.config.execution_lag_days)
        weights = weight.to_frame(self.config.ticker)
        weights.index.name = "date"

        return StrategyResult(
            strategy_name=self.config.name,
            weights=weights,
            metadata={"raw_signal": signal,
                      "lookback_months": self.config.lookback_months,
                      "neutral_band": self.config.neutral_band},
        )


# ---------------------------------------------------------------------------
# Metrics (365-day annualisation)
# ---------------------------------------------------------------------------

def crypto_metrics(values: pd.Series, weights: Optional[pd.Series] = None,
                   periods_per_year: int = CRYPTO_PERIODS_PER_YEAR) -> Dict[str, float]:
    """Performance stats for a daily (calendar-day) equity curve."""
    values = values.dropna()
    rets = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.0
    total = values.iloc[-1] / values.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = rets.mean() / rets.std() * np.sqrt(periods_per_year) if rets.std() > 0 else np.nan
    downside = rets[rets < 0].std() * np.sqrt(periods_per_year)
    sortino = rets.mean() * periods_per_year / downside if downside > 0 else np.nan
    dd = values / values.cummax() - 1
    mdd = dd.min()
    out = {
        "total_return": total,
        "cagr": cagr,
        "annual_volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
    }
    if weights is not None:
        w = weights.reindex(values.index).ffill().fillna(0.0)
        out["exposure"] = float((w > 0).mean())
        out["round_trips"] = int((w.diff() > 0).sum() + (w.iloc[0] > 0))
    return out


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_bt(name: str, prices: pd.DataFrame, weights: pd.DataFrame,
           cost: float, start: str, end: str) -> pd.Series:
    """Run one strategy through the repo's bt-based BacktestEngine."""
    from src.backtest.backtest_engine import BacktestConfig, BacktestEngine

    cfg = BacktestConfig(
        start_date=start,
        end_date=end,
        transaction_cost=cost,
        benchmark_tickers=[],       # BTC buy-and-hold is run explicitly below
        integer_positions=False,    # 1 BTC is too coarse a lot for integer shares
    )
    res = BacktestEngine(cfg).run_backtest(name, prices, weights)
    return res.portfolio_values


def backtest(prices: pd.DataFrame, lookback: int, band: float, cost: float,
             start: str, end: Optional[str], ticker: str = "BTC") -> Dict[str, dict]:
    end = end or prices.index[-1].strftime("%Y-%m-%d")
    strat = BTCMomentumStrategy(BTCMomentumConfig(ticker=ticker, lookback_months=lookback,
                                                  neutral_band=band))
    # Signals use the full history (for warm-up); the backtest window is [start, end].
    result = strat.generate_weights({"prices": prices}, target_date=end)

    px = prices.loc[start:end, ["close"]].rename(columns={"close": ticker})
    w_daily = result.weights.reindex(result.weights.index.union(px.index)).ffill() \
        .reindex(px.index).fillna(0.0)
    bh = pd.DataFrame({ticker: 1.0}, index=px.index)

    tsmom_vals = run_bt(f"TSMOM_{lookback}m", px, w_daily, cost, start, end)
    bh_vals = run_bt("BuyHold", px, bh, cost, start, end)
    return {
        f"TSMOM {lookback}m": {"values": tsmom_vals, "weights": w_daily[ticker]},
        "Buy & Hold": {"values": bh_vals, "weights": bh[ticker]},
    }


def vectorized_check(px: pd.Series, w: pd.Series, cost: float) -> pd.Series:
    """Independent close-to-close equity curve, used to sanity-check bt."""
    rets = px.pct_change().fillna(0.0)
    pos = w.shift(1).fillna(0.0)           # weight set at close t earns t+1
    turnover = w.diff().abs().fillna(w.iloc[0])
    return (1 + pos * rets - turnover * cost).cumprod()


def _fmt_table(rows: Dict[str, Dict[str, float]]) -> str:
    cols = ["total_return", "cagr", "annual_volatility", "sharpe", "sortino",
            "max_drawdown", "calmar", "exposure", "round_trips"]
    pct = {"total_return", "cagr", "annual_volatility", "max_drawdown", "exposure"}
    lines = ["| strategy | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for name, m in rows.items():
        cells = []
        for c in cols:
            v = m.get(c, np.nan)
            if c in pct:
                cells.append(f"{v:.1%}")
            elif c == "round_trips":
                cells.append(f"{int(v)}")
            else:
                cells.append(f"{v:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    from src.data.crypto_data_fetcher import fetch_crypto_daily

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default="BTC")
    ap.add_argument("--data-start", default="2013-01-01", help="history used for signal warm-up")
    ap.add_argument("--start", default="2015-01-01", help="backtest start")
    ap.add_argument("--end", default=None)
    ap.add_argument("--lookback", type=int, default=12, help="momentum lookback in months")
    ap.add_argument("--band", type=float, default=0.0, help="neutral band on trailing return")
    ap.add_argument("--cost", type=float, default=0.001, help="cost per unit traded (0.001 = 10 bp)")
    ap.add_argument("--source", default="auto", choices=["auto", "yfinance", "ccxt", "coinmetrics"])
    ap.add_argument("--sweep", action="store_true", help="also report lookback / cost sensitivity")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "crypto", "results"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    warnings.simplefilter("ignore", FutureWarning)  # resample("M") noise from the engine
    prices = fetch_crypto_daily(args.ticker, start=args.data_start, end=args.end, source=args.source)
    end = args.end or prices.index[-1].strftime("%Y-%m-%d")
    print(f"Data: {args.ticker} {prices.index[0].date()} -> {prices.index[-1].date()} "
          f"({len(prices)} daily bars, source={prices.attrs.get('source')})")
    print(f"Backtest: {args.start} -> {end}, cost={args.cost:.2%} per trade, "
          f"annualisation={CRYPTO_PERIODS_PER_YEAR} days\n")

    runs = backtest(prices, args.lookback, args.band, args.cost, args.start, end, args.ticker)
    rows = {k: crypto_metrics(v["values"], v["weights"]) for k, v in runs.items()}
    print(_fmt_table(rows))

    # Cross-check bt against a plain vectorised calculation.
    px = prices.loc[args.start:end, "close"]
    key = f"TSMOM {args.lookback}m"
    vec = vectorized_check(px, runs[key]["weights"], args.cost)
    bt_curve = runs[key]["values"] / runs[key]["values"].iloc[0]
    print(f"\nbt vs vectorised final equity: {bt_curve.iloc[-1]:.3f} vs {vec.iloc[-1]:.3f}")

    os.makedirs(args.out, exist_ok=True)
    curves = pd.DataFrame({k: v["values"] / v["values"].iloc[0] for k, v in runs.items()})
    curves.to_csv(os.path.join(args.out, "btc_tsmom_equity.csv"))

    if args.sweep:
        sweep = {}
        for cost in (0.001, 0.0025):
            for lb in (3, 6, 9, 12):  # 1m is degenerate: TS-MOM skips the last month
                r = backtest(prices, lb, args.band, cost, args.start, end, args.ticker)
                sweep[f"TSMOM {lb}m @ {cost:.2%}"] = crypto_metrics(
                    r[f"TSMOM {lb}m"]["values"], r[f"TSMOM {lb}m"]["weights"])
            sweep[f"Buy & Hold @ {cost:.2%}"] = crypto_metrics(
                r["Buy & Hold"]["values"], r["Buy & Hold"]["weights"])
        print("\nSensitivity (lookback x cost):\n")
        print(_fmt_table(sweep))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        curves.plot(ax=axes[0], logy=True, title=f"{args.ticker}: TS-MOM timing vs buy & hold")
        axes[0].set_ylabel("growth of 1 (log)")
        (curves / curves.cummax() - 1).plot(ax=axes[1], legend=False)
        axes[1].set_ylabel("drawdown")
        fig.tight_layout()
        path = os.path.join(args.out, "btc_tsmom_equity.png")
        fig.savefig(path, dpi=120)
        print(f"\nSaved equity curve to {path}")
    except Exception as e:  # plotting is optional
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
