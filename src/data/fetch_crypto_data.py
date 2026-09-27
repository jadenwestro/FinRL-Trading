#!/usr/bin/env python3
"""Fetch crypto OHLCV (default: BTC/USDT daily + hourly) via ccxt and cache it.

Examples:
    python3 src/data/fetch_crypto_data.py
    python3 src/data/fetch_crypto_data.py --symbols BTC/USDT ETH/USDT --timeframes 1d 4h \
        --start-date 2019-01-01 --exchange okx --check-backtest
"""

import argparse
import logging
import os
import sys

# Allow running as standalone script: python3 src/data/fetch_crypto_data.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

from src.data.crypto_fetcher import CryptoDataFetcher, to_price_wide


def check_backtest(price_long: pd.DataFrame, label: str) -> None:
    """Run a buy-and-hold backtest to confirm the data plugs into BacktestEngine."""
    from src.backtest.backtest_engine import BacktestConfig, BacktestEngine

    prices = to_price_wide(price_long)
    weights = pd.DataFrame(1.0 / prices.shape[1], index=prices.index[:1], columns=prices.columns)
    config = BacktestConfig(
        start_date=str(prices.index.min().date()),
        end_date=str(prices.index.max().date()),
        benchmark_tickers=[],  # SPY/QQQ defaults need the FMP stock feed
        integer_positions=False,  # 1 BTC is too coarse a lot size
    )
    result = BacktestEngine(config).run_backtest(f"{label} buy&hold", prices, weights)
    print(f"  backtest OK: {len(result.portfolio_values)} bars, "
          f"total return {result.metrics.get('total_return', float('nan')):.2%}")


def main():
    parser = argparse.ArgumentParser(description="Fetch & cache crypto OHLCV via ccxt")
    parser.add_argument("--exchange", default="binance", help="ccxt exchange id (default: binance)")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT"])
    parser.add_argument("--timeframes", nargs="+", default=["1d", "1h"])
    parser.add_argument("--start-date", default="2018-01-01", help="UTC, inclusive")
    parser.add_argument("--end-date", default=None, help="UTC, inclusive (default: last closed bar)")
    parser.add_argument("--output-dir", default=None, help="Also save wide close-price CSVs here")
    parser.add_argument("--check-backtest", action="store_true",
                        help="Run a buy-and-hold backtest on each timeframe as a smoke test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fetcher = CryptoDataFetcher(args.exchange)

    for tf in args.timeframes:
        long_df = fetcher.get_price_data(args.symbols, args.start_date, args.end_date, timeframe=tf)
        if long_df.empty:
            print(f"[{tf}] no data")
            continue
        wide = to_price_wide(long_df)
        print(f"[{tf}] {wide.shape[0]} bars x {wide.shape[1]} symbols, "
              f"{wide.index.min()} -> {wide.index.max()} (UTC)")
        print(wide.tail(3).to_string())
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir, f"crypto_{args.exchange}_{tf}.csv")
            wide.to_csv(path)
            print(f"  saved {path}")
        if args.check_backtest:
            check_backtest(long_df, tf)


if __name__ == "__main__":
    main()
