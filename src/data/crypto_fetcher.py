# -*- coding: utf-8 -*-
"""
Crypto Data Fetcher
===================

Fetches crypto OHLCV bars (e.g. BTC/USDT daily and hourly) from any exchange
supported by ccxt and caches them in the project's SQLite database
(``crypto_ohlcv`` table in ``finrl_trading.db``).

Crypto trades 24/7, so unlike ``data_fetcher.FMPFetcher`` this module never
consults the NYSE trading calendar: cache coverage is tracked as plain UTC
time intervals.

Output matches the long format the rest of the project uses
(``tic, datadate, prcod, prchd, prcld, prccd, adj_close, cshtrd, gvkey``), so
it can be passed straight to ``BacktestEngine.run_backtest`` or pivoted into a
price wide-table with :func:`to_price_wide`.

Example::

    from src.data.crypto_fetcher import fetch_crypto_price_data, to_price_wide
    long_df = fetch_crypto_price_data(['BTC/USDT'], '2020-01-01', timeframe='1d')
    prices = to_price_wide(long_df)   # index: datetime, columns: ['BTC-USDT']
"""

import logging
from typing import List, Optional

import pandas as pd

from src.data.data_store import DataStore, get_data_store

logger = logging.getLogger(__name__)

DEFAULT_EXCHANGE = 'binance'


def symbol_to_tic(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC-USDT' (a column-name friendly ticker)."""
    return symbol.replace('/', '-').replace(':', '-')


def timeframe_to_ms(timeframe: str) -> int:
    """Length of one bar in milliseconds for minute/hour/day timeframes."""
    units = {'m': 60_000, 'h': 3_600_000, 'd': 86_400_000}
    unit = timeframe[-1]
    if unit not in units or not timeframe[:-1].isdigit():
        # Weekly/monthly bars aren't aligned to the Unix epoch (weeks start on
        # Monday, months vary in length), which breaks the gap bookkeeping.
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Use minutes/hours/days "
            f"(e.g. '1h', '4h', '1d') and resample for weekly/monthly bars."
        )
    return int(timeframe[:-1]) * units[unit]


def _to_utc_ms(value, end_of_day: bool = False) -> int:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
    # A bare date as end bound means "through the end of that day"
    if end_of_day and ts == ts.normalize() and isinstance(value, str) and len(value) <= 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    return int(ts.value // 1_000_000)


class CryptoDataFetcher:
    """Fetch and cache crypto OHLCV bars via ccxt."""

    def __init__(self, exchange_id: str = DEFAULT_EXCHANGE,
                 data_store: Optional[DataStore] = None,
                 exchange=None, page_limit: int = 1000):
        """
        Args:
            exchange_id: ccxt exchange id ('binance', 'okx', 'bybit', ...)
            data_store: DataStore to cache into (defaults to the global one)
            exchange: Pre-built ccxt exchange instance (mainly for tests /
                custom proxies). Built from exchange_id when None.
            page_limit: Max bars requested per API call
        """
        self.exchange_id = exchange_id
        self.data_store = data_store or get_data_store()
        self.page_limit = page_limit
        if exchange is None:
            import ccxt
            exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        self.exchange = exchange

    def _now_ms(self) -> int:
        return int(self.exchange.milliseconds())

    def _download(self, symbol: str, timeframe: str,
                  start_ts: int, end_ts: int, step_ms: int) -> int:
        """Page through the exchange for [start_ts, end_ts] and cache the bars."""
        since = start_ts
        saved = 0
        while since <= end_ts:
            batch = self.exchange.fetch_ohlcv(symbol, timeframe, since=since,
                                              limit=self.page_limit)
            if not batch:
                break
            bars = [b for b in batch if start_ts <= b[0] <= end_ts]
            saved += self.data_store.save_crypto_ohlcv(self.exchange_id, symbol,
                                                       timeframe, bars)
            last_ts = batch[-1][0]
            if last_ts < since:  # defensive: exchange ignored `since`
                break
            since = last_ts + step_ms
        return saved

    def get_ohlcv(self, symbol: str, timeframe: str = '1d',
                  start_date: str = '2020-01-01',
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get OHLCV bars for one symbol, fetching only what the cache lacks.

        Only closed bars are cached and returned; the still-forming current
        bar is excluded so cached data never goes stale.

        Returns:
            DataFrame with columns ts (ms), datetime (UTC string), open, high,
            low, close, volume.
        """
        step_ms = timeframe_to_ms(timeframe)
        start_ts = _to_utc_ms(start_date)
        start_ts = -(-start_ts // step_ms) * step_ms  # ceil to bar boundary
        end_ts = _to_utc_ms(end_date, end_of_day=True) if end_date else self._now_ms()
        last_closed = (self._now_ms() // step_ms) * step_ms - step_ms
        end_ts = min((end_ts // step_ms) * step_ms, last_closed)
        if end_ts < start_ts:
            return pd.DataFrame(columns=['ts', 'datetime', 'open', 'high', 'low', 'close', 'volume'])

        missing = self.data_store.get_missing_crypto_ranges(
            self.exchange_id, symbol, timeframe, start_ts, end_ts, step_ms)
        for s, e in missing:
            logger.info(f"Fetching {self.exchange_id} {symbol} {timeframe} "
                        f"{pd.Timestamp(s, unit='ms')} -> {pd.Timestamp(e, unit='ms')}")
            n = self._download(symbol, timeframe, s, e, step_ms)
            self.data_store.save_crypto_fetch_range(self.exchange_id, symbol,
                                                    timeframe, s, e, n)
        if not missing:
            logger.info(f"{self.exchange_id} {symbol} {timeframe}: served from cache")

        return self.data_store.get_crypto_ohlcv(self.exchange_id, symbol,
                                                timeframe, start_ts, end_ts)

    def get_price_data(self, symbols: List[str], start_date: str,
                       end_date: Optional[str] = None,
                       timeframe: str = '1d') -> pd.DataFrame:
        """
        Get bars for several symbols in the project's long price format.

        Returns:
            DataFrame with columns tic, gvkey, datadate, prcod, prchd, prcld,
            prccd, adj_close, cshtrd. datadate is 'YYYY-MM-DD' for daily bars
            and 'YYYY-MM-DD HH:MM:SS' (UTC) for intraday bars.
        """
        daily = timeframe_to_ms(timeframe) % 86_400_000 == 0
        frames = []
        for symbol in symbols:
            df = self.get_ohlcv(symbol, timeframe, start_date, end_date)
            if df.empty:
                logger.warning(f"No {timeframe} data for {symbol} on {self.exchange_id}")
                continue
            tic = symbol_to_tic(symbol)
            frames.append(pd.DataFrame({
                'tic': tic,
                'gvkey': tic,
                'datadate': df['datetime'].str[:10] if daily else df['datetime'],
                'prcod': df['open'],
                'prchd': df['high'],
                'prcld': df['low'],
                'prccd': df['close'],
                # No splits/dividends in crypto: adjusted close == close
                'adj_close': df['close'],
                'cshtrd': df['volume'],
            }))
        if not frames:
            return pd.DataFrame(columns=['tic', 'gvkey', 'datadate', 'prcod', 'prchd',
                                         'prcld', 'prccd', 'adj_close', 'cshtrd'])
        return pd.concat(frames, ignore_index=True)


def fetch_crypto_price_data(symbols: List[str], start_date: str,
                            end_date: Optional[str] = None,
                            timeframe: str = '1d',
                            exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    """Convenience wrapper around :meth:`CryptoDataFetcher.get_price_data`."""
    return CryptoDataFetcher(exchange).get_price_data(symbols, start_date, end_date, timeframe)


def to_price_wide(price_long: pd.DataFrame, value_col: str = 'adj_close') -> pd.DataFrame:
    """Pivot long price data into a wide table (DatetimeIndex x tickers)."""
    wide = price_long.pivot(index='datadate', columns='tic', values=value_col)
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = 'datetime'
    wide.columns.name = None
    return wide.sort_index()
