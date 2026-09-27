"""
Crypto Data Fetcher
===================

Daily BTC (and other crypto) close prices for single-asset research.

Crypto trades 24/7, so unlike ``data_fetcher.py`` there is no NYSE calendar
here: every calendar day is a bar (UTC close).

Sources, tried in order when ``source="auto"``:
  1. yfinance  (``BTC-USD``)
  2. ccxt      (Binance ``BTC/USDT`` daily candles, paginated)
  3. coinmetrics community CSV on GitHub (``PriceUSD``, daily since 2010)

Results are cached as CSV under ``data/crypto/`` (gitignored).
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "crypto"

COINMETRICS_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/{asset}.csv"


def _fetch_yfinance(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(f"{symbol}-USD", start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _fetch_ccxt(symbol: str, start: str, end: Optional[str], exchange: str = "binance") -> pd.DataFrame:
    import ccxt

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    pair = f"{symbol}/USDT"
    since = ex.parse8601(f"{start}T00:00:00Z")
    end_ms = ex.parse8601(f"{end}T00:00:00Z") if end else ex.milliseconds()
    rows = []
    while since < end_ms:
        batch = ex.fetch_ohlcv(pair, timeframe="1d", since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 24 * 3600 * 1000
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df.drop(columns="ts")


def _fetch_coinmetrics(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    raw = pd.read_csv(COINMETRICS_URL.format(asset=symbol.lower()), usecols=["time", "PriceUSD"])
    raw = raw.dropna(subset=["PriceUSD"])
    df = pd.DataFrame({"close": raw["PriceUSD"].astype(float).values},
                      index=pd.to_datetime(raw["time"]))
    return df


_SOURCES = {
    "yfinance": _fetch_yfinance,
    "ccxt": _fetch_ccxt,
    "coinmetrics": _fetch_coinmetrics,
}


def fetch_crypto_daily(
    symbol: str = "BTC",
    start: str = "2014-01-01",
    end: Optional[str] = None,
    source: str = "auto",
    cache_dir: Optional[os.PathLike] = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily crypto prices.

    Returns a DataFrame indexed by calendar date (DatetimeIndex named ``date``)
    with at least a ``close`` column and a ``source`` attribute in ``df.attrs``.
    """
    order = list(_SOURCES) if source == "auto" else [source]
    cache_path = Path(cache_dir) / f"{symbol.upper()}_daily.csv" if cache_dir else None

    df = pd.DataFrame()
    if use_cache and cache_path is not None and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.attrs["source"] = f"cache:{cache_path.name}"
        logger.info(f"Loaded {len(df)} rows from cache {cache_path}")
    else:
        errors = {}
        for name in order:
            try:
                df = _SOURCES[name](symbol, start, end)
            except Exception as e:  # network / API failures fall through to next source
                errors[name] = f"{type(e).__name__}: {e}"
                logger.warning(f"{name} failed for {symbol}: {errors[name][:200]}")
                continue
            if not df.empty:
                df.attrs["source"] = name
                break
        if df.empty:
            raise RuntimeError(f"No data for {symbol} from any source: {errors}")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path)

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "date"
    df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end)]
    return df
