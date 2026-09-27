"""Offline tests for the ccxt crypto data module (uses a fake exchange)."""

import pandas as pd
import pytest

from src.data.crypto_fetcher import CryptoDataFetcher, to_price_wide, timeframe_to_ms
from src.data.data_store import DataStore

HOUR = 3_600_000
DAY = 24 * HOUR


class FakeExchange:
    """Serves synthetic bars from `first_ts` up to `now`, `page` bars per call."""

    def __init__(self, now_ms, first_ts=0, page=500, gap=None):
        self.now = now_ms
        self.first_ts = first_ts
        self.page = page
        self.gap = gap or ()  # (start, end) ms range with no bars (outage)
        self.calls = []

    def milliseconds(self):
        return self.now

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls.append((timeframe, since))
        step = timeframe_to_ms(timeframe)
        ts = max(since, self.first_ts)
        ts = -(-ts // step) * step
        out = []
        while ts < self.now and len(out) < min(limit, self.page):
            if not (self.gap and self.gap[0] <= ts <= self.gap[1]):
                price = 100.0 + ts / step
                out.append([ts, price, price + 1, price - 1, price + 0.5, 10.0])
            ts += step
        return out


@pytest.fixture
def store(tmp_path):
    return DataStore(str(tmp_path))


def _now():
    return int(pd.Timestamp('2024-03-10 15:30', tz='UTC').value // 1_000_000)


def test_daily_fetch_is_cached_and_excludes_open_bar(store):
    ex = FakeExchange(_now())
    f = CryptoDataFetcher('fake', data_store=store, exchange=ex)
    df = f.get_ohlcv('BTC/USDT', '1d', '2024-01-01')
    assert df['datetime'].iloc[0] == '2024-01-01 00:00:00'
    assert df['datetime'].iloc[-1] == '2024-03-09 00:00:00'  # 03-10 still forming
    assert len(df) == 69

    n_calls = len(ex.calls)
    again = f.get_ohlcv('BTC/USDT', '1d', '2024-01-01', '2024-03-09')
    assert len(ex.calls) == n_calls  # served from cache
    pd.testing.assert_frame_equal(df, again)


def test_hourly_pagination_and_incremental_extension(store):
    ex = FakeExchange(_now(), page=100)
    f = CryptoDataFetcher('fake', data_store=store, exchange=ex)
    df = f.get_ohlcv('BTC/USDT', '1h', '2024-03-01', '2024-03-05')
    assert len(df) == 5 * 24
    assert (df['ts'].diff().dropna() == HOUR).all()  # continuous 24/7, no calendar gaps

    ex.calls.clear()
    df2 = f.get_ohlcv('BTC/USDT', '1h', '2024-02-28', '2024-03-06')
    assert len(df2) == 8 * 24
    # Only the two uncached edges were requested
    starts = sorted(c[1] for c in ex.calls)
    assert starts[0] == int(pd.Timestamp('2024-02-28', tz='UTC').value // 1_000_000)
    assert starts[-1] == int(pd.Timestamp('2024-03-06', tz='UTC').value // 1_000_000)


def test_outage_gap_is_not_refetched(store):
    gap_start = int(pd.Timestamp('2024-03-02 05:00', tz='UTC').value // 1_000_000)
    ex = FakeExchange(_now(), gap=(gap_start, gap_start + 2 * HOUR))
    f = CryptoDataFetcher('fake', data_store=store, exchange=ex)
    df = f.get_ohlcv('BTC/USDT', '1h', '2024-03-02', '2024-03-02')
    assert len(df) == 21
    ex.calls.clear()
    f.get_ohlcv('BTC/USDT', '1h', '2024-03-02', '2024-03-02')
    assert ex.calls == []


def test_price_long_format_and_wide_table(store):
    f = CryptoDataFetcher('fake', data_store=store, exchange=FakeExchange(_now()))
    long_df = f.get_price_data(['BTC/USDT', 'ETH/USDT'], '2024-01-01', '2024-01-31', '1d')
    assert set(long_df.columns) >= {'tic', 'datadate', 'adj_close', 'cshtrd'}
    assert long_df['datadate'].iloc[0] == '2024-01-01'
    wide = to_price_wide(long_df)
    assert list(wide.columns) == ['BTC-USDT', 'ETH-USDT']
    assert isinstance(wide.index, pd.DatetimeIndex) and len(wide) == 31
    # Weekends are present (no NYSE calendar filtering)
    assert (wide.index.dayofweek >= 5).sum() == 8


def test_rejects_weekly_timeframe():
    with pytest.raises(ValueError):
        timeframe_to_ms('1w')


def test_wide_table_runs_through_backtest_engine(store):
    bt = pytest.importorskip('bt')
    from src.backtest.backtest_engine import BacktestConfig, BacktestEngine

    for tf, start, end in [('1d', '2023-06-01', '2024-03-01'), ('1h', '2024-03-01', '2024-03-05')]:
        f = CryptoDataFetcher('fake', data_store=store, exchange=FakeExchange(_now()))
        long_df = f.get_price_data(['BTC/USDT'], start, end, tf)
        prices = to_price_wide(long_df)
        weights = pd.DataFrame({'BTC-USDT': [1.0]}, index=prices.index[:1])
        cfg = BacktestConfig(start_date=start, end_date=end, benchmark_tickers=[],
                             integer_positions=False)
        res = BacktestEngine(cfg).run_backtest('hold', prices, weights)
        assert len(res.portfolio_values) >= len(prices)
        # Fully invested buy-and-hold tracks the asset, minus the 0.1% entry fee
        asset_ret = prices['BTC-USDT'].iloc[-1] / prices['BTC-USDT'].iloc[0]
        port_ret = res.portfolio_values.iloc[-1] / res.portfolio_values.iloc[0]
        assert port_ret == pytest.approx(asset_ret * (1 - cfg.transaction_cost), rel=1e-4)
