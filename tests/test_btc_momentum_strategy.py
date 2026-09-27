import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from strategies.btc_momentum_strategy import (  # noqa: E402
    BTCMomentumConfig, BTCMomentumStrategy, crypto_metrics, vectorized_check,
)


def _prices(n_days=900, seed=0):
    idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rets = np.random.default_rng(seed).normal(0.001, 0.03, n_days)
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(rets))}, index=idx)


def test_weights_are_long_only_and_lagged():
    px = _prices()
    res = BTCMomentumStrategy(BTCMomentumConfig(lookback_months=6)).generate_weights({"prices": px})
    w = res.weights["BTC"]
    assert set(w.unique()) <= {0.0, 1.0}
    # every trade date is the day after a month end
    assert all((d - pd.Timedelta(days=1)).is_month_end for d in w.index)
    # a negative (-1) raw signal must map to cash, not a short
    raw = res.metadata["raw_signal"]
    assert (w[(raw < 0).values] == 0).all()


def test_no_lookahead():
    px = _prices()
    strat = BTCMomentumStrategy(BTCMomentumConfig(lookback_months=6))
    full = strat.generate_weights({"prices": px}).weights
    cut = "2021-06-30"
    shocked = px.copy()
    shocked.loc[pd.Timestamp(cut) + pd.Timedelta(days=1):, "close"] *= 0.1
    partial = strat.generate_weights({"prices": shocked}).weights
    upto = pd.Timestamp(cut) + pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(full.loc[:upto], partial.loc[:upto])


def test_metrics_use_365_day_annualisation():
    idx = pd.date_range("2021-01-01", periods=366, freq="D")
    values = pd.Series(1.001 ** np.arange(366), index=idx)
    m = crypto_metrics(values)
    assert np.isclose(m["cagr"], 1.001 ** 365 - 1)
    assert m["max_drawdown"] == 0


def test_vectorized_check_charges_costs():
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    px = pd.Series(100.0, index=idx)
    w = pd.Series([1, 1, 0, 0, 1.0], index=idx)
    eq = vectorized_check(px, w, cost=0.01)
    assert np.isclose(eq.iloc[-1], 0.99 ** 3)
