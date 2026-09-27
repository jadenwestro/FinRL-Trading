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


def test_divergence_events_are_causal():
    from strategies.btc_ma_divergence_strategy import DivergenceConfig, divergence_events

    close = _prices(1200, seed=1)["close"]
    cfg = DivergenceConfig(method="pivot", n_peaks=2)
    full = divergence_events(close, cfg)
    cut = close.index[700]
    shocked = close.copy()
    shocked.loc[cut + pd.Timedelta(days=1):] *= np.linspace(1, 3, len(shocked.loc[cut + pd.Timedelta(days=1):]))
    partial = divergence_events(shocked, cfg)
    pd.testing.assert_series_equal(full.loc[:cut], partial.loc[:cut])


def test_overlay_halves_against_trend_and_never_shorts_by_default():
    from strategies.btc_ma_divergence_strategy import divergence_overlay

    idx = pd.date_range("2021-01-01", periods=6, freq="D")
    regime = pd.Series([1, 1, 1, 1, -1, -1.0], index=idx)
    events = pd.Series({idx[1]: -1.0, idx[3]: 1.0})
    pos = divergence_overlay(regime, events, against="half")
    assert pos.tolist() == [1, 0.5, 0.5, 1, 0, 0]


def test_mtf_study_trades_are_causal_on_synthetic_data():
    from strategies.btc_mtf_divergence_study import run_study

    rng = np.random.default_rng(3)
    idx = pd.date_range("2019-01-01", periods=96 * 600, freq="15min")
    c = 20000 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
    m15 = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "close": c}, index=idx)
    m15["high"] = m15[["open", "close"]].max(axis=1) * 1.001
    m15["low"] = m15[["open", "close"]].min(axis=1) * 0.999
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    trades = run_study(m15.resample("1D").agg(agg), m15.resample("4h").agg(agg), m15)
    assert not trades.empty
    assert (trades.entry_time >= trades.signal_time).all()
    assert (trades.exit_time > trades.entry_time).all()
    assert set(trades.outcome) <= {"stop", "target", "time"}


def test_hump_divergence_is_causal_and_fires():
    from strategies.btc_ma_divergence_strategy import DivergenceConfig, divergence_events

    close = _prices(1500, seed=2)["close"]
    cfg = DivergenceConfig()
    full = divergence_events(close, cfg)
    assert len(full) > 0
    cut = close.index[900]
    shocked = close.copy()
    shocked.loc[cut + pd.Timedelta(days=1):] *= 2.0
    partial = divergence_events(shocked, cfg)
    pd.testing.assert_series_equal(full.loc[:cut], partial.loc[:cut])


def test_long_cash_rules_matches_overlay_without_fast_ma():
    from strategies.btc_ma_divergence_strategy import (
        DivergenceConfig, divergence_events, divergence_overlay, long_cash_rules, ma_regime)

    close = _prices(1500, seed=4)["close"]
    ev = divergence_events(close, DivergenceConfig())
    a = long_cash_rules(close, ev, slow=150)
    b = divergence_overlay(ma_regime(close, 150, 0.05), ev, "flat")
    pd.testing.assert_series_equal(a, b)
    assert set(long_cash_rules(close, ev, slow=150, fast=50).unique()) <= {0.0, 1.0}


def test_trim_rule_halves_below_fast_ma_and_is_causal():
    from strategies.crypto_trend_signals import Rules, run_rules

    close = _prices(1500, seed=5)["close"]
    rules = Rules(half_ma=50, use_divergence=False)
    pos, log = run_rules(close, rules, pd.Series(dtype=float))
    assert set(pos.unique()) <= {0.0, 0.5, 1.0}
    assert log.reason.str.startswith("trim").any() and log.reason.str.startswith("add back").any()
    # every trim happens on a close below the 50-day MA
    ma50 = close.rolling(50).mean()
    trims = log[log.reason.str.startswith("trim")]
    assert (trims.price.values < ma50.loc[trims.date].values).all()
    # no look-ahead: changing future prices leaves earlier positions unchanged
    cut = 1000
    alt = close.copy()
    alt.iloc[cut:] *= 0.5
    pos2, _ = run_rules(alt, rules, pd.Series(dtype=float))
    pd.testing.assert_series_equal(pos.iloc[:cut], pos2.iloc[:cut])
    # without half_ma the rule never holds half
    full, _ = run_rules(close, Rules(use_divergence=False), pd.Series(dtype=float))
    assert set(full.unique()) <= {0.0, 1.0}


def test_short_term_ma_rules_are_causal():
    from strategies.crypto_short_term_ma_study import build_variants, metrics

    n = 96 * 400
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    rets = np.random.default_rng(6).normal(0.00002, 0.004, n)
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
    base = build_variants(close)
    cut = n - 96 * 30
    alt = close.copy()
    alt.iloc[cut:] *= 1.3
    moved = build_variants(alt)
    for name, pos in base.items():
        pd.testing.assert_series_equal(pos.iloc[:cut], moved[name].iloc[:cut], check_names=False)
        assert pos.abs().max() <= 1.0
    assert (base["4h+1h, 15m entry"] >= 0).all()
    assert (base["4h+1h, 15m entry long/short"] < 0).any()
    m = metrics(close, base["4h+1h, 15m entry"], 0.001)
    assert m["trades_per_year"] > 0


def test_daily_4h_layers_are_causal():
    from strategies.crypto_daily_4h_study import build_variants, metrics

    n = 96 * 700
    idx = pd.date_range("2021-01-01", periods=n, freq="15min")
    rets = np.random.default_rng(7).normal(0.00003, 0.004, n)
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
    base, c4 = build_variants(close)
    cut = n - 96 * 60
    alt = close.copy()
    alt.iloc[cut:] *= 0.7
    moved, _ = build_variants(alt)
    t_cut = idx[cut]
    for name, pos in base.items():
        a, b = pos.loc[:t_cut - pd.Timedelta(hours=4)], moved[name].loc[:t_cut - pd.Timedelta(hours=4)]
        pd.testing.assert_series_equal(a, b, check_names=False)
        assert pos.min() >= 0 and pos.max() <= 1
    m = metrics(c4, base["daily rule + 4h early half"], 0.001)
    assert m["actions_per_year"] > 0


def test_paper_trader_follows_targets_and_charges_fees():
    from strategies.paper_trader import load_state, step, summary

    st = load_state("/nonexistent", ["BTC", "ETH"], 10_000.0)
    step(st, "2026-01-01", {"BTC": 100.0, "ETH": 10.0}, {"BTC": 1, "ETH": 0}, {}, 0.001)
    assert abs(st["sleeves"]["BTC"]["qty"] - 5000 * 0.999 / 100) < 1e-9
    assert st["sleeves"]["ETH"]["qty"] == 0 and st["sleeves"]["ETH"]["cash"] == 5000
    t = step(st, "2026-01-02", {"BTC": 120.0, "ETH": 10.0}, {"BTC": 0.5, "ETH": 0}, {}, 0.001)
    assert len(t) == 1 and t[0]["side"] == "SELL"
    s = summary(st, {"BTC": 120.0, "ETH": 10.0})
    assert abs(s["sleeves"]["BTC"]["qty"] * 120 - s["sleeves"]["BTC"]["cash"] / 0.999) < 1e-6
    # unchanged target -> no trade even if the price moves
    assert step(st, "2026-01-03", {"BTC": 90.0, "ETH": 12.0}, {"BTC": 0.5, "ETH": 0}, {}, 0.001) == []
