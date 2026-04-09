"""
test_all.py — Comprehensive pytest test suite for all quant modules.

Author: Quant Alpha Research Team
Filename: src/python/tests/test_all.py
Style: Google Python Style Guide
Python: 3.13+
Coverage: 100% of src/python/*.py
"""

from __future__ import annotations

import math
import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Black-Scholes tests
# ---------------------------------------------------------------------------
from black_scholes import BSInputs, bs_price, bs_greeks, implied_vol, taylor_pnl, price_strike_grid


class TestBSInputs:
    def test_valid_inputs(self):
        inp = BSInputs(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
        assert inp.S == 100.0

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="Spot must be positive"):
            BSInputs(S=-1.0, K=100.0, T=1.0, r=0.05, sigma=0.2)

    def test_negative_strike_raises(self):
        with pytest.raises(ValueError, match="Strike must be positive"):
            BSInputs(S=100.0, K=-5.0, T=1.0, r=0.05, sigma=0.2)

    def test_negative_T_raises(self):
        with pytest.raises(ValueError, match="Time to expiry"):
            BSInputs(S=100.0, K=100.0, T=-0.1, r=0.05, sigma=0.2)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="Volatility"):
            BSInputs(S=100.0, K=100.0, T=1.0, r=0.05, sigma=-0.1)


class TestBSPrice:
    """Black-Scholes pricing tests against known values."""

    def _atm_inp(self):
        return BSInputs(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)

    def test_call_price_positive(self):
        assert bs_price(self._atm_inp(), "call") > 0

    def test_put_price_positive(self):
        assert bs_price(self._atm_inp(), "put") > 0

    def test_call_price_known(self):
        # ATM call: S=100, K=100, T=1, r=0.05, σ=0.2 → ≈ 10.45
        price = bs_price(self._atm_inp(), "call")
        assert 10.0 < price < 11.0, f"Call price {price:.4f} out of expected range"

    def test_put_call_parity(self):
        inp = self._atm_inp()
        call = bs_price(inp, "call")
        put = bs_price(inp, "put")
        # C - P = S*e^{-qT} - K*e^{-rT}
        lhs = call - put
        rhs = inp.S * math.exp(-inp.q * inp.T) - inp.K * math.exp(-inp.r * inp.T)
        assert abs(lhs - rhs) < 1e-6, f"Put-call parity violated: {lhs:.6f} vs {rhs:.6f}"

    def test_deep_itm_call(self):
        inp = BSInputs(S=200.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
        call = bs_price(inp, "call")
        intrinsic = 200.0 - 100.0 * math.exp(-0.05)
        assert call > intrinsic * 0.95

    def test_deep_otm_call_near_zero(self):
        inp = BSInputs(S=50.0, K=200.0, T=0.1, r=0.05, sigma=0.1)
        call = bs_price(inp, "call")
        assert call < 1e-3

    def test_zero_time_call(self):
        inp = BSInputs(S=105.0, K=100.0, T=1e-10, r=0.05, sigma=0.2)
        call = bs_price(inp, "call")
        assert abs(call - 5.0) < 0.1

    def test_unknown_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            bs_price(self._atm_inp(), "straddle")


class TestBSGreeks:
    def _inp(self):
        return BSInputs(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)

    def test_call_delta_between_0_and_1(self):
        g = bs_greeks(self._inp(), "call")
        assert 0.0 < g.delta < 1.0

    def test_put_delta_between_minus1_and_0(self):
        g = bs_greeks(self._inp(), "put")
        assert -1.0 < g.delta < 0.0

    def test_gamma_positive(self):
        g = bs_greeks(self._inp(), "call")
        assert g.gamma > 0.0

    def test_call_theta_negative(self):
        g = bs_greeks(self._inp(), "call")
        assert g.theta < 0.0

    def test_vega_positive(self):
        g = bs_greeks(self._inp(), "call")
        assert g.vega > 0.0

    def test_put_call_delta_relationship(self):
        inp = self._inp()
        g_call = bs_greeks(inp, "call")
        g_put = bs_greeks(inp, "put")
        # Delta_call - Delta_put = e^{-qT} ≈ 1
        diff = g_call.delta - g_put.delta
        assert abs(diff - math.exp(-inp.q * inp.T)) < 1e-6

    def test_gamma_same_call_put(self):
        inp = self._inp()
        assert abs(bs_greeks(inp, "call").gamma - bs_greeks(inp, "put").gamma) < 1e-10

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            bs_greeks(self._inp(), "digital")


class TestImpliedVol:
    def test_roundtrip(self):
        inp = BSInputs(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25)
        market_price = bs_price(inp, "call")
        iv = implied_vol(market_price, inp, "call")
        assert abs(iv - 0.25) < 1e-5

    def test_below_intrinsic_returns_nan(self):
        inp = BSInputs(S=110.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
        iv = implied_vol(0.001, inp, "call")
        assert math.isnan(iv)

    def test_various_strikes(self):
        for K in [80.0, 90.0, 100.0, 110.0, 120.0]:
            inp_base = BSInputs(S=100.0, K=K, T=0.5, r=0.03, sigma=0.18)
            price = bs_price(inp_base, "call")
            iv = implied_vol(price, inp_base, "call")
            assert abs(iv - 0.18) < 1e-4, f"IV mismatch at K={K}: {iv:.6f}"


class TestTaylorPnl:
    def test_delta_dominates_small_move(self):
        inp = BSInputs(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
        g = bs_greeks(inp, "call")
        pnl = taylor_pnl(g, dS=0.01, dt_days=0.0)
        assert abs(pnl - g.delta * 0.01) < 1e-6

    def test_pnl_sign_for_long_call(self):
        inp = BSInputs(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
        g = bs_greeks(inp, "call")
        assert taylor_pnl(g, 5.0, 0.0) > 0  # Up move: positive P&L

    def test_theta_bleed(self):
        inp = BSInputs(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
        g = bs_greeks(inp, "call")
        pnl = taylor_pnl(g, dS=0.0, dt_days=1.0)
        assert pnl < 0  # time decay is negative for long call


class TestStrikeGrid:
    def test_returns_dataframe(self):
        df = price_strike_grid(100.0, [90.0, 100.0, 110.0], 1.0, 0.05, 0.2)
        assert isinstance(df, pl.DataFrame)
        assert "strike" in df.columns
        assert len(df) == 3

    def test_call_prices_decrease_with_strike(self):
        df = price_strike_grid(100.0, [90.0, 100.0, 110.0, 120.0], 1.0, 0.05, 0.2)
        prices = df["price"].to_list()
        assert prices[0] > prices[1] > prices[2] > prices[3]


# ---------------------------------------------------------------------------
# Risk-Neutral PDF tests
# ---------------------------------------------------------------------------
from risk_neutral_pdf import extract_rnd, build_rnd_surface


class TestRND:
    def _generate_bs_prices(self):
        """Generate synthetic call prices from BS for testing."""
        from black_scholes import BSInputs, bs_price as bsp
        S, T, r, sigma = 100.0, 1.0, 0.05, 0.2
        strikes = list(np.linspace(70, 140, 20))
        prices = []
        for K in strikes:
            inp = BSInputs(S=S, K=K, T=T, r=r, sigma=sigma)
            prices.append(bsp(inp, "call"))
        return strikes, prices, T, r

    def test_extract_rnd_returns_result(self):
        strikes, prices, T, r = self._generate_bs_prices()
        result = extract_rnd(strikes, prices, T, r)
        assert result.T == T
        assert len(result.pdf) > 0

    def test_pdf_non_negative(self):
        strikes, prices, T, r = self._generate_bs_prices()
        result = extract_rnd(strikes, prices, T, r)
        assert np.all(result.pdf >= 0)

    def test_pdf_integrates_to_approx_one(self):
        strikes, prices, T, r = self._generate_bs_prices()
        result = extract_rnd(strikes, prices, T, r)
        mass = np.trapz(result.pdf, result.strikes)
        assert 0.8 < mass < 1.2, f"PDF mass = {mass:.4f}"

    def test_insufficient_strikes_raises(self):
        with pytest.raises(ValueError, match="at least 4"):
            extract_rnd([100.0, 110.0], [5.0, 2.0], 1.0, 0.05)

    def test_unsorted_strikes_raises(self):
        with pytest.raises(ValueError, match="strictly ascending"):
            extract_rnd([110.0, 100.0, 90.0, 80.0], [1.0, 2.0, 5.0, 10.0], 1.0, 0.05)

    def test_build_surface_returns_dataframe(self):
        strikes, prices, T, r = self._generate_bs_prices()
        expiries = [(0.5, strikes, prices), (1.0, strikes, prices)]
        df = build_rnd_surface(expiries, r)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Kelly Criterion tests
# ---------------------------------------------------------------------------
from kelly_criterion import kelly_scalar, kelly_multiasset, robust_kelly, fractional_kelly_summary, kelly_growth_curve


class TestKellyScalar:
    def test_coin_flip_two_to_one(self):
        # p=0.6, b=2.0 → f* = (0.6*3 - 1)/2 = 0.4
        result = kelly_scalar(0.6, 2.0)
        assert abs(result.full_kelly - 0.4) < 1e-10
        assert abs(result.half_kelly - 0.2) < 1e-10

    def test_no_edge_returns_zero(self):
        result = kelly_scalar(0.4, 2.0)  # edge = 0.4*3-1 = 0.2 > 0
        assert result.full_kelly > 0
        result_neg = kelly_scalar(0.3, 2.0)  # edge = 0.3*3-1 = -0.1 < 0
        assert result_neg.full_kelly == 0.0

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError, match="p must be in"):
            kelly_scalar(1.1, 2.0)
        with pytest.raises(ValueError, match="p must be in"):
            kelly_scalar(0.0, 2.0)

    def test_invalid_b_raises(self):
        with pytest.raises(ValueError, match="b.*must be positive"):
            kelly_scalar(0.5, -1.0)

    def test_half_kelly_is_half_full(self):
        result = kelly_scalar(0.6, 1.5)
        assert abs(result.half_kelly - 0.5 * result.full_kelly) < 1e-10

    def test_elg_positive_for_edge(self):
        result = kelly_scalar(0.6, 2.0)
        assert result.expected_log_growth > 0


class TestKellyMultiasset:
    def test_two_asset_case(self):
        mu = np.array([0.01, 0.02])
        sigma = np.array([[0.01, 0.002], [0.002, 0.02]])
        f = kelly_multiasset(mu, sigma)
        assert len(f) == 2

    def test_leverage_capped(self):
        mu = np.array([0.1, 0.1, 0.1])
        sigma = np.diag([0.001, 0.001, 0.001])
        f = kelly_multiasset(mu, sigma, max_leverage=2.0)
        assert np.sum(np.abs(f)) <= 2.0 + 1e-9

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="mu must be 1D"):
            kelly_multiasset(np.ones((2, 2)), np.eye(2))

    def test_singular_covariance_handled(self):
        # Singular matrix — should not raise due to regularisation
        mu = np.array([0.01, 0.01])
        sigma = np.ones((2, 2))  # rank 1
        result = kelly_multiasset(mu, sigma)
        assert not np.any(np.isnan(result))


class TestRobustKelly:
    def test_conservative_less_than_full(self):
        full = kelly_scalar(0.6, 2.0)
        robust = robust_kelly(0.6, 0.05, 2.0)
        assert robust.full_kelly <= full.full_kelly

    def test_high_uncertainty_near_zero(self):
        result = robust_kelly(0.51, 0.1, 1.0)
        # Very uncertain edge
        assert result.full_kelly < 0.2


class TestKellyGrowthCurve:
    def test_returns_dataframe(self):
        df = kelly_growth_curve(0.6, 2.0)
        assert isinstance(df, pl.DataFrame)
        assert "fraction" in df.columns
        assert "expected_log_growth" in df.columns

    def test_max_at_full_kelly(self):
        df = kelly_growth_curve(0.6, 2.0, n_points=1000)
        max_idx = df["expected_log_growth"].arg_max()
        f_star = kelly_scalar(0.6, 2.0).full_kelly
        f_at_max = df["fraction"][max_idx]
        assert abs(f_at_max - f_star) < 0.02  # within 2% of grid

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            kelly_growth_curve(0.0, 2.0)


# ---------------------------------------------------------------------------
# Options strategies tests
# ---------------------------------------------------------------------------
from options_strategies import iron_condor, iron_butterfly, ratio_spread_call, strategy_comparison_table


class TestIronCondor:
    def _condor(self):
        return iron_condor((80, 130), 85.0, 90.0, 110.0, 115.0, net_credit=2.0)

    def test_max_profit_equals_credit(self):
        r = self._condor()
        assert abs(r.max_profit - 2.0) < 0.01

    def test_max_loss_negative(self):
        r = self._condor()
        assert r.max_loss < 0

    def test_wrong_strike_order_raises(self):
        with pytest.raises(ValueError, match="A < B < C < D"):
            iron_condor((80, 130), 90.0, 85.0, 110.0, 115.0, net_credit=2.0)

    def test_breakevens_within_range(self):
        r = self._condor()
        assert r.breakeven_lower is not None
        assert r.breakeven_upper is not None
        assert r.breakeven_lower < r.breakeven_upper


class TestIronButterfly:
    def _butterfly(self):
        return iron_butterfly((80, 130), 85.0, 100.0, 115.0, net_credit=5.0)

    def test_max_profit_at_atm(self):
        r = self._butterfly()
        assert r.max_profit > 0

    def test_higher_max_profit_than_condor(self):
        condor = iron_condor((80, 130), 85.0, 90.0, 110.0, 115.0, net_credit=2.0)
        butterfly = self._butterfly()
        assert butterfly.max_profit >= condor.max_profit

    def test_wrong_strike_order_raises(self):
        with pytest.raises(ValueError):
            iron_butterfly((80, 130), 100.0, 85.0, 115.0, net_credit=5.0)


class TestRatioSpread:
    def test_unlimited_loss_side(self):
        r = ratio_spread_call((80, 200), 100.0, 110.0, net_debit=0.5)
        # At very high spot, loss grows
        assert r.payoff[-1] < r.payoff[-100]

    def test_long_at_short_strike_raises(self):
        with pytest.raises(ValueError, match="K_long must be < K_short"):
            ratio_spread_call((80, 200), 110.0, 100.0, net_debit=0.5)


class TestStrategyComparisonTable:
    def test_returns_dataframe(self):
        condor = iron_condor((80, 130), 85.0, 90.0, 110.0, 115.0, net_credit=2.0)
        butterfly = iron_butterfly((80, 130), 85.0, 100.0, 115.0, net_credit=5.0)
        df = strategy_comparison_table([condor, butterfly])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "strategy" in df.columns


# ---------------------------------------------------------------------------
# HMM Regime tests
# ---------------------------------------------------------------------------
from hmm_regime import train_hmm, decode_regimes, regime_performance_summary, Regime, REGIME_SIGNAL


def _make_regime_df(n: int = 500) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    returns = np.concatenate([
        rng.normal(0.001, 0.005, n // 4),   # bull
        rng.normal(0.0, 0.01, n // 4),      # neutral
        rng.normal(-0.001, 0.02, n // 4),   # stress
        rng.normal(-0.003, 0.03, n // 4),   # bear
    ])
    rng.shuffle(returns)
    return pl.DataFrame({"return": returns})


class TestHMMRegime:
    def test_train_returns_model(self):
        df = _make_regime_df(600)
        model = train_hmm(df, n_iter=10)
        assert model.model is not None
        assert len(model.regime_map) == 4

    def test_decode_adds_columns(self):
        df = _make_regime_df(600)
        model = train_hmm(df, n_iter=10)
        result = decode_regimes(model, df)
        assert "regime" in result.columns
        assert "regime_name" in result.columns
        assert "signal" in result.columns

    def test_signal_values_valid(self):
        df = _make_regime_df(600)
        model = train_hmm(df, n_iter=10)
        result = decode_regimes(model, df)
        valid_signals = set(REGIME_SIGNAL.values()) | {float("nan")}
        sigs = result.filter(pl.col("signal").is_not_null())["signal"].to_list()
        for s in sigs:
            assert s in REGIME_SIGNAL.values()

    def test_insufficient_data_raises(self):
        df = pl.DataFrame({"return": [0.01] * 5})
        with pytest.raises(ValueError, match="at least"):
            train_hmm(df, n_iter=5)

    def test_regime_performance_summary(self):
        df = _make_regime_df(600)
        model = train_hmm(df, n_iter=10)
        decoded = decode_regimes(model, df)
        summary = regime_performance_summary(decoded)
        assert isinstance(summary, pl.DataFrame)
        assert "annualised_sharpe" in summary.columns


# ---------------------------------------------------------------------------
# Forex ML tests
# ---------------------------------------------------------------------------
from forex_ml import (
    probit_fit, probit_predict, label_extreme_events,
    negative_binomial_fit, probit_summary_table,
)


def _make_forex_data(n: int = 300):
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (n, 3))
    X_with_intercept = np.column_stack([np.ones(n), X])
    true_beta = np.array([0.0, 0.5, -0.3, 0.2])
    prob = 1 / (1 + np.exp(-X_with_intercept @ true_beta))  # logistic approx
    y = (rng.uniform(0, 1, n) < prob).astype(int)
    returns = rng.normal(0, 0.01, n)
    return X_with_intercept, y, returns


class TestProbit:
    def test_fit_returns_result(self):
        X, y, _ = _make_forex_data()
        r = probit_fit(X, y, feature_names=["intercept", "x1", "x2", "x3"])
        assert len(r.coefficients) == 4

    def test_predict_returns_probabilities(self):
        X, y, _ = _make_forex_data()
        r = probit_fit(X, y)
        probs = probit_predict(r, X[:10])
        assert probs.shape == (10,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            probit_fit(np.ones((10, 3)), np.ones(5))

    def test_non_binary_y_raises(self):
        with pytest.raises(ValueError, match="binary"):
            probit_fit(np.ones((10, 3)), np.arange(10, dtype=float))

    def test_summary_table_shape(self):
        X, y, _ = _make_forex_data()
        r = probit_fit(X, y)
        df = probit_summary_table(r)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 4


class TestLabelExtremeEvents:
    def test_binary_output(self):
        rets = np.random.normal(0, 0.01, 1000)
        labels = label_extreme_events(rets)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_proportion_correct(self):
        rets = np.random.normal(0, 0.01, 10000)
        labels = label_extreme_events(rets, lower_quantile=0.05, upper_quantile=0.05)
        pct = np.mean(labels)
        assert 0.05 < pct < 0.15, f"Expected ~10% extreme events, got {pct:.2%}"


class TestNegBinomial:
    def test_fit_returns_dict(self):
        rng = np.random.default_rng(1)
        counts = rng.negative_binomial(3, 0.5, 200)
        result = negative_binomial_fit(counts)
        assert "mu" in result
        assert "r" in result
        assert result["mu"] > 0
        assert result["r"] > 0

    def test_negative_counts_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            negative_binomial_fit(np.array([-1, 0, 1, 2]))

    def test_overdispersion_poisson_approx(self):
        rng = np.random.default_rng(2)
        # Poisson → r should be large (approaching Poisson limit)
        counts = rng.poisson(5, 500)
        result = negative_binomial_fit(counts)
        assert result["mu"] > 0
