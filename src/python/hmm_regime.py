"""
hmm_regime.py — Hidden Markov Model regime detection for systematic macro.

Author: Quant Alpha Research Team
Filename: src/python/hmm_regime.py
Style: Google Python Style Guide
Python: 3.13+
Libs: numpy 2.x, scipy 1.14+, polars 1.x, hmmlearn 0.3+

Algorithm:
  1. Feature engineering: z-score returns, vol, skew, carry, momentum.
  2. Train Gaussian HMM via Baum-Welch EM (hmmlearn).
  3. Decode states via Viterbi algorithm.
  4. Online filtering: forward algorithm for real-time regime probability.
  5. Regime labelling heuristic: sort states by mean return.
  6. Regime-conditioned strategy signal.

States: BULL=3, NEUTRAL=2, STRESS=1, BEAR=0 (sorted by mean return)

Time Complexity:  O(T·N²) for Viterbi/forward (T=time steps, N=states).
Space Complexity: O(T·N) for trellis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final

import numpy as np
import polars as pl
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_N_STATES: Final[int] = 4
_MIN_COVAR_DIAG: Final[float] = 1e-3
_RANDOM_SEED: Final[int] = 42


class Regime(IntEnum):
    """Market regime labels sorted by expected return (ascending)."""
    BEAR = 0
    STRESS = 1
    NEUTRAL = 2
    BULL = 3


REGIME_SIGNAL: dict[Regime, float] = {
    Regime.BULL: 1.0,
    Regime.NEUTRAL: 0.5,
    Regime.STRESS: 0.0,
    Regime.BEAR: 0.0,
}


@dataclass(slots=True)
class HMMModel:
    """Trained HMM with regime metadata."""
    model: hmm.GaussianHMM
    scaler: StandardScaler
    regime_map: dict[int, Regime]   # raw state → Regime
    n_features: int
    feature_names: list[str]
    log_likelihood: float = field(default=0.0)


def _engineer_features(
    df: pl.DataFrame,
    return_col: str = "return",
    vol_col: str | None = None,
    lookback: int = 20,
) -> np.ndarray:
    """Engineer features for HMM from price returns.

    Features: rolling mean return, rolling vol, rolling skew,
              momentum (1m, 3m), VIX proxy (realised vol ratio).

    Args:
        df: Polars DataFrame with at least a return column.
        return_col: Name of returns column.
        vol_col: Optional pre-computed vol column.
        lookback: Rolling window for feature computation.

    Returns:
        Feature matrix (T, n_features) as numpy array, NaN rows dropped.
    """
    rets = df[return_col].to_numpy()
    T = len(rets)

    roll_mean = np.full(T, np.nan)
    roll_std = np.full(T, np.nan)
    roll_skew = np.full(T, np.nan)
    mom_1m = np.full(T, np.nan)
    mom_3m = np.full(T, np.nan)

    for i in range(lookback, T):
        window = rets[i - lookback: i]
        roll_mean[i] = np.mean(window)
        roll_std[i] = np.std(window, ddof=1)
        if roll_std[i] > 1e-10:
            # Simple skew
            roll_skew[i] = np.mean(((window - np.mean(window)) / roll_std[i]) ** 3)
        else:
            roll_skew[i] = 0.0

    # Momentum
    for i in range(21, T):
        mom_1m[i] = np.sum(rets[i - 21: i])
    for i in range(63, T):
        mom_3m[i] = np.sum(rets[i - 63: i])

    # Use provided vol if available
    if vol_col is not None and vol_col in df.columns:
        ext_vol = df[vol_col].to_numpy()
    else:
        ext_vol = roll_std

    features = np.column_stack([roll_mean, roll_std, roll_skew, mom_1m, mom_3m, ext_vol])
    valid_mask = ~np.any(np.isnan(features), axis=1)
    return features[valid_mask], np.where(valid_mask)[0]


def train_hmm(
    df: pl.DataFrame,
    return_col: str = "return",
    vol_col: str | None = None,
    n_iter: int = 200,
    lookback: int = 20,
) -> HMMModel:
    """Train a Gaussian HMM on market features.

    Args:
        df: Polars DataFrame with returns (and optionally vol).
        return_col: Returns column name.
        vol_col: Volatility column name.
        n_iter: Maximum Baum-Welch iterations.
        lookback: Feature engineering rolling window.

    Returns:
        Trained HMMModel.

    Raises:
        ValueError: If insufficient data.
    """
    features, _ = _engineer_features(df, return_col, vol_col, lookback)
    n_features = features.shape[1]

    if len(features) < _N_STATES * 10:
        raise ValueError(
            f"Need at least {_N_STATES * 10} valid observations, got {len(features)}"
        )

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Train Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=_N_STATES,
        covariance_type="full",
        n_iter=n_iter,
        tol=1e-4,
        min_covar=_MIN_COVAR_DIAG,
        random_state=_RANDOM_SEED,
        verbose=False,
    )
    model.fit(X)
    log_likelihood = model.score(X)
    logger.info("HMM trained. Log-likelihood=%.4f. AIC=%.2f", log_likelihood, -2*log_likelihood + 2*_N_STATES**2)

    # Map raw states to Regimes by sorting on mean return (feature 0 = rolling mean return)
    raw_means = model.means_[:, 0]  # mean of rolling-mean-return feature per state
    order = np.argsort(raw_means)   # ascending: bear → bull
    regime_map = {int(order[i]): Regime(i) for i in range(_N_STATES)}

    feature_names = ["roll_mean", "roll_std", "roll_skew", "mom_1m", "mom_3m", "vol"]

    return HMMModel(
        model=model,
        scaler=scaler,
        regime_map=regime_map,
        n_features=n_features,
        feature_names=feature_names,
        log_likelihood=float(log_likelihood),
    )


def decode_regimes(
    hmm_model: HMMModel,
    df: pl.DataFrame,
    return_col: str = "return",
    vol_col: str | None = None,
    lookback: int = 20,
) -> pl.DataFrame:
    """Decode Viterbi regime sequence for a DataFrame.

    Args:
        hmm_model: Trained HMMModel.
        df: Polars DataFrame with returns.
        return_col: Returns column name.
        vol_col: Vol column name.
        lookback: Rolling window.

    Returns:
        Input DataFrame with additional columns:
        raw_state, regime, regime_name, signal, and regime probabilities.
    """
    features, valid_idx = _engineer_features(df, return_col, vol_col, lookback)
    X = hmm_model.scaler.transform(features)

    # Viterbi
    log_prob, raw_states = hmm_model.model.decode(X, algorithm="viterbi")

    # Posterior probabilities (forward-backward)
    posteriors = hmm_model.model.predict_proba(X)

    # Map to Regimes
    regimes = [hmm_model.regime_map[int(s)] for s in raw_states]
    regime_names = [r.name for r in regimes]
    signals = [REGIME_SIGNAL[r] for r in regimes]

    # Build result aligned with original df (NaN for warm-up rows)
    T = len(df)
    regime_arr = np.full(T, -1, dtype=np.int32)
    regime_name_arr = ["UNKNOWN"] * T
    signal_arr = np.full(T, np.nan)
    prob_arrays = {f"prob_{Regime(i).name}": np.full(T, np.nan) for i in range(_N_STATES)}

    for local_i, orig_i in enumerate(valid_idx):
        regime_arr[orig_i] = int(regimes[local_i])
        regime_name_arr[orig_i] = regime_names[local_i]
        signal_arr[orig_i] = signals[local_i]
        for j in range(_N_STATES):
            raw_state_j = [k for k, v in hmm_model.regime_map.items() if v == Regime(j)][0]
            prob_arrays[f"prob_{Regime(j).name}"][orig_i] = posteriors[local_i, raw_state_j]

    result_df = df.clone()
    result_df = result_df.with_columns([
        pl.Series("regime", regime_arr),
        pl.Series("regime_name", regime_name_arr),
        pl.Series("signal", signal_arr),
        *[pl.Series(k, v) for k, v in prob_arrays.items()],
    ])

    return result_df


def regime_performance_summary(
    decoded_df: pl.DataFrame,
    return_col: str = "return",
) -> pl.DataFrame:
    """Compute per-regime performance statistics.

    Args:
        decoded_df: Output from decode_regimes().
        return_col: Returns column name.

    Returns:
        Polars DataFrame: regime, count, mean_return, sharpe, annualised_vol.
    """
    valid = decoded_df.filter(pl.col("regime") >= 0)

    rows = []
    for regime_val in range(_N_STATES):
        subset = valid.filter(pl.col("regime") == regime_val)[return_col].to_numpy()
        if len(subset) == 0:
            continue
        mean_ret = float(np.mean(subset))
        vol = float(np.std(subset, ddof=1))
        sharpe = mean_ret / vol * np.sqrt(252) if vol > 1e-10 else 0.0
        rows.append({
            "regime": regime_val,
            "regime_name": Regime(regime_val).name,
            "count": len(subset),
            "mean_daily_return": mean_ret,
            "annualised_vol": vol * np.sqrt(252),
            "annualised_sharpe": sharpe,
        })

    return pl.DataFrame(rows)
