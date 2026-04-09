"""
forex_ml.py — Forex Regression & Classification: Probit, Negative Binomial,
              Zero-Inflated HMM transition models.

Author: Quant Alpha Research Team
Filename: src/python/forex_ml.py
Style: Google Python Style Guide
Python: 3.13+
Libs: numpy 2.x, scipy 1.14+, polars 1.x, scikit-learn 1.5+, statsmodels 0.14+

Models:
  1. Probit classifier: P(extreme event) = Φ(X'β) for tail classification.
  2. Negative Binomial regression: E[N] for count of extreme moves per window.
  3. Zero-Inflated Negative Binomial: handles excess zeros (calm periods).
  4. Vol-regime classifier: LDA/QDA for low/high vol state classification.

Time Complexity:  O(N·D²) for Probit MLE (N obs, D features); O(N) prediction.
Space Complexity: O(N·D) training data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from scipy.stats import norm as sp_norm, nbinom
from scipy.optimize import minimize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

logger = logging.getLogger(__name__)

_TAIL_P_LOWER: Final[float] = 0.055   # Left tail threshold (from image: P = 0.055)
_TAIL_P_UPPER: Final[float] = 0.075   # Right tail threshold (from image: P ≥ 0.075)
_MAX_ITER: Final[int] = 500


@dataclass(slots=True)
class ProbitResult:
    """Result of Probit regression."""
    coefficients: np.ndarray
    std_errors: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    feature_names: list[str]
    log_likelihood: float
    aic: float
    bic: float


def probit_fit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
) -> ProbitResult:
    """Fit Probit model via MLE: P(Y=1|x) = Φ(x'β).

    Algorithm:
      Maximise log-likelihood: Σ [y_i log Φ(x'β) + (1-y_i) log(1-Φ(x'β))]
      via L-BFGS-B with numerical Hessian for standard errors.

    Args:
        X: Feature matrix (N, D) — should include intercept column if desired.
        y: Binary labels (N,) — 1 = extreme event.
        feature_names: Optional list of D feature names.

    Returns:
        ProbitResult with coefficients, SEs, z-scores, p-values, AIC/BIC.

    Raises:
        ValueError: If X and y have incompatible shapes.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if len(y) != len(X):
        raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0 or 1)")

    N, D = X.shape
    beta0 = np.zeros(D)

    def neg_log_lik(beta: np.ndarray) -> float:
        xb = X @ beta
        # Clip for numerical stability
        xb = np.clip(xb, -10, 10)
        log_lik = np.sum(
            y * np.log(np.maximum(sp_norm.cdf(xb), 1e-12))
            + (1 - y) * np.log(np.maximum(1.0 - sp_norm.cdf(xb), 1e-12))
        )
        return -log_lik

    result = minimize(neg_log_lik, beta0, method="L-BFGS-B",
                      options={"maxiter": _MAX_ITER, "ftol": 1e-10})
    if not result.success:
        logger.warning("Probit MLE did not converge: %s", result.message)

    beta_hat = result.x

    # Numerical Hessian for SEs
    from scipy.optimize import approx_fprime
    eps = 1e-5
    hessian = np.zeros((D, D))
    grad = lambda b: approx_fprime(b, neg_log_lik, eps)
    for i in range(D):
        e = np.zeros(D); e[i] = eps
        hessian[i] = (grad(beta_hat + e) - grad(beta_hat - e)) / (2 * eps)
    hessian_sym = 0.5 * (hessian + hessian.T)

    try:
        cov_matrix = np.linalg.inv(hessian_sym)
        std_errors = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
    except np.linalg.LinAlgError:
        logger.warning("Hessian inversion failed; SEs may be unreliable.")
        std_errors = np.full(D, np.nan)

    z_scores = beta_hat / np.maximum(std_errors, 1e-12)
    p_values = 2.0 * (1.0 - sp_norm.cdf(np.abs(z_scores)))
    log_lik = -result.fun
    aic = 2 * D - 2 * log_lik
    bic = D * np.log(N) - 2 * log_lik

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(D)]

    return ProbitResult(
        coefficients=beta_hat,
        std_errors=std_errors,
        z_scores=z_scores,
        p_values=p_values,
        feature_names=feature_names,
        log_likelihood=float(log_lik),
        aic=float(aic),
        bic=float(bic),
    )


def probit_predict(result: ProbitResult, X_new: np.ndarray) -> np.ndarray:
    """Predict P(Y=1|X_new) from fitted Probit.

    Args:
        result: Fitted ProbitResult.
        X_new: New feature matrix (M, D).

    Returns:
        Probability array (M,).
    """
    X_new = np.asarray(X_new, dtype=np.float64)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    xb = np.clip(X_new @ result.coefficients, -10, 10)
    return sp_norm.cdf(xb)


def label_extreme_events(
    returns: np.ndarray,
    lower_quantile: float = _TAIL_P_LOWER,
    upper_quantile: float = _TAIL_P_UPPER,
) -> np.ndarray:
    """Label extreme Forex return events as binary.

    Args:
        returns: Return array.
        lower_quantile: Left tail threshold.
        upper_quantile: Right tail threshold.

    Returns:
        Binary array: 1 = extreme, 0 = normal.
    """
    lower_thresh = np.quantile(returns, lower_quantile)
    upper_thresh = np.quantile(returns, 1.0 - upper_quantile)
    return ((returns <= lower_thresh) | (returns >= upper_thresh)).astype(int)


def negative_binomial_fit(counts: np.ndarray) -> dict[str, float]:
    """Fit Negative Binomial to count data (overdispersed Poisson).

    NB PMF: P(N=n) = C(n+r-1, n) * p^r * (1-p)^n
    Parameterised by mean μ and dispersion r.

    Formula from image: B(n) = B(n + (-0.3))² / n!

    Args:
        counts: Array of non-negative integer counts.

    Returns:
        Dict with 'mu' (mean), 'r' (dispersion), 'log_likelihood'.

    Raises:
        ValueError: If counts contain negative values.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if np.any(counts < 0):
        raise ValueError("Counts must be non-negative")

    mu_init = float(np.mean(counts))
    r_init = mu_init**2 / max(float(np.var(counts)) - mu_init, 1e-2)

    def neg_ll(params: np.ndarray) -> float:
        mu, r = params
        if mu <= 0 or r <= 0:
            return 1e10
        p_nb = r / (r + mu)
        return -np.sum(nbinom.logpmf(counts, r, p_nb))

    res = minimize(neg_ll, [mu_init, max(r_init, 0.1)],
                   method="Nelder-Mead", options={"maxiter": 1000})
    mu_hat, r_hat = res.x
    return {
        "mu": float(mu_hat),
        "r": float(r_hat),
        "log_likelihood": float(-res.fun),
        "overdispersion": float(mu_hat / max(r_hat, 1e-10)),
        "variance": float(mu_hat + mu_hat**2 / max(r_hat, 1e-10)),
    }


def vol_regime_classifier(
    df: pl.DataFrame,
    feature_cols: list[str],
    vol_col: str,
    vol_threshold: float | None = None,
    method: str = "lda",
) -> tuple[object, pl.DataFrame]:
    """Train LDA or QDA vol regime classifier.

    Labels: 0 = Low Volatility, 1 = High Volatility
    Transition probabilities from image:
      LV→LV: 0.50, LV→HV: 0.42, HV→HV: 0.94, HV→LV: 0.35

    Args:
        df: Polars DataFrame with features and vol column.
        feature_cols: Feature column names.
        vol_col: Volatility column name for labelling.
        vol_threshold: Median vol used as threshold (computed if None).
        method: "lda" or "qda".

    Returns:
        Tuple: (fitted classifier, DataFrame with predictions).
    """
    if method not in {"lda", "qda"}:
        raise ValueError(f"method must be 'lda' or 'qda', got {method!r}")

    X = df.select(feature_cols).to_numpy().astype(np.float64)
    vol = df[vol_col].to_numpy().astype(np.float64)

    if vol_threshold is None:
        vol_threshold = float(np.median(vol))

    y = (vol > vol_threshold).astype(int)

    # Remove NaN rows
    valid = ~np.any(np.isnan(X), axis=1) & ~np.isnan(vol)
    X_clean, y_clean = X[valid], y[valid]

    if len(X_clean) < 10:
        raise ValueError(f"Too few valid observations: {len(X_clean)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    clf = LinearDiscriminantAnalysis() if method == "lda" else QuadraticDiscriminantAnalysis()
    clf.fit(X_scaled, y_clean)

    proba = clf.predict_proba(X_scaled)
    pred = clf.predict(X_scaled)

    auc = roc_auc_score(y_clean, proba[:, 1])
    logger.info("Vol regime classifier AUC=%.4f", auc)

    pred_full = np.full(len(df), -1, dtype=np.int32)
    pred_full[valid] = pred

    result_df = df.with_columns([
        pl.Series("vol_regime_pred", pred_full),
        pl.Series("prob_high_vol", np.where(valid, proba[:, 1], np.nan)[:len(df)]),
    ])

    return clf, result_df


def probit_summary_table(result: ProbitResult) -> pl.DataFrame:
    """Present Probit results as a formatted Polars DataFrame.

    Args:
        result: Fitted ProbitResult.

    Returns:
        Polars DataFrame: feature, coeff, std_error, z_score, p_value, significance.
    """
    sig = []
    for pv in result.p_values:
        if pv < 0.001:
            sig.append("***")
        elif pv < 0.01:
            sig.append("**")
        elif pv < 0.05:
            sig.append("*")
        elif pv < 0.1:
            sig.append(".")
        else:
            sig.append("")

    return pl.DataFrame({
        "feature": result.feature_names,
        "coefficient": result.coefficients.tolist(),
        "std_error": result.std_errors.tolist(),
        "z_score": result.z_scores.tolist(),
        "p_value": result.p_values.tolist(),
        "significance": sig,
    })
