"""
kelly_criterion.py — Kelly Criterion and Half-Kelly position sizing engine.

Author: Quant Alpha Research Team
Filename: src/python/kelly_criterion.py
Style: Google Python Style Guide
Python: 3.13+
Libs: numpy 2.x, scipy 1.14+, polars 1.x, cvxpy 1.5+

Algorithm:
  1. Scalar Kelly: f* = (p(b+1) - 1) / b
  2. Half-Kelly: f_half = 0.5 * f*
  3. Continuous Kelly (log-utility): E[log W] maximisation
  4. Multi-asset Kelly: f_vec = Σ⁻¹ μ (with regularisation)
  5. Fractional Kelly grid: compute growth/variance tradeoff curve
  6. Kelly under parameter uncertainty (robust Kelly)

Time Complexity:  O(1) scalar; O(N³) multi-asset (matrix inversion).
Space Complexity: O(N²) for covariance matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar, minimize

logger = logging.getLogger(__name__)

_MAX_KELLY_FRAC: Final[float] = 1.0   # Cap at 100% to prevent leverage > 1x per signal
_MIN_EDGE: Final[float] = 1e-10


@dataclass(slots=True, frozen=True)
class KellyResult:
    """Kelly sizing result for a single bet/signal."""
    full_kelly: float
    half_kelly: float
    expected_log_growth: float
    variance_of_returns: float
    edge: float       # p(b+1) - 1
    odds: float       # b


def kelly_scalar(p: float, b: float) -> KellyResult:
    """Compute Kelly fraction for binary bet.

    Formula: f* = (p(b+1) - 1) / b

    Args:
        p: Probability of win (0 < p < 1).
        b: Net odds (win b units per 1 unit risked).

    Returns:
        KellyResult with full/half Kelly fractions.

    Raises:
        ValueError: If p or b are out of valid range.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")
    if b <= 0:
        raise ValueError(f"b (net odds) must be positive, got {b}")

    edge = p * (b + 1.0) - 1.0
    f_star = edge / b

    if f_star <= 0:
        logger.debug("No edge: f*=%.4f, returning zero bet.", f_star)
        return KellyResult(
            full_kelly=0.0, half_kelly=0.0,
            expected_log_growth=0.0, variance_of_returns=0.0,
            edge=edge, odds=b,
        )

    f_star = min(f_star, _MAX_KELLY_FRAC)
    f_half = 0.5 * f_star

    # Expected log growth at full Kelly
    elg = p * np.log(1.0 + b * f_star) + (1.0 - p) * np.log(1.0 - f_star)

    # Variance of log returns at full Kelly
    ev = p * np.log(1.0 + b * f_star) ** 2 + (1.0 - p) * np.log(1.0 - f_star) ** 2
    var_returns = ev - elg**2

    return KellyResult(
        full_kelly=f_star,
        half_kelly=f_half,
        expected_log_growth=float(elg),
        variance_of_returns=float(var_returns),
        edge=float(edge),
        odds=float(b),
    )


def kelly_growth_curve(p: float, b: float, n_points: int = 200) -> pl.DataFrame:
    """Compute expected log-growth vs betting fraction curve.

    Args:
        p: Win probability.
        b: Net odds.
        n_points: Number of fraction grid points.

    Returns:
        Polars DataFrame: fraction, expected_log_growth, variance.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}")

    fracs = np.linspace(0.0, 1.0 - 1e-6, n_points)
    elg_values = p * np.log(1.0 + b * fracs) + (1.0 - p) * np.log(1.0 - fracs)
    ev2 = p * np.log(1.0 + b * fracs) ** 2 + (1.0 - p) * np.log(1.0 - fracs) ** 2
    var_values = ev2 - elg_values**2

    return pl.DataFrame({
        "fraction": fracs,
        "expected_log_growth": elg_values,
        "variance": var_values,
    })


def kelly_multiasset(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = 1.0,
    regularisation: float = 1e-4,
    max_leverage: float = 2.0,
) -> np.ndarray:
    """Compute multi-asset Kelly allocation: f* = (1/λ) Σ⁻¹ μ.

    This is equivalent to the mean-variance tangency portfolio scaled
    by risk aversion λ (= 1 for pure Kelly / log-utility).

    Args:
        mu: Expected returns vector (N,).
        sigma: Covariance matrix (N, N).
        risk_aversion: λ, scales position size (default=1 for pure Kelly).
        regularisation: Ridge term added to diagonal for numerical stability.
        max_leverage: Cap total portfolio leverage (L1 norm of weights).

    Returns:
        Optimal fractional allocation vector (N,).

    Raises:
        ValueError: If inputs have incompatible shapes.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if mu.ndim != 1:
        raise ValueError(f"mu must be 1D, got shape {mu.shape}")
    N = len(mu)
    if sigma.shape != (N, N):
        raise ValueError(f"sigma must be ({N},{N}), got {sigma.shape}")

    # Regularise covariance matrix
    sigma_reg = sigma + regularisation * np.eye(N)

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(sigma_reg)
    if np.any(eigvals <= 0):
        logger.warning("Covariance not positive-definite after regularisation; clipping eigenvalues.")
        sigma_reg += (abs(eigvals.min()) + 1e-6) * np.eye(N)

    sigma_inv = np.linalg.inv(sigma_reg)
    f_kelly = (1.0 / risk_aversion) * sigma_inv @ mu

    # Cap leverage
    leverage = np.sum(np.abs(f_kelly))
    if leverage > max_leverage:
        logger.debug("Scaling down Kelly allocation from leverage=%.2f to %.2f", leverage, max_leverage)
        f_kelly = f_kelly * (max_leverage / leverage)

    return f_kelly


def robust_kelly(
    p_estimate: float,
    p_uncertainty: float,
    b: float,
    confidence: float = 0.95,
) -> KellyResult:
    """Compute conservative Kelly under estimation uncertainty.

    Uses worst-case p within confidence interval:
    p_conservative = p_estimate - z_{conf} * p_uncertainty

    Args:
        p_estimate: Point estimate of win probability.
        p_uncertainty: Standard error of p estimate.
        b: Net odds.
        confidence: One-sided confidence level (e.g. 0.95 → z=1.645).

    Returns:
        KellyResult using conservative p.
    """
    from scipy.stats import norm as sp_norm
    z = sp_norm.ppf(confidence)
    p_conservative = max(p_estimate - z * p_uncertainty, _MIN_EDGE)
    p_conservative = min(p_conservative, 1.0 - _MIN_EDGE)
    logger.info(
        "Robust Kelly: p_est=%.4f ± %.4f → p_conservative=%.4f",
        p_estimate, p_uncertainty, p_conservative,
    )
    return kelly_scalar(p_conservative, b)


def fractional_kelly_summary(p: float, b: float) -> pl.DataFrame:
    """Summarise growth/risk tradeoff for fractional Kelly levels.

    Args:
        p: Win probability.
        b: Net odds.

    Returns:
        Polars DataFrame comparing quarter/half/full Kelly.
    """
    result = kelly_scalar(p, b)
    if result.full_kelly <= 0:
        return pl.DataFrame({"fraction_label": [], "f": [], "elg": [], "variance_reduction_pct": []})

    rows = []
    for label, frac_mult in [("quarter_kelly", 0.25), ("half_kelly", 0.5),
                              ("three_quarter_kelly", 0.75), ("full_kelly", 1.0)]:
        f = result.full_kelly * frac_mult
        elg = p * np.log(1.0 + b * f) + (1.0 - p) * np.log(1.0 - f)
        ev2 = p * np.log(1.0 + b * f) ** 2 + (1.0 - p) * np.log(1.0 - f) ** 2
        var_f = ev2 - elg**2
        var_reduction = 100.0 * (1.0 - var_f / max(result.variance_of_returns, 1e-12))
        rows.append({
            "fraction_label": label,
            "f": float(f),
            "expected_log_growth": float(elg),
            "variance": float(var_f),
            "variance_reduction_vs_full_pct": float(var_reduction),
            "growth_pct_of_max": 100.0 * float(elg) / max(result.expected_log_growth, 1e-12),
        })

    return pl.DataFrame(rows)
