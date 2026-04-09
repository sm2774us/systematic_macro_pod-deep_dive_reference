"""
risk_neutral_pdf.py — Breeden-Litzenberger risk-neutral probability density extraction.

Author: Quant Alpha Research Team
Filename: src/python/risk_neutral_pdf.py
Style: Google Python Style Guide
Python: 3.13+
Libs: numpy 2.x, scipy 1.14+, polars 1.x

Algorithm:
  1. Fit a smooth call-price-vs-strike curve (cubic spline or SVI parameterisation).
  2. Compute second derivative ∂²C/∂K² numerically on the smoothed curve.
  3. Apply Breeden-Litzenberger: PDF(K) = e^{rT} * ∂²C/∂K².
  4. Normalise and compute distribution moments (mean, variance, skew, kurtosis).
  5. Build a surface over multiple expiries T₁ < T₂ < ... < T_n.

Time Complexity:  O(N log N) for spline fit over N strikes; O(N) for numerical differentiation.
Space Complexity: O(N*M) for N strikes × M expiries surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)

_MIN_DENSITY: Final[float] = 0.0
_FINITE_DIFF_H: Final[float] = 0.5  # strike spacing for numerical 2nd derivative


@dataclass(slots=True)
class RNDResult:
    """Result of Breeden-Litzenberger extraction for one expiry."""
    T: float
    strikes: np.ndarray
    pdf: np.ndarray         # normalised risk-neutral PDF
    cdf: np.ndarray         # cumulative
    mean: float
    variance: float
    skewness: float
    excess_kurtosis: float


def _numerical_second_derivative(
    K_grid: np.ndarray,
    C_smooth: np.ndarray,
) -> np.ndarray:
    """Compute ∂²C/∂K² via central finite differences.

    Args:
        K_grid: Uniformly spaced strike grid.
        C_smooth: Smoothed call prices on same grid.

    Returns:
        Second derivative array (same length, endpoints extrapolated).
    """
    h = K_grid[1] - K_grid[0]
    d2C = np.gradient(np.gradient(C_smooth, h), h)
    return d2C


def extract_rnd(
    strikes_obs: list[float],
    call_prices_obs: list[float],
    T: float,
    r: float,
    n_interp: int = 500,
    K_min_factor: float = 0.5,
    K_max_factor: float = 2.0,
) -> RNDResult:
    """Extract risk-neutral density from observed call prices.

    Uses Breeden-Litzenberger: f^Q(K) = e^{rT} * ∂²C/∂K²

    Args:
        strikes_obs: Observed strikes (must be sorted ascending).
        call_prices_obs: Corresponding call prices.
        T: Time to expiry (years).
        r: Risk-free rate.
        n_interp: Number of interpolation points for smooth curve.
        K_min_factor: Lower bound = spot * factor (approx: min(strikes) * factor).
        K_max_factor: Upper bound = spot * factor.

    Returns:
        RNDResult with PDF, CDF, and distribution moments.

    Raises:
        ValueError: If inputs are malformed.
    """
    strikes_arr = np.asarray(strikes_obs, dtype=np.float64)
    prices_arr = np.asarray(call_prices_obs, dtype=np.float64)

    if len(strikes_arr) != len(prices_arr):
        raise ValueError("strikes and call_prices must have same length")
    if len(strikes_arr) < 4:
        raise ValueError("Need at least 4 strikes for cubic spline fit")
    if not np.all(np.diff(strikes_arr) > 0):
        raise ValueError("strikes_obs must be strictly ascending")

    # Fit cubic spline to observed call prices
    cs = CubicSpline(strikes_arr, prices_arr, bc_type="natural")

    # Dense uniform grid for differentiation
    K_min = strikes_arr[0] * K_min_factor
    K_max = strikes_arr[-1] * K_max_factor
    K_grid = np.linspace(K_min, K_max, n_interp)
    C_smooth = cs(K_grid)

    # Clip to prevent negative prices (no-arb enforcement)
    C_smooth = np.maximum(C_smooth, 0.0)

    # Breeden-Litzenberger: PDF = e^{rT} * d²C/dK²
    d2C = _numerical_second_derivative(K_grid, C_smooth)
    pdf_raw = np.exp(r * T) * d2C

    # Enforce non-negativity (numerical errors near boundaries)
    pdf_raw = np.maximum(pdf_raw, _MIN_DENSITY)

    # Normalise via trapezoidal integration
    dk = K_grid[1] - K_grid[0]
    total_mass = np.trapz(pdf_raw, K_grid)
    if total_mass <= 1e-12:
        logger.warning("PDF mass near zero — check input prices (T=%.3f)", T)
        total_mass = 1.0
    pdf_norm = pdf_raw / total_mass

    # CDF
    cdf = np.cumsum(pdf_norm) * dk
    cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)

    # Moments
    mean_rnd = np.trapz(K_grid * pdf_norm, K_grid)
    variance_rnd = np.trapz((K_grid - mean_rnd) ** 2 * pdf_norm, K_grid)
    skewness_rnd = np.trapz(
        ((K_grid - mean_rnd) / max(variance_rnd**0.5, 1e-12)) ** 3 * pdf_norm, K_grid
    )
    kurtosis_rnd = np.trapz(
        ((K_grid - mean_rnd) / max(variance_rnd**0.5, 1e-12)) ** 4 * pdf_norm, K_grid
    ) - 3.0  # excess kurtosis

    return RNDResult(
        T=T,
        strikes=K_grid,
        pdf=pdf_norm,
        cdf=cdf,
        mean=float(mean_rnd),
        variance=float(variance_rnd),
        skewness=float(skewness_rnd),
        excess_kurtosis=float(kurtosis_rnd),
    )


def build_rnd_surface(
    expiry_data: list[tuple[float, list[float], list[float]]],
    r: float,
) -> pl.DataFrame:
    """Build risk-neutral PDF surface across multiple expiries.

    Args:
        expiry_data: List of (T, strikes, call_prices) tuples sorted by T.
        r: Risk-free rate.

    Returns:
        Polars DataFrame with columns:
        T, mean, variance, skewness, excess_kurtosis, implied_vol (approx).
    """
    rows = []
    for T, strikes, prices in expiry_data:
        try:
            result = extract_rnd(strikes, prices, T, r)
            implied_vol_approx = float(np.sqrt(result.variance)) / result.mean if result.mean > 0 else float("nan")
            rows.append({
                "T": T,
                "mean": result.mean,
                "variance": result.variance,
                "skewness": result.skewness,
                "excess_kurtosis": result.excess_kurtosis,
                "implied_vol_approx": implied_vol_approx,
            })
        except (ValueError, RuntimeError) as exc:
            logger.error("Failed RND extraction for T=%.3f: %s", T, exc)

    return pl.DataFrame(rows)
