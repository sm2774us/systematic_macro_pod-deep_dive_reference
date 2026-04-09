"""
black_scholes.py — Black-Scholes PDE solver, option pricer, and Greeks engine.

Author: Quant Alpha Research Team
Filename: src/python/black_scholes.py
Style: Google Python Style Guide
Python: 3.13+
Libs: numpy 2.x, scipy 1.14+, polars 1.x

Algorithm:
  1. Compute d1, d2 from BS formula inputs.
  2. Price European calls/puts via closed-form BS.
  3. Compute all first/second-order Greeks analytically.
  4. Provide Taylor-series P&L approximation.
  5. Compute implied vol via Brent's root-finding.

Time Complexity:  O(1) per scalar call; O(N) vectorized over strike grid.
Space Complexity: O(N) for vectorized outputs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import polars as pl
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SQRT_2PI: Final[float] = math.sqrt(2.0 * math.pi)
_MIN_VOL: Final[float] = 1e-8
_MAX_VOL: Final[float] = 10.0
_MIN_T: Final[float] = 1e-10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class BSInputs:
    """Validated inputs for Black-Scholes pricing."""
    S: float   # Spot price
    K: float   # Strike price
    T: float   # Time to expiry (years)
    r: float   # Risk-free rate (annualised, continuous)
    sigma: float  # Implied/historical vol (annualised)
    q: float = 0.0  # Continuous dividend yield

    def __post_init__(self) -> None:
        if self.S <= 0:
            raise ValueError(f"Spot must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike must be positive, got {self.K}")
        if self.T < 0:
            raise ValueError(f"Time to expiry must be non-negative, got {self.T}")
        if self.sigma < 0:
            raise ValueError(f"Volatility must be non-negative, got {self.sigma}")


@dataclass(slots=True)
class BSGreeks:
    """Container for option Greeks."""
    delta: float
    gamma: float
    theta: float   # per calendar day
    vega: float    # per 1% move in vol
    rho: float     # per 1% move in rate
    vanna: float   # dDelta/dSigma
    volga: float   # dVega/dSigma (Vomma)
    charm: float   # dDelta/dt


# ---------------------------------------------------------------------------
# Core BS functions
# ---------------------------------------------------------------------------
def _d1_d2(inp: BSInputs) -> tuple[float, float]:
    """Compute d1 and d2 components.

    Returns:
        Tuple (d1, d2).
    """
    T = max(inp.T, _MIN_T)
    sigma = max(inp.sigma, _MIN_VOL)
    vol_sqrt_T = sigma * math.sqrt(T)
    d1 = (math.log(inp.S / inp.K) + (inp.r - inp.q + 0.5 * sigma**2) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2


def bs_price(inp: BSInputs, option_type: str = "call") -> float:
    """Price European option via Black-Scholes closed form.

    Args:
        inp: BSInputs with validated market parameters.
        option_type: "call" or "put".

    Returns:
        Option fair value.

    Raises:
        ValueError: For unknown option_type.
    """
    if option_type not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    d1, d2 = _d1_d2(inp)
    discount = math.exp(-inp.r * inp.T)
    fwd_discount = math.exp(-inp.q * inp.T)

    if option_type == "call":
        price = inp.S * fwd_discount * norm.cdf(d1) - inp.K * discount * norm.cdf(d2)
    else:
        price = inp.K * discount * norm.cdf(-d2) - inp.S * fwd_discount * norm.cdf(-d1)

    return max(price, 0.0)


def bs_greeks(inp: BSInputs, option_type: str = "call") -> BSGreeks:
    """Compute all first and second order Greeks analytically.

    Algorithm:
      - Delta: N(d1) for call, N(d1)-1 for put.
      - Gamma: φ(d1) / (S σ √T) — same for call and put.
      - Theta: involves d1, d2 and risk-free discounting.
      - Vega: S φ(d1) √T.
      - Higher-order greeks from chain rule.

    Args:
        inp: BSInputs.
        option_type: "call" or "put".

    Returns:
        BSGreeks dataclass.
    """
    if option_type not in {"call", "put"}:
        raise ValueError(f"Unknown option_type: {option_type!r}")

    T = max(inp.T, _MIN_T)
    sigma = max(inp.sigma, _MIN_VOL)
    sqrt_T = math.sqrt(T)
    d1, d2 = _d1_d2(inp)
    phi_d1 = norm.pdf(d1)
    fwd_disc = math.exp(-inp.q * T)
    disc = math.exp(-inp.r * T)

    # Delta
    if option_type == "call":
        delta = fwd_disc * norm.cdf(d1)
    else:
        delta = fwd_disc * (norm.cdf(d1) - 1.0)

    # Gamma (same for call/put)
    gamma = fwd_disc * phi_d1 / (inp.S * sigma * sqrt_T)

    # Theta (per calendar day, /365)
    theta_annual = (
        -inp.S * fwd_disc * phi_d1 * sigma / (2.0 * sqrt_T)
        - inp.r * inp.K * disc * (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2))
        + inp.q * inp.S * fwd_disc * (norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1))
    )
    theta = theta_annual / 365.0

    # Vega (per 1% vol change)
    vega = inp.S * fwd_disc * phi_d1 * sqrt_T / 100.0

    # Rho (per 1% rate change)
    if option_type == "call":
        rho = inp.K * T * disc * norm.cdf(d2) / 100.0
    else:
        rho = -inp.K * T * disc * norm.cdf(-d2) / 100.0

    # Vanna: dDelta/dSigma
    vanna = -fwd_disc * phi_d1 * d2 / sigma

    # Volga (Vomma): dVega/dSigma
    volga = inp.S * fwd_disc * phi_d1 * sqrt_T * d1 * d2 / sigma / 100.0

    # Charm: dDelta/dt (per calendar day)
    if option_type == "call":
        charm_annual = inp.q * fwd_disc * norm.cdf(d1) - fwd_disc * phi_d1 * (
            2.0 * (inp.r - inp.q) * T - d2 * sigma * sqrt_T
        ) / (2.0 * T * sigma * sqrt_T)
    else:
        charm_annual = -inp.q * fwd_disc * norm.cdf(-d1) - fwd_disc * phi_d1 * (
            2.0 * (inp.r - inp.q) * T - d2 * sigma * sqrt_T
        ) / (2.0 * T * sigma * sqrt_T)
    charm = charm_annual / 365.0

    return BSGreeks(
        delta=delta, gamma=gamma, theta=theta, vega=vega,
        rho=rho, vanna=vanna, volga=volga, charm=charm,
    )


def implied_vol(
    market_price: float,
    inp: BSInputs,
    option_type: str = "call",
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Invert BS formula to find implied volatility via Brent's method.

    Args:
        market_price: Observed market price.
        inp: BSInputs (sigma field will be overridden in search).
        option_type: "call" or "put".
        tol: Root-finding tolerance.
        max_iter: Maximum Brent iterations.

    Returns:
        Implied volatility. Returns NaN if not found.
    """
    intrinsic = max(inp.S - inp.K * math.exp(-inp.r * inp.T), 0.0)
    if market_price < intrinsic - tol:
        logger.warning("Market price %.4f below intrinsic %.4f", market_price, intrinsic)
        return float("nan")

    def objective(sigma: float) -> float:
        try:
            candidate = BSInputs(inp.S, inp.K, inp.T, inp.r, sigma, inp.q)
            return bs_price(candidate, option_type) - market_price
        except Exception:
            return float("nan")

    try:
        iv = brentq(objective, _MIN_VOL, _MAX_VOL, xtol=tol, maxiter=max_iter)
        return float(iv)
    except ValueError as exc:
        logger.debug("Implied vol not found: %s", exc)
        return float("nan")


def taylor_pnl(
    greeks: BSGreeks,
    dS: float,
    dt_days: float,
) -> float:
    """Approximate P&L via second-order Taylor expansion.

    Formula: dV ≈ Δ·dS + ½Γ·dS² + Θ·dt

    Args:
        greeks: BSGreeks from bs_greeks().
        dS: Change in underlying price.
        dt_days: Time elapsed in calendar days.

    Returns:
        Approximate P&L.
    """
    return greeks.delta * dS + 0.5 * greeks.gamma * dS**2 + greeks.theta * dt_days


# ---------------------------------------------------------------------------
# Vectorised Polars helpers
# ---------------------------------------------------------------------------
def price_strike_grid(
    S: float,
    strikes: list[float],
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> pl.DataFrame:
    """Price options over a strike grid, returning a Polars DataFrame.

    Args:
        S: Spot price.
        strikes: List of strike prices.
        T: Time to expiry.
        r: Risk-free rate.
        sigma: Volatility.
        q: Dividend yield.
        option_type: "call" or "put".

    Returns:
        Polars DataFrame with columns: strike, price, delta, gamma, theta, vega.
    """
    records = []
    for K in strikes:
        try:
            inp = BSInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
            price = bs_price(inp, option_type)
            g = bs_greeks(inp, option_type)
            records.append({
                "strike": K,
                "price": price,
                "delta": g.delta,
                "gamma": g.gamma,
                "theta": g.theta,
                "vega": g.vega,
            })
        except ValueError as exc:
            logger.warning("Skipping K=%.2f: %s", K, exc)

    return pl.DataFrame(records)
