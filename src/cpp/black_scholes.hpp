/**
 * black_scholes.hpp — Black-Scholes pricer, Greeks, and Taylor P&L engine.
 *
 * Filename: src/cpp/black_scholes.hpp
 * Author:   Quant Alpha Research Team
 * Style:    Google C++ Style Guide
 * Standard: C++26
 * Libs:     Eigen 3.4+, <numbers>, <expected>
 *
 * Algorithm:
 *   1. Validate inputs; compute d1, d2.
 *   2. Price via closed-form BS formula.
 *   3. Compute all Greeks analytically.
 *   4. Provide Taylor-series P&L approximation.
 *   5. Invert BS via Brent's method for implied vol.
 *
 * Time Complexity:  O(1) per pricing call.
 * Space Complexity: O(1).
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <expected>
#include <numbers>
#include <stdexcept>
#include <string>

namespace quant {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
inline constexpr double kMinVol   = 1e-8;
inline constexpr double kMaxVol   = 10.0;
inline constexpr double kMinT     = 1e-10;
inline constexpr double kSqrt2Pi  = std::numbers::sqrt2 * std::numbers::sqrt3;  // approx; use constexpr below
inline constexpr double kInvSqrt2 = 0.70710678118654752440;

// ---------------------------------------------------------------------------
// Standard normal CDF/PDF (Abramowitz & Stegun approximation)
// ---------------------------------------------------------------------------
[[nodiscard]] inline double NormCdf(double x) noexcept {
  return 0.5 * std::erfc(-x * kInvSqrt2);
}

[[nodiscard]] inline double NormPdf(double x) noexcept {
  static constexpr double kInvSqrt2Pi = 0.39894228040143267794;
  return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

// ---------------------------------------------------------------------------
// Input struct
// ---------------------------------------------------------------------------
struct BSInputs {
  double S;       ///< Spot price
  double K;       ///< Strike price
  double T;       ///< Time to expiry (years)
  double r;       ///< Risk-free rate (annualised)
  double sigma;   ///< Volatility (annualised)
  double q{0.0};  ///< Dividend yield

  /// Validate inputs. Returns error string if invalid.
  [[nodiscard]] std::expected<void, std::string> Validate() const noexcept {
    if (S <= 0) return std::unexpected("Spot must be positive");
    if (K <= 0) return std::unexpected("Strike must be positive");
    if (T < 0)  return std::unexpected("T must be non-negative");
    if (sigma < 0) return std::unexpected("sigma must be non-negative");
    return {};
  }
};

// ---------------------------------------------------------------------------
// Greeks struct
// ---------------------------------------------------------------------------
struct BSGreeks {
  double delta{0.0};
  double gamma{0.0};
  double theta{0.0};   ///< Per calendar day
  double vega{0.0};    ///< Per 1% vol move
  double rho{0.0};     ///< Per 1% rate move
  double vanna{0.0};
  double volga{0.0};
  double charm{0.0};   ///< Per calendar day
};

// ---------------------------------------------------------------------------
// Option type enum
// ---------------------------------------------------------------------------
enum class OptionType : uint8_t { kCall, kPut };

// ---------------------------------------------------------------------------
// Core BS helpers
// ---------------------------------------------------------------------------
namespace internal {

struct D1D2 { double d1, d2; };

[[nodiscard]] inline D1D2 ComputeD1D2(const BSInputs& inp) noexcept {
  const double T     = std::max(inp.T, kMinT);
  const double sigma = std::max(inp.sigma, kMinVol);
  const double sqrtT = std::sqrt(T);
  const double d1    = (std::log(inp.S / inp.K) +
                        (inp.r - inp.q + 0.5 * sigma * sigma) * T) /
                       (sigma * sqrtT);
  return {d1, d1 - sigma * sqrtT};
}

}  // namespace internal

// ---------------------------------------------------------------------------
// Price
// ---------------------------------------------------------------------------
/**
 * Price a European option via Black-Scholes closed form.
 *
 * @param inp     Validated market inputs.
 * @param type    OptionType::kCall or OptionType::kPut.
 * @return        Option fair value (non-negative).
 */
[[nodiscard]] inline double BSPrice(const BSInputs& inp,
                                     OptionType type = OptionType::kCall) noexcept {
  const auto [d1, d2] = internal::ComputeD1D2(inp);
  const double disc      = std::exp(-inp.r * std::max(inp.T, kMinT));
  const double fwd_disc  = std::exp(-inp.q * std::max(inp.T, kMinT));

  double price{0.0};
  if (type == OptionType::kCall) {
    price = inp.S * fwd_disc * NormCdf(d1) - inp.K * disc * NormCdf(d2);
  } else {
    price = inp.K * disc * NormCdf(-d2) - inp.S * fwd_disc * NormCdf(-d1);
  }
  return std::max(price, 0.0);
}

// ---------------------------------------------------------------------------
// Greeks
// ---------------------------------------------------------------------------
/**
 * Compute all first- and second-order Greeks analytically.
 *
 * @param inp   BSInputs.
 * @param type  Call or Put.
 * @return      BSGreeks struct.
 */
[[nodiscard]] inline BSGreeks BSComputeGreeks(
    const BSInputs& inp, OptionType type = OptionType::kCall) noexcept {
  const double T      = std::max(inp.T, kMinT);
  const double sigma  = std::max(inp.sigma, kMinVol);
  const double sqrtT  = std::sqrt(T);
  const auto [d1, d2] = internal::ComputeD1D2(inp);

  const double phi_d1    = NormPdf(d1);
  const double fwd_disc  = std::exp(-inp.q * T);
  const double disc      = std::exp(-inp.r * T);
  const bool   is_call   = (type == OptionType::kCall);

  BSGreeks g;

  // Delta
  g.delta = is_call ? fwd_disc * NormCdf(d1)
                    : fwd_disc * (NormCdf(d1) - 1.0);

  // Gamma (same call/put)
  g.gamma = fwd_disc * phi_d1 / (inp.S * sigma * sqrtT);

  // Theta (per calendar day / 365)
  const double nd2 = is_call ? NormCdf(d2) : NormCdf(-d2);
  const double nd1 = is_call ? NormCdf(d1) : -NormCdf(-d1);
  const double theta_annual =
      -inp.S * fwd_disc * phi_d1 * sigma / (2.0 * sqrtT)
      - inp.r * inp.K * disc * nd2
      + inp.q * inp.S * fwd_disc * nd1;
  g.theta = theta_annual / 365.0;

  // Vega (per 1% vol)
  g.vega = inp.S * fwd_disc * phi_d1 * sqrtT / 100.0;

  // Rho (per 1% rate)
  g.rho = is_call
          ? inp.K * T * disc * NormCdf(d2) / 100.0
          : -inp.K * T * disc * NormCdf(-d2) / 100.0;

  // Vanna: dDelta/dSigma
  g.vanna = -fwd_disc * phi_d1 * d2 / sigma;

  // Volga (Vomma): dVega/dSigma
  g.volga = inp.S * fwd_disc * phi_d1 * sqrtT * d1 * d2 / sigma / 100.0;

  // Charm: dDelta/dt per calendar day
  const double charm_annual = is_call
      ? inp.q * fwd_disc * NormCdf(d1) -
        fwd_disc * phi_d1 *
            (2.0 * (inp.r - inp.q) * T - d2 * sigma * sqrtT) /
            (2.0 * T * sigma * sqrtT)
      : -inp.q * fwd_disc * NormCdf(-d1) -
        fwd_disc * phi_d1 *
            (2.0 * (inp.r - inp.q) * T - d2 * sigma * sqrtT) /
            (2.0 * T * sigma * sqrtT);
  g.charm = charm_annual / 365.0;

  return g;
}

// ---------------------------------------------------------------------------
// Taylor P&L approximation
// ---------------------------------------------------------------------------
/**
 * Approximate option P&L: dV ≈ Δ·dS + ½Γ·dS² + Θ·dt
 *
 * @param g        BSGreeks from BSComputeGreeks.
 * @param dS       Change in underlying price.
 * @param dt_days  Elapsed time in calendar days.
 * @return         Approximate P&L.
 */
[[nodiscard]] inline double TaylorPnl(const BSGreeks& g,
                                       double dS,
                                       double dt_days) noexcept {
  return g.delta * dS + 0.5 * g.gamma * dS * dS + g.theta * dt_days;
}

// ---------------------------------------------------------------------------
// Implied Volatility via Brent's method
// ---------------------------------------------------------------------------
/**
 * Find implied volatility by inverting BS price via Brent's method.
 *
 * @param market_price  Observed market price.
 * @param inp           BSInputs (sigma will be overridden in search).
 * @param type          Call or Put.
 * @param tol           Root-finding tolerance.
 * @param max_iter      Maximum iterations.
 * @return              Implied vol, or NaN if not found.
 */
[[nodiscard]] inline double ImpliedVol(
    double market_price, BSInputs inp,
    OptionType type = OptionType::kCall,
    double tol = 1e-8, int max_iter = 200) noexcept {
  auto objective = [&](double sigma) -> double {
    inp.sigma = sigma;
    return BSPrice(inp, type) - market_price;
  };

  double lo = kMinVol, hi = kMaxVol;
  double f_lo = objective(lo), f_hi = objective(hi);

  if (f_lo * f_hi > 0) return std::numeric_limits<double>::quiet_NaN();

  for (int i = 0; i < max_iter; ++i) {
    double mid = 0.5 * (lo + hi);
    if (hi - lo < tol) return mid;
    double f_mid = objective(mid);
    if (f_lo * f_mid < 0) { hi = mid; f_hi = f_mid; }
    else                  { lo = mid; f_lo = f_mid; }
  }
  return 0.5 * (lo + hi);
}

}  // namespace quant
