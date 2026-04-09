/**
 * kelly_criterion.hpp — Kelly Criterion and multi-asset Kelly allocation.
 *
 * Filename: src/cpp/kelly_criterion.hpp
 * Author:   Quant Alpha Research Team
 * Style:    Google C++ Style Guide
 * Standard: C++26
 * Libs:     Eigen 3.4+
 *
 * Algorithm:
 *   1. Scalar Kelly: f* = (p(b+1) - 1) / b
 *   2. Half-Kelly: f_half = 0.5 * f*
 *   3. Multi-asset Kelly: f_vec = (1/λ) Σ⁻¹ μ (with ridge regularisation)
 *   4. Growth curve: E[log W] = p*log(1+b*f) + (1-p)*log(1-f)
 *
 * Time Complexity:  O(1) scalar; O(N³) multi-asset (LLT decomposition).
 * Space Complexity: O(N²) for covariance.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <expected>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace quant {

inline constexpr double kMaxKellyFrac = 1.0;
inline constexpr double kMinEdge      = 1e-10;

// ---------------------------------------------------------------------------
// Result structs
// ---------------------------------------------------------------------------
struct KellyResult {
  double full_kelly{0.0};
  double half_kelly{0.0};
  double expected_log_growth{0.0};
  double variance_of_returns{0.0};
  double edge{0.0};
  double odds{0.0};
};

struct KellyGrowthPoint {
  double fraction;
  double expected_log_growth;
  double variance;
};

// ---------------------------------------------------------------------------
// Scalar Kelly
// ---------------------------------------------------------------------------
/**
 * Compute Kelly fraction for binary bet.
 *
 * Formula: f* = (p(b+1) - 1) / b
 *
 * @param p  Win probability in (0, 1).
 * @param b  Net odds (b to 1).
 * @return   KellyResult or error string.
 */
[[nodiscard]] inline std::expected<KellyResult, std::string>
KellyScalar(double p, double b) noexcept {
  if (p <= 0.0 || p >= 1.0)
    return std::unexpected("p must be in (0, 1)");
  if (b <= 0.0)
    return std::unexpected("b (net odds) must be positive");

  const double edge   = p * (b + 1.0) - 1.0;
  const double f_star = (edge > kMinEdge) ? std::min(edge / b, kMaxKellyFrac) : 0.0;
  const double f_half = 0.5 * f_star;

  double elg{0.0}, var{0.0};
  if (f_star > kMinEdge) {
    elg = p * std::log(1.0 + b * f_star) + (1.0 - p) * std::log(1.0 - f_star);
    const double ev2 = p * std::pow(std::log(1.0 + b * f_star), 2.0) +
                       (1.0 - p) * std::pow(std::log(1.0 - f_star), 2.0);
    var = ev2 - elg * elg;
  }

  return KellyResult{f_star, f_half, elg, var, edge, b};
}

// ---------------------------------------------------------------------------
// Growth curve
// ---------------------------------------------------------------------------
/**
 * Compute expected log-growth vs fraction for plotting.
 *
 * @param p        Win probability.
 * @param b        Net odds.
 * @param n_points Number of grid points.
 * @return         Vector of (fraction, elg, variance) points.
 */
[[nodiscard]] inline std::vector<KellyGrowthPoint>
KellyGrowthCurve(double p, double b, int n_points = 200) {
  if (p <= 0.0 || p >= 1.0)
    throw std::invalid_argument("p must be in (0, 1)");
  if (b <= 0.0)
    throw std::invalid_argument("b must be positive");

  std::vector<KellyGrowthPoint> result;
  result.reserve(static_cast<size_t>(n_points));

  for (int i = 0; i < n_points; ++i) {
    const double f = static_cast<double>(i) / (n_points - 1) * (1.0 - 1e-6);
    const double elg = p * std::log(1.0 + b * f) + (1.0 - p) * std::log(1.0 - f);
    const double ev2 = p * std::pow(std::log(1.0 + b * f), 2.0) +
                       (1.0 - p) * std::pow(std::log(1.0 - f), 2.0);
    const double var = ev2 - elg * elg;
    result.push_back({f, elg, var});
  }
  return result;
}

// ---------------------------------------------------------------------------
// Multi-asset Kelly
// ---------------------------------------------------------------------------
/**
 * Compute multi-asset Kelly allocation: f* = (1/λ) Σ⁻¹ μ
 *
 * Uses Eigen LLT (Cholesky) decomposition for O(N³) efficiency.
 * Ridge regularisation ensures positive-definiteness.
 *
 * @param mu              Expected returns (N).
 * @param sigma           Covariance matrix (N×N).
 * @param risk_aversion   λ (default 1.0 = pure Kelly / log-utility).
 * @param regularisation  Ridge term for numerical stability.
 * @param max_leverage    Cap on L1 norm of allocation vector.
 * @return                Allocation vector (N) or error.
 */
[[nodiscard]] inline std::expected<Eigen::VectorXd, std::string>
KellyMultiAsset(
    const Eigen::VectorXd& mu,
    const Eigen::MatrixXd& sigma,
    double risk_aversion   = 1.0,
    double regularisation  = 1e-4,
    double max_leverage    = 2.0) noexcept {
  const int N = static_cast<int>(mu.size());
  if (sigma.rows() != N || sigma.cols() != N)
    return std::unexpected("sigma must be N×N matching mu");
  if (risk_aversion <= 0.0)
    return std::unexpected("risk_aversion must be positive");

  Eigen::MatrixXd sigma_reg = sigma + regularisation * Eigen::MatrixXd::Identity(N, N);

  // Cholesky decomposition
  Eigen::LLT<Eigen::MatrixXd> llt(sigma_reg);
  if (llt.info() != Eigen::Success) {
    // Fall back to stronger regularisation
    sigma_reg += 1e-3 * Eigen::MatrixXd::Identity(N, N);
    llt.compute(sigma_reg);
    if (llt.info() != Eigen::Success)
      return std::unexpected("Covariance matrix is not positive-definite");
  }

  Eigen::VectorXd f_kelly = (1.0 / risk_aversion) * llt.solve(mu);

  // Cap leverage (L1 norm)
  const double leverage = f_kelly.lpNorm<1>();
  if (leverage > max_leverage) {
    f_kelly *= (max_leverage / leverage);
  }

  return f_kelly;
}

}  // namespace quant
