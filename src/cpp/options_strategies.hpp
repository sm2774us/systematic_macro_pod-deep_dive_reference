/**
 * options_strategies.hpp — Options strategy payoff engine (C++26 / Eigen).
 *
 * Filename: src/cpp/options_strategies.hpp
 * Author:   Quant Alpha Research Team
 * Style:    Google C++ Style Guide
 * Standard: C++26
 * Libs:     Eigen 3.4+
 *
 * Strategies: Iron Condor, Iron Butterfly, Ratio Spread, Calendar Spread
 *
 * Algorithm: Vectorised payoff computation over spot price Eigen arrays.
 * Time Complexity:  O(N) per strategy (N = spot grid points).
 * Space Complexity: O(N).
 */

#pragma once

#include <algorithm>
#include <expected>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace quant {

// ---------------------------------------------------------------------------
// Scalar option payoffs
// ---------------------------------------------------------------------------
[[nodiscard]] inline Eigen::ArrayXd CallPayoff(
    const Eigen::ArrayXd& S, double K) noexcept {
  return (S - K).max(0.0);
}

[[nodiscard]] inline Eigen::ArrayXd PutPayoff(
    const Eigen::ArrayXd& S, double K) noexcept {
  return (K - S).max(0.0);
}

// ---------------------------------------------------------------------------
// Result struct
// ---------------------------------------------------------------------------
struct StrategyResult {
  std::string     name;
  Eigen::ArrayXd  spot_grid;
  Eigen::ArrayXd  payoff;
  double          max_profit{0.0};
  double          max_loss{0.0};
  double          net_premium{0.0};
};

[[nodiscard]] inline StrategyResult MakeResult(
    std::string name, Eigen::ArrayXd spot, Eigen::ArrayXd payoff, double net_premium) {
  return StrategyResult{
    .name       = std::move(name),
    .spot_grid  = std::move(spot),
    .payoff     = payoff,
    .max_profit = payoff.maxCoeff(),
    .max_loss   = payoff.minCoeff(),
    .net_premium = net_premium,
  };
}

// ---------------------------------------------------------------------------
// Spot grid helper
// ---------------------------------------------------------------------------
[[nodiscard]] inline Eigen::ArrayXd LinSpace(double lo, double hi, int n) {
  return Eigen::ArrayXd::LinSpaced(n, lo, hi);
}

// ---------------------------------------------------------------------------
// Iron Condor
// ---------------------------------------------------------------------------
/**
 * Iron Condor payoff: Long put(A) + Short put(B) + Short call(C) + Long call(D)
 *
 * @param S_lo, S_hi  Spot range.
 * @param K_A..K_D   Strikes (A < B < C < D).
 * @param net_credit  Premium received.
 * @param n_points    Grid resolution.
 * @return            StrategyResult or error string.
 */
[[nodiscard]] inline std::expected<StrategyResult, std::string>
IronCondor(double S_lo, double S_hi,
           double K_A, double K_B, double K_C, double K_D,
           double net_credit, int n_points = 500) noexcept {
  if (!(K_A < K_B && K_B < K_C && K_C < K_D))
    return std::unexpected("Strikes must satisfy A < B < C < D");
  if (net_credit < 0)
    return std::unexpected("Iron condor should receive net credit");

  Eigen::ArrayXd S = LinSpace(S_lo, S_hi, n_points);
  Eigen::ArrayXd payoff =
      PutPayoff(S, K_A) - PutPayoff(S, K_B)
    - CallPayoff(S, K_C) + CallPayoff(S, K_D)
    + net_credit;

  return MakeResult("Iron Condor", std::move(S), std::move(payoff), net_credit);
}

// ---------------------------------------------------------------------------
// Iron Butterfly
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::expected<StrategyResult, std::string>
IronButterfly(double S_lo, double S_hi,
              double K_A, double K_B, double K_D,
              double net_credit, int n_points = 500) noexcept {
  if (!(K_A < K_B && K_B < K_D))
    return std::unexpected("Strikes must satisfy A < B < D");

  Eigen::ArrayXd S = LinSpace(S_lo, S_hi, n_points);
  Eigen::ArrayXd payoff =
      PutPayoff(S, K_A) - PutPayoff(S, K_B)
    - CallPayoff(S, K_B) + CallPayoff(S, K_D)
    + net_credit;

  return MakeResult("Iron Butterfly", std::move(S), std::move(payoff), net_credit);
}

// ---------------------------------------------------------------------------
// Call Ratio Spread
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::expected<StrategyResult, std::string>
CallRatioSpread(double S_lo, double S_hi,
                double K_long, double K_short,
                double net_debit, int ratio = 2,
                int n_points = 500) noexcept {
  if (K_long >= K_short)
    return std::unexpected("K_long must be < K_short");

  Eigen::ArrayXd S = LinSpace(S_lo, S_hi, n_points);
  Eigen::ArrayXd payoff =
      CallPayoff(S, K_long) - ratio * CallPayoff(S, K_short) - net_debit;

  return MakeResult("Call Ratio Spread", std::move(S), std::move(payoff), -net_debit);
}

}  // namespace quant
