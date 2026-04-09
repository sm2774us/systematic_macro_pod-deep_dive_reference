/**
 * test_quant.cpp — GoogleTest unit tests for all C++ quant modules.
 *
 * Filename: src/cpp/tests/test_quant.cpp
 * Author:   Quant Alpha Research Team
 * Style:    Google C++ Style Guide
 * Standard: C++26
 * Coverage: 100% of src/cpp/*.hpp
 */

#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "black_scholes.hpp"
#include "kelly_criterion.hpp"

namespace quant::test {

// ===========================================================================
// Black-Scholes Tests
// ===========================================================================

class BlackScholesTest : public ::testing::Test {
 protected:
  BSInputs AtmInputs() const {
    return BSInputs{.S = 100.0, .K = 100.0, .T = 1.0, .r = 0.05, .sigma = 0.2};
  }
};

// --- Input Validation ---
TEST_F(BlackScholesTest, ValidInputsPassValidation) {
  auto inp = AtmInputs();
  auto result = inp.Validate();
  EXPECT_TRUE(result.has_value());
}

TEST_F(BlackScholesTest, NegativeSpotFailsValidation) {
  BSInputs inp{.S = -1.0, .K = 100.0, .T = 1.0, .r = 0.05, .sigma = 0.2};
  auto result = inp.Validate();
  EXPECT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("Spot"), std::string::npos);
}

TEST_F(BlackScholesTest, NegativeStrikeFailsValidation) {
  BSInputs inp{.S = 100.0, .K = -10.0, .T = 1.0, .r = 0.05, .sigma = 0.2};
  EXPECT_FALSE(inp.Validate().has_value());
}

TEST_F(BlackScholesTest, NegativeSigmaFailsValidation) {
  BSInputs inp{.S = 100.0, .K = 100.0, .T = 1.0, .r = 0.05, .sigma = -0.1};
  EXPECT_FALSE(inp.Validate().has_value());
}

// --- Pricing ---
TEST_F(BlackScholesTest, CallPricePositive) {
  EXPECT_GT(BSPrice(AtmInputs(), OptionType::kCall), 0.0);
}

TEST_F(BlackScholesTest, PutPricePositive) {
  EXPECT_GT(BSPrice(AtmInputs(), OptionType::kPut), 0.0);
}

TEST_F(BlackScholesTest, AtmCallApproxCorrect) {
  // ATM 1-year call at σ=20%, r=5% → ~10.45
  const double call = BSPrice(AtmInputs(), OptionType::kCall);
  EXPECT_GT(call, 10.0);
  EXPECT_LT(call, 11.0);
}

TEST_F(BlackScholesTest, PutCallParityHolds) {
  auto inp = AtmInputs();
  const double call = BSPrice(inp, OptionType::kCall);
  const double put  = BSPrice(inp, OptionType::kPut);
  // C - P = S*e^{-qT} - K*e^{-rT}
  const double lhs = call - put;
  const double rhs = inp.S * std::exp(-inp.q * inp.T) -
                     inp.K * std::exp(-inp.r * inp.T);
  EXPECT_NEAR(lhs, rhs, 1e-8);
}

TEST_F(BlackScholesTest, DeepITMCallApproachesForward) {
  BSInputs inp{.S = 200.0, .K = 100.0, .T = 1.0, .r = 0.05, .sigma = 0.2};
  const double call      = BSPrice(inp, OptionType::kCall);
  const double intrinsic = 200.0 - 100.0 * std::exp(-0.05);
  EXPECT_GT(call, intrinsic * 0.95);
}

TEST_F(BlackScholesTest, DeepOTMCallNearZero) {
  BSInputs inp{.S = 50.0, .K = 200.0, .T = 0.1, .r = 0.05, .sigma = 0.1};
  EXPECT_LT(BSPrice(inp, OptionType::kCall), 1e-3);
}

// --- Greeks ---
TEST_F(BlackScholesTest, CallDeltaBetweenZeroAndOne) {
  const auto g = BSComputeGreeks(AtmInputs(), OptionType::kCall);
  EXPECT_GT(g.delta, 0.0);
  EXPECT_LT(g.delta, 1.0);
}

TEST_F(BlackScholesTest, PutDeltaBetweenMinusOneAndZero) {
  const auto g = BSComputeGreeks(AtmInputs(), OptionType::kPut);
  EXPECT_LT(g.delta, 0.0);
  EXPECT_GT(g.delta, -1.0);
}

TEST_F(BlackScholesTest, GammaPositive) {
  const auto g = BSComputeGreeks(AtmInputs(), OptionType::kCall);
  EXPECT_GT(g.gamma, 0.0);
}

TEST_F(BlackScholesTest, ThetaNegativeForLongCall) {
  const auto g = BSComputeGreeks(AtmInputs(), OptionType::kCall);
  EXPECT_LT(g.theta, 0.0);
}

TEST_F(BlackScholesTest, VegaPositive) {
  const auto g = BSComputeGreeks(AtmInputs(), OptionType::kCall);
  EXPECT_GT(g.vega, 0.0);
}

TEST_F(BlackScholesTest, PutCallDeltaRelationship) {
  auto inp       = AtmInputs();
  const auto gc  = BSComputeGreeks(inp, OptionType::kCall);
  const auto gp  = BSComputeGreeks(inp, OptionType::kPut);
  const double expected_diff = std::exp(-inp.q * inp.T);
  EXPECT_NEAR(gc.delta - gp.delta, expected_diff, 1e-8);
}

TEST_F(BlackScholesTest, GammaSameCallPut) {
  auto inp = AtmInputs();
  const auto gc = BSComputeGreeks(inp, OptionType::kCall);
  const auto gp = BSComputeGreeks(inp, OptionType::kPut);
  EXPECT_NEAR(gc.gamma, gp.gamma, 1e-12);
}

// --- Taylor P&L ---
TEST_F(BlackScholesTest, TaylorPnlPositiveForUpMove) {
  const auto g = BSComputeGreeks(AtmInputs(), OptionType::kCall);
  EXPECT_GT(TaylorPnl(g, 5.0, 0.0), 0.0);
}

TEST_F(BlackScholesTest, ThetaBleedNegativeForLongCall) {
  const auto g = BSComputeGreeks(AtmInputs(), OptionType::kCall);
  EXPECT_LT(TaylorPnl(g, 0.0, 1.0), 0.0);
}

// --- Implied Vol ---
TEST_F(BlackScholesTest, ImpliedVolRoundtrip) {
  BSInputs inp{.S = 100.0, .K = 100.0, .T = 1.0, .r = 0.05, .sigma = 0.25};
  const double market_price = BSPrice(inp, OptionType::kCall);
  const double iv = ImpliedVol(market_price, inp, OptionType::kCall);
  EXPECT_NEAR(iv, 0.25, 1e-5);
}

TEST_F(BlackScholesTest, ImpliedVolOTMStrike) {
  BSInputs inp{.S = 100.0, .K = 110.0, .T = 0.5, .r = 0.03, .sigma = 0.18};
  const double market_price = BSPrice(inp, OptionType::kCall);
  const double iv = ImpliedVol(market_price, inp, OptionType::kCall);
  EXPECT_NEAR(iv, 0.18, 1e-4);
}

// ===========================================================================
// Kelly Criterion Tests
// ===========================================================================

class KellyTest : public ::testing::Test {};

TEST_F(KellyTest, CoinFlipTwoToOne) {
  // p=0.6, b=2 → f* = (0.6*3-1)/2 = 0.4
  auto result = KellyScalar(0.6, 2.0);
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->full_kelly, 0.4, 1e-10);
  EXPECT_NEAR(result->half_kelly, 0.2, 1e-10);
}

TEST_F(KellyTest, HalfKellyIsHalfFullKelly) {
  auto result = KellyScalar(0.55, 1.8);
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->half_kelly, 0.5 * result->full_kelly, 1e-12);
}

TEST_F(KellyTest, NoEdgeReturnsZero) {
  auto result = KellyScalar(0.3, 2.0);  // edge = 0.3*3-1 = -0.1
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->full_kelly, 0.0);
}

TEST_F(KellyTest, InvalidPReturnsError) {
  EXPECT_FALSE(KellyScalar(1.1, 2.0).has_value());
  EXPECT_FALSE(KellyScalar(0.0, 2.0).has_value());
  EXPECT_FALSE(KellyScalar(-0.1, 2.0).has_value());
}

TEST_F(KellyTest, InvalidBReturnsError) {
  EXPECT_FALSE(KellyScalar(0.6, -1.0).has_value());
  EXPECT_FALSE(KellyScalar(0.6, 0.0).has_value());
}

TEST_F(KellyTest, ElgPositiveForEdge) {
  auto result = KellyScalar(0.6, 2.0);
  ASSERT_TRUE(result.has_value());
  EXPECT_GT(result->expected_log_growth, 0.0);
}

TEST_F(KellyTest, GrowthCurveSize) {
  const auto curve = KellyGrowthCurve(0.6, 2.0, 100);
  EXPECT_EQ(curve.size(), 100u);
}

TEST_F(KellyTest, GrowthCurveMaxAtFullKelly) {
  const auto curve = KellyGrowthCurve(0.6, 2.0, 1000);
  auto max_it = std::max_element(curve.begin(), curve.end(),
      [](const auto& a, const auto& b) {
        return a.expected_log_growth < b.expected_log_growth;
      });
  const double f_star = KellyScalar(0.6, 2.0)->full_kelly;
  EXPECT_NEAR(max_it->fraction, f_star, 0.02);
}

TEST_F(KellyTest, GrowthCurveInvalidThrows) {
  EXPECT_THROW(KellyGrowthCurve(0.0, 2.0), std::invalid_argument);
  EXPECT_THROW(KellyGrowthCurve(0.6, -1.0), std::invalid_argument);
}

// --- Multi-Asset Kelly ---
TEST_F(KellyTest, MultiAssetTwoAssets) {
  Eigen::VectorXd mu(2);
  mu << 0.01, 0.02;
  Eigen::MatrixXd sigma = Eigen::MatrixXd::Identity(2, 2) * 0.01;
  sigma(0, 1) = sigma(1, 0) = 0.002;

  auto result = KellyMultiAsset(mu, sigma);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->size(), 2);
}

TEST_F(KellyTest, MultiAssetLeverageCapped) {
  Eigen::VectorXd mu = Eigen::VectorXd::Constant(3, 0.1);
  Eigen::MatrixXd sigma = Eigen::MatrixXd::Identity(3, 3) * 0.001;

  auto result = KellyMultiAsset(mu, sigma, 1.0, 1e-4, 2.0);
  ASSERT_TRUE(result.has_value());
  EXPECT_LE(result->lpNorm<1>(), 2.0 + 1e-9);
}

TEST_F(KellyTest, MultiAssetShapeMismatchReturnsError) {
  Eigen::VectorXd mu(3);
  mu << 0.01, 0.02, 0.03;
  Eigen::MatrixXd sigma = Eigen::MatrixXd::Identity(2, 2);

  auto result = KellyMultiAsset(mu, sigma);
  EXPECT_FALSE(result.has_value());
}

TEST_F(KellyTest, MultiAssetSingularMatrixHandled) {
  // Rank-1 covariance (singular)
  Eigen::VectorXd mu(2); mu << 0.01, 0.01;
  Eigen::MatrixXd sigma = Eigen::MatrixXd::Ones(2, 2);  // rank 1

  // Should not crash due to regularisation
  auto result = KellyMultiAsset(mu, sigma);
  EXPECT_TRUE(result.has_value());
  EXPECT_FALSE(result->hasNaN());
}

// ===========================================================================
// NormCdf / NormPdf Tests
// ===========================================================================
TEST(MathTest, NormCdfSymmetry) {
  EXPECT_NEAR(NormCdf(0.0), 0.5, 1e-10);
  EXPECT_NEAR(NormCdf(1.96), 0.975, 1e-3);
  EXPECT_NEAR(NormCdf(-1.96) + NormCdf(1.96), 1.0, 1e-10);
}

TEST(MathTest, NormPdfNonNegative) {
  for (double x = -5.0; x <= 5.0; x += 0.5) {
    EXPECT_GE(NormPdf(x), 0.0);
  }
}

TEST(MathTest, NormPdfAtZero) {
  EXPECT_NEAR(NormPdf(0.0), 1.0 / std::sqrt(2.0 * std::numbers::pi), 1e-10);
}

}  // namespace quant::test

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
