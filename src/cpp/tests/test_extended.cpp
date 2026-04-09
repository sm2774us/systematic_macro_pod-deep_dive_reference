/**
 * test_extended.cpp — Additional GoogleTest coverage for hmm_regime and options_strategies.
 *
 * Filename: src/cpp/tests/test_extended.cpp
 * Author:   Quant Alpha Research Team
 * Style:    Google C++ Style Guide
 * Standard: C++26
 */

#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "hmm_regime.hpp"
#include "options_strategies.hpp"

namespace quant::test {

// ===========================================================================
// Options Strategies Tests
// ===========================================================================
class OptionsTest : public ::testing::Test {
 protected:
  static constexpr double S_lo = 80.0, S_hi = 130.0;
  static constexpr double K_A = 85.0, K_B = 90.0, K_C = 110.0, K_D = 115.0;
  static constexpr double K_atm = 100.0;
};

TEST_F(OptionsTest, CallPayoffZeroBelowStrike) {
  Eigen::ArrayXd S = LinSpace(50.0, 95.0, 100);
  auto payoff = CallPayoff(S, 100.0);
  EXPECT_TRUE((payoff == 0.0).all());
}

TEST_F(OptionsTest, CallPayoffLinearAboveStrike) {
  Eigen::ArrayXd S = LinSpace(105.0, 120.0, 50);
  auto payoff = CallPayoff(S, 100.0);
  for (int i = 0; i < 50; ++i) {
    EXPECT_NEAR(payoff(i), S(i) - 100.0, 1e-10);
  }
}

TEST_F(OptionsTest, PutPayoffZeroAboveStrike) {
  Eigen::ArrayXd S = LinSpace(105.0, 150.0, 100);
  auto payoff = PutPayoff(S, 100.0);
  EXPECT_TRUE((payoff == 0.0).all());
}

TEST_F(OptionsTest, IronCondorMaxProfitEqualsCredit) {
  auto result = IronCondor(S_lo, S_hi, K_A, K_B, K_C, K_D, 2.0);
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->max_profit, 2.0, 0.01);
}

TEST_F(OptionsTest, IronCondorMaxLossNegative) {
  auto result = IronCondor(S_lo, S_hi, K_A, K_B, K_C, K_D, 2.0);
  ASSERT_TRUE(result.has_value());
  EXPECT_LT(result->max_loss, 0.0);
}

TEST_F(OptionsTest, IronCondorWrongStrikeOrderReturnsError) {
  auto result = IronCondor(S_lo, S_hi, K_B, K_A, K_C, K_D, 2.0);
  EXPECT_FALSE(result.has_value());
}

TEST_F(OptionsTest, IronButterflyMaxProfitHigherThanCondor) {
  auto condor    = IronCondor(S_lo, S_hi, K_A, K_B, K_C, K_D, 2.0);
  auto butterfly = IronButterfly(S_lo, S_hi, K_A, K_atm, K_D, 8.0);
  ASSERT_TRUE(condor.has_value());
  ASSERT_TRUE(butterfly.has_value());
  EXPECT_GT(butterfly->max_profit, condor->max_profit);
}

TEST_F(OptionsTest, IronButterflyWrongOrderReturnsError) {
  auto result = IronButterfly(S_lo, S_hi, K_atm, K_A, K_D, 5.0);
  EXPECT_FALSE(result.has_value());
}

TEST_F(OptionsTest, RatioSpreadUnlimitedLoss) {
  auto result = CallRatioSpread(80.0, 200.0, 100.0, 110.0, 0.5);
  ASSERT_TRUE(result.has_value());
  // At very high spot, loss grows (last element should be very negative)
  EXPECT_LT(result->payoff(result->payoff.size()-1),
            result->payoff(result->payoff.size()-100));
}

TEST_F(OptionsTest, RatioSpreadInvalidStrikesReturnsError) {
  auto result = CallRatioSpread(80.0, 200.0, 110.0, 100.0, 0.5);
  EXPECT_FALSE(result.has_value());
}

// ===========================================================================
// HMM Regime Tests
// ===========================================================================
class HMMTest : public ::testing::Test {
 protected:
  /// Generate synthetic univariate regime data
  static Eigen::MatrixXd SyntheticObs(int T = 600, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> bull(0.001, 0.005);
    std::normal_distribution<double> bear(-0.003, 0.025);

    Eigen::MatrixXd obs(T, 1);
    for (int t = 0; t < T; ++t) {
      obs(t, 0) = (t % 100 < 60) ? bull(rng) : bear(rng);
    }
    return obs;
  }
};

TEST_F(HMMTest, TrainHMMReturnsModel) {
  auto obs   = SyntheticObs(600);
  auto model = TrainHMM(obs, 4, 42);
  ASSERT_TRUE(model.has_value());
  EXPECT_EQ(model->N, 4);
  EXPECT_EQ(model->states.size(), 4u);
}

TEST_F(HMMTest, TransitionMatrixRowStochastic) {
  auto obs   = SyntheticObs(600);
  auto model = TrainHMM(obs, 4, 42);
  ASSERT_TRUE(model.has_value());
  for (int i = 0; i < model->N; ++i) {
    EXPECT_NEAR(model->A.row(i).sum(), 1.0, 1e-8);
  }
}

TEST_F(HMMTest, InitialDistributionSumsToOne) {
  auto obs   = SyntheticObs(600);
  auto model = TrainHMM(obs, 4, 42);
  ASSERT_TRUE(model.has_value());
  EXPECT_NEAR(model->pi.sum(), 1.0, 1e-8);
}

TEST_F(HMMTest, ViterbiReturnsSameLengthAsObs) {
  auto obs   = SyntheticObs(600);
  auto model = TrainHMM(obs, 4, 42);
  ASSERT_TRUE(model.has_value());
  auto seq = Viterbi(*model, obs);
  EXPECT_EQ(static_cast<int>(seq.size()), 600);
}

TEST_F(HMMTest, ViterbiStatesInRange) {
  auto obs   = SyntheticObs(600);
  auto model = TrainHMM(obs, 4, 42);
  ASSERT_TRUE(model.has_value());
  auto seq = Viterbi(*model, obs);
  for (int s : seq) {
    EXPECT_GE(s, 0);
    EXPECT_LT(s, model->N);
  }
}

TEST_F(HMMTest, ForwardLogDimensionsCorrect) {
  auto obs   = SyntheticObs(100);
  auto model = TrainHMM(obs, 4, 42);
  ASSERT_TRUE(model.has_value());
  auto log_alpha = ForwardLog(*model, obs);
  EXPECT_EQ(log_alpha.rows(), 100);
  EXPECT_EQ(log_alpha.cols(), 4);
}

TEST_F(HMMTest, InsufficientDataReturnsError) {
  Eigen::MatrixXd tiny_obs(5, 1);
  tiny_obs.setRandom();
  auto result = TrainHMM(tiny_obs, 4, 42);
  EXPECT_FALSE(result.has_value());
}

TEST_F(HMMTest, LabelRegimesReturnsNLabels) {
  auto obs   = SyntheticObs(600);
  auto model = TrainHMM(obs, 4, 42);
  ASSERT_TRUE(model.has_value());
  auto labels = LabelRegimes(*model);
  EXPECT_EQ(static_cast<int>(labels.size()), 4);
}

TEST_F(HMMTest, LabelRegimesCoversAllRegimes) {
  auto obs    = SyntheticObs(600);
  auto model  = TrainHMM(obs, 4, 42);
  ASSERT_TRUE(model.has_value());
  auto labels = LabelRegimes(*model);
  std::vector<int> label_ints;
  for (auto l : labels) label_ints.push_back(static_cast<int>(l));
  std::sort(label_ints.begin(), label_ints.end());
  label_ints.erase(std::unique(label_ints.begin(), label_ints.end()), label_ints.end());
  EXPECT_EQ(label_ints.size(), 4u);
}

TEST_F(HMMTest, GaussianLogPdfAtMeanHigherThanTail) {
  GaussianState gs;
  gs.mu    = Eigen::VectorXd::Zero(1);
  gs.sigma = Eigen::MatrixXd::Identity(1, 1);
  const double log_pdf_mean = gs.LogPdf(Eigen::VectorXd::Zero(1));
  const double log_pdf_tail = gs.LogPdf(Eigen::VectorXd::Constant(1, 5.0));
  EXPECT_GT(log_pdf_mean, log_pdf_tail);
}

}  // namespace quant::test

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
