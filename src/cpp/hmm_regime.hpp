/**
 * hmm_regime.hpp — Hidden Markov Model regime detection (C++26 / Eigen).
 *
 * Filename: src/cpp/hmm_regime.hpp
 * Author:   Quant Alpha Research Team
 * Style:    Google C++ Style Guide
 * Standard: C++26
 * Libs:     Eigen 3.4+
 *
 * Algorithm (Gaussian HMM from scratch):
 *   1. Initialise A (transition), B (Gaussian emission), π (initial).
 *   2. Baum-Welch EM: iterate Forward → Backward → Update until convergence.
 *   3. Viterbi: find most likely hidden state sequence.
 *   4. Online forward filter: P(Z_t | O_{1:t}) in O(N²) per step.
 *
 * States: 0=BEAR, 1=STRESS, 2=NEUTRAL, 3=BULL (sorted by mean return)
 *
 * Time Complexity: O(T·N²·D) per EM iteration; O(T·N²) for Viterbi.
 * Space Complexity: O(T·N) for trellis; O(N·D) for Gaussian params.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <expected>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace quant {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
inline constexpr int    kNumStates   = 4;
inline constexpr double kLogEps      = -1e30;   ///< Log of near-zero
inline constexpr double kMinVariance = 1e-6;
inline constexpr int    kMaxEmIter   = 200;
inline constexpr double kEmTol       = 1e-4;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------
enum class RegimeLabel : int { kBear = 0, kStress = 1, kNeutral = 2, kBull = 3 };

// ---------------------------------------------------------------------------
// Gaussian emission parameters for one state
// ---------------------------------------------------------------------------
struct GaussianState {
  Eigen::VectorXd mu;      ///< Mean vector (D,)
  Eigen::MatrixXd sigma;   ///< Covariance matrix (D×D)

  /// Log PDF of observation under this Gaussian
  [[nodiscard]] double LogPdf(const Eigen::VectorXd& obs) const noexcept {
    const int D = static_cast<int>(mu.size());
    const Eigen::VectorXd diff = obs - mu;
    Eigen::LLT<Eigen::MatrixXd> llt(sigma);
    if (llt.info() != Eigen::Success) return kLogEps;
    const double log_det  = 2.0 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
    const double mahal    = diff.dot(llt.solve(diff));
    return -0.5 * (D * std::log(2.0 * std::numbers::pi) + log_det + mahal);
  }
};

// ---------------------------------------------------------------------------
// HMM Model
// ---------------------------------------------------------------------------
struct HMMModel {
  Eigen::MatrixXd A;                         ///< Transition matrix (N×N)
  std::vector<GaussianState> states;          ///< Emission params per state
  Eigen::VectorXd pi;                        ///< Initial state distribution
  int N{kNumStates};                         ///< Number of states
  int D{1};                                  ///< Feature dimension
  double log_likelihood{0.0};
};

// ---------------------------------------------------------------------------
// Forward algorithm: α_t(j) = P(O_{1:t}, Z_t=j)
// ---------------------------------------------------------------------------
/**
 * Compute log-scaled forward variables.
 *
 * @param model  Trained HMM.
 * @param obs    Observation sequence (T×D matrix, each row is O_t).
 * @return       log_alpha matrix (T×N).
 */
[[nodiscard]] inline Eigen::MatrixXd ForwardLog(
    const HMMModel& model, const Eigen::MatrixXd& obs) {
  const int T = static_cast<int>(obs.rows());
  const int N = model.N;

  Eigen::MatrixXd log_alpha(T, N);

  // Initialise
  for (int j = 0; j < N; ++j) {
    log_alpha(0, j) = std::log(std::max(model.pi(j), 1e-300)) +
                      model.states[j].LogPdf(obs.row(0).transpose());
  }

  // Recursion
  for (int t = 1; t < T; ++t) {
    for (int j = 0; j < N; ++j) {
      // log-sum-exp for numerical stability
      double max_val = kLogEps;
      for (int i = 0; i < N; ++i) {
        max_val = std::max(max_val, log_alpha(t-1, i) + std::log(std::max(model.A(i,j), 1e-300)));
      }
      double sum = 0.0;
      for (int i = 0; i < N; ++i) {
        const double val = log_alpha(t-1, i) + std::log(std::max(model.A(i,j), 1e-300));
        sum += std::exp(val - max_val);
      }
      log_alpha(t, j) = max_val + std::log(sum) +
                        model.states[j].LogPdf(obs.row(t).transpose());
    }
  }
  return log_alpha;
}

// ---------------------------------------------------------------------------
// Viterbi: find most likely state sequence
// ---------------------------------------------------------------------------
/**
 * Viterbi decoding for most probable state sequence.
 *
 * Algorithm:
 *   δ_t(j) = max_i [δ_{t-1}(i) * A_{ij}] * b_j(O_t)   (log-space)
 *
 * @param model  Trained HMM.
 * @param obs    Observation matrix (T×D).
 * @return       State sequence (T,) in {0,...,N-1}.
 */
[[nodiscard]] inline std::vector<int> Viterbi(
    const HMMModel& model, const Eigen::MatrixXd& obs) {
  const int T = static_cast<int>(obs.rows());
  const int N = model.N;

  Eigen::MatrixXd delta(T, N);   ///< Log probability trellis
  Eigen::MatrixXi psi(T, N);    ///< Back-pointer

  // Initialise
  for (int j = 0; j < N; ++j) {
    delta(0, j) = std::log(std::max(model.pi(j), 1e-300)) +
                  model.states[j].LogPdf(obs.row(0).transpose());
    psi(0, j) = 0;
  }

  // Forward pass
  for (int t = 1; t < T; ++t) {
    for (int j = 0; j < N; ++j) {
      double max_val = kLogEps;
      int    argmax  = 0;
      for (int i = 0; i < N; ++i) {
        const double val = delta(t-1, i) + std::log(std::max(model.A(i,j), 1e-300));
        if (val > max_val) { max_val = val; argmax = i; }
      }
      delta(t, j) = max_val + model.states[j].LogPdf(obs.row(t).transpose());
      psi(t, j)   = argmax;
    }
  }

  // Backtrack
  std::vector<int> seq(T);
  seq[T-1] = 0;
  for (int j = 1; j < N; ++j) {
    if (delta(T-1, j) > delta(T-1, seq[T-1])) seq[T-1] = j;
  }
  for (int t = T-2; t >= 0; --t) {
    seq[t] = psi(t+1, seq[t+1]);
  }
  return seq;
}

// ---------------------------------------------------------------------------
// Baum-Welch EM training
// ---------------------------------------------------------------------------
/**
 * Train HMM via Baum-Welch EM algorithm.
 *
 * @param obs        Observation matrix (T×D).
 * @param n_states   Number of hidden states.
 * @param seed       Random seed for initialisation.
 * @return           Trained HMMModel or error.
 */
[[nodiscard]] inline std::expected<HMMModel, std::string>
TrainHMM(const Eigen::MatrixXd& obs, int n_states = kNumStates, uint32_t seed = 42) {
  const int T = static_cast<int>(obs.rows());
  const int D = static_cast<int>(obs.cols());
  const int N = n_states;

  if (T < N * 5)
    return std::unexpected("Too few observations for HMM training");

  std::mt19937 rng(seed);
  std::normal_distribution<double> gauss(0.0, 1.0);

  // ---------------------------------------------------------------------------
  // Initialise model randomly
  // ---------------------------------------------------------------------------
  HMMModel model;
  model.N = N;
  model.D = D;

  // Uniform initial distribution
  model.pi = Eigen::VectorXd::Constant(N, 1.0 / N);

  // Random transition matrix (row-stochastic)
  model.A = Eigen::MatrixXd::Random(N, N).cwiseAbs();
  for (int i = 0; i < N; ++i) model.A.row(i) /= model.A.row(i).sum();

  // K-means-like initialisation: partition data into N segments
  model.states.resize(N);
  const int seg_len = T / N;
  for (int j = 0; j < N; ++j) {
    const int start = j * seg_len;
    const int end   = (j < N-1) ? (j+1)*seg_len : T;
    Eigen::MatrixXd seg = obs.middleRows(start, end - start);
    model.states[j].mu    = seg.colwise().mean().transpose();
    model.states[j].sigma = ((seg.rowwise() - seg.colwise().mean())
                               .transpose() * (seg.rowwise() - seg.colwise().mean()))
                              / std::max(1, end - start - 1)
                            + kMinVariance * Eigen::MatrixXd::Identity(D, D);
  }

  // ---------------------------------------------------------------------------
  // EM iterations
  // ---------------------------------------------------------------------------
  double prev_ll = std::numeric_limits<double>::lowest();

  for (int iter = 0; iter < kMaxEmIter; ++iter) {
    // --- E-step: Forward ---
    Eigen::MatrixXd log_alpha = ForwardLog(model, obs);

    // --- E-step: Backward ---
    Eigen::MatrixXd log_beta(T, N);
    log_beta.row(T-1).setZero();

    for (int t = T-2; t >= 0; --t) {
      for (int i = 0; i < N; ++i) {
        double max_val = kLogEps;
        for (int j = 0; j < N; ++j) {
          double val = std::log(std::max(model.A(i,j), 1e-300)) +
                       model.states[j].LogPdf(obs.row(t+1).transpose()) +
                       log_beta(t+1, j);
          max_val = std::max(max_val, val);
        }
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
          double val = std::log(std::max(model.A(i,j), 1e-300)) +
                       model.states[j].LogPdf(obs.row(t+1).transpose()) +
                       log_beta(t+1, j);
          sum += std::exp(val - max_val);
        }
        log_beta(t, i) = max_val + std::log(std::max(sum, 1e-300));
      }
    }

    // --- Log-likelihood ---
    double ll = 0.0;
    for (int j = 0; j < N; ++j) ll = std::max(ll, log_alpha(T-1, j));
    double sum_ll = 0.0;
    for (int j = 0; j < N; ++j) sum_ll += std::exp(log_alpha(T-1, j) - ll);
    ll += std::log(sum_ll);
    model.log_likelihood = ll;

    if (std::abs(ll - prev_ll) < kEmTol && iter > 0) break;
    prev_ll = ll;

    // --- Gamma: P(Z_t=j | O) ---
    Eigen::MatrixXd log_gamma(T, N);
    for (int t = 0; t < T; ++t) {
      double max_val = kLogEps;
      for (int j = 0; j < N; ++j) max_val = std::max(max_val, log_alpha(t,j) + log_beta(t,j));
      double norm = 0.0;
      for (int j = 0; j < N; ++j) norm += std::exp(log_alpha(t,j) + log_beta(t,j) - max_val);
      for (int j = 0; j < N; ++j)
        log_gamma(t, j) = log_alpha(t,j) + log_beta(t,j) - max_val - std::log(norm);
    }

    // --- M-step: Update π, A, emission params ---
    // π
    for (int j = 0; j < N; ++j) model.pi(j) = std::exp(log_gamma(0, j));
    model.pi /= model.pi.sum();

    // A: ξ_t(i,j) = P(Z_t=i, Z_{t+1}=j | O)
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double max_val = kLogEps;
        for (int t = 0; t < T-1; ++t) {
          double val = log_alpha(t,i) +
                       std::log(std::max(model.A(i,j), 1e-300)) +
                       model.states[j].LogPdf(obs.row(t+1).transpose()) +
                       log_beta(t+1,j);
          max_val = std::max(max_val, val);
        }
        double xi_sum = 0.0;
        for (int t = 0; t < T-1; ++t) {
          double val = log_alpha(t,i) +
                       std::log(std::max(model.A(i,j), 1e-300)) +
                       model.states[j].LogPdf(obs.row(t+1).transpose()) +
                       log_beta(t+1,j);
          xi_sum += std::exp(val - max_val);
        }
        model.A(i,j) = std::max(xi_sum * std::exp(max_val - ll), 1e-10);
      }
      model.A.row(i) /= model.A.row(i).sum();
    }

    // Emission (Gaussian) params
    for (int j = 0; j < N; ++j) {
      Eigen::VectorXd gamma_j(T);
      for (int t = 0; t < T; ++t) gamma_j(t) = std::exp(log_gamma(t, j));
      const double gamma_sum = gamma_j.sum();

      // Update mean
      Eigen::VectorXd new_mu = Eigen::VectorXd::Zero(D);
      for (int t = 0; t < T; ++t) new_mu += gamma_j(t) * obs.row(t).transpose();
      model.states[j].mu = new_mu / std::max(gamma_sum, 1e-12);

      // Update covariance
      Eigen::MatrixXd new_sigma = Eigen::MatrixXd::Zero(D, D);
      for (int t = 0; t < T; ++t) {
        Eigen::VectorXd diff = obs.row(t).transpose() - model.states[j].mu;
        new_sigma += gamma_j(t) * diff * diff.transpose();
      }
      model.states[j].sigma = new_sigma / std::max(gamma_sum, 1e-12)
                              + kMinVariance * Eigen::MatrixXd::Identity(D, D);
    }
  }

  return model;
}

/**
 * Map raw HMM states to RegimeLabel by sorting on mean return (feature 0).
 *
 * @param model  Trained HMMModel.
 * @return       Mapping from raw state index → RegimeLabel.
 */
[[nodiscard]] inline std::vector<RegimeLabel> LabelRegimes(const HMMModel& model) {
  std::vector<std::pair<double, int>> mean_return_per_state;
  for (int j = 0; j < model.N; ++j) {
    mean_return_per_state.push_back({model.states[j].mu(0), j});
  }
  std::sort(mean_return_per_state.begin(), mean_return_per_state.end());

  std::vector<RegimeLabel> labels(model.N);
  for (int rank = 0; rank < model.N; ++rank) {
    labels[mean_return_per_state[rank].second] = static_cast<RegimeLabel>(rank);
  }
  return labels;
}

}  // namespace quant
