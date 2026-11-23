#include "perception/SignalPerception.hpp"
#include <algorithm>

namespace cybercab::signal {
namespace {

constexpr int kNumStates = SignalConfusionMatrix::kN;
constexpr int kTierRoleAndLane = 4;
constexpr int kTierRoleOnly = 3;
constexpr int kTierLaneOnly = 1;
constexpr int kTierVisibleBonus = 10;
constexpr int kIdxRed = static_cast<int>(SignalState::Red);
constexpr int kIdxYellow = static_cast<int>(SignalState::Yellow);
constexpr int kIdxGreen = static_cast<int>(SignalState::Green);
constexpr int kIdxUnknown = static_cast<int>(SignalState::Unknown);
constexpr float kEps = 1e-6f;
constexpr float kObsProbFloor = 0.02f;
constexpr std::array<float, kNumStates> kInitBelief{0.45f, 0.15f, 0.15f, 0.25f};
constexpr float kHmmTrans[kNumStates][kNumStates] = {
  {0.840f, 0.005f, 0.080f, 0.075f},
  {0.120f, 0.840f, 0.010f, 0.030f},
  {0.040f, 0.100f, 0.840f, 0.020f},
  {0.400f, 0.150f, 0.200f, 0.250f},
};

int headMatchTier(int egoLane, SignalManeuver maneuver, const SignalObs& head) {
  const bool roleOk = (head.role == maneuver);
  const bool laneOk = (head.targetLane == egoLane);
  int tier = 0;
  if (roleOk && laneOk) {
    tier = kTierRoleAndLane;
  } else if (roleOk) {
    tier = kTierRoleOnly;
  } else if (laneOk) {
    tier = kTierLaneOnly;
  }
  if (tier > 0 && head.visible) {
    tier += kTierVisibleBonus;
  }
  return tier;
}

void hmmPredictAndUpdate(std::array<float, kNumStates>& belief,
                         const std::array<float, kNumStates>& obsProbs) {
  std::array<float, kNumStates> predicted{};
  float norm = 0.0f;
  for (int j = 0; j < kNumStates; ++j) {
    float prior = 0.0f;
    for (int i = 0; i < kNumStates; ++i) {
      prior += belief[static_cast<size_t>(i)] * kHmmTrans[i][j];
    }
    const float obs = std::max(kObsProbFloor, obsProbs[static_cast<size_t>(j)]);
    predicted[static_cast<size_t>(j)] = prior * obs;
    norm += predicted[static_cast<size_t>(j)];
  }
  const float denom = std::max(kEps, norm);
  for (int j = 0; j < kNumStates; ++j) {
    belief[static_cast<size_t>(j)] = predicted[static_cast<size_t>(j)] / denom;
  }
}

SignalState stepAsymmetricFsm(SignalState current, const SignalObs& obs,
                              const std::array<float, kNumStates>& belief,
                              int& greenStreak, int& redStreak) {
  const SignalState raw = obs.rawState;
  const bool fastRed =
    (obs.confidence >= SignalTemporalFilter::kFastRedConf) ||
    (belief[kIdxRed] >= SignalTemporalFilter::kRedHmmMinProb);
  if (raw == SignalState::Red && fastRed) {
    greenStreak = 0;
    ++redStreak;
    return SignalState::Red;
  }
  if (current == SignalState::Red) {
    redStreak = 0;
    const bool strongGreen =
      (raw == SignalState::Green) &&
      (belief[kIdxGreen] >= SignalTemporalFilter::kGreenHmmMinProb);
    greenStreak = strongGreen ? (greenStreak + 1) : 0;
    if (greenStreak >= SignalTemporalFilter::kRedToGreenConfirmFrames) {
      return SignalState::Green;
    }
    return SignalState::Red;
  }
  if (raw == SignalState::Yellow &&
      belief[kIdxYellow] >= SignalTemporalFilter::kYelHmmMinProb) {
    greenStreak = 0;
    return SignalState::Yellow;
  }
  if (raw == SignalState::Green &&
      belief[kIdxGreen] >= SignalTemporalFilter::kGrnHmmMinProb) {
    ++greenStreak;
    if (current == SignalState::Green ||
        greenStreak >= SignalTemporalFilter::kInitGreenConfirmFrames) {
      return SignalState::Green;
    }
    return current;
  }
  if (raw == SignalState::Unknown &&
      belief[kIdxUnknown] >= SignalTemporalFilter::kUnkHmmMinProb) {
    greenStreak = 0;
    return SignalState::Unknown;
  }
  return current;
}

}  // namespace

void SignalConfusionMatrix::add(SignalState gt, SignalState pred) {
  ++m_[static_cast<size_t>(gt)][static_cast<size_t>(pred)];
}

int SignalConfusionMatrix::total() const {
  int tot = 0;
  for (const auto& row : m_) {
    for (int v : row) {
      tot += v;
    }
  }
  return tot;
}

double SignalConfusionMatrix::accuracy() const {
  int correct = 0;
  const int tot = total();
  for (int r = 0; r < kN; ++r) {
    correct += m_[static_cast<size_t>(r)][static_cast<size_t>(r)];
  }
  return tot > 0 ? static_cast<double>(correct) / tot : 0.0;
}

double SignalConfusionMatrix::weightedCost() const {
  double sum = 0.0;
  for (int r = 0; r < kN; ++r) {
    for (int c = 0; c < kN; ++c) {
      sum += kCost[r][c] * m_[static_cast<size_t>(r)][static_cast<size_t>(c)];
    }
  }
  return sum;
}

SignalAssocResult SignalLaneAssociator::associate(
    int egoLane, SignalManeuver maneuver, const std::vector<SignalObs>& heads) {
  int bestIdx = -1;
  int bestTier = 0;
  for (size_t i = 0; i < heads.size(); ++i) {
    const int tier = headMatchTier(egoLane, maneuver, heads[i]);
    if (tier == 0) {
      continue;
    }
    const bool tieBreak =
      (tier == bestTier && bestIdx >= 0 &&
       heads[i].confidence > heads[static_cast<size_t>(bestIdx)].confidence);
    if (tier > bestTier || tieBreak) {
      bestTier = tier;
      bestIdx = static_cast<int>(i);
    }
  }
  if (bestIdx < 0) {
    return {};
  }
  return {bestIdx, true, heads[static_cast<size_t>(bestIdx)]};
}

void SignalTemporalFilter::reset() {
  belief_ = kInitBelief;
  filtered_ = SignalState::Unknown;
  greenStreak_ = 0;
  redStreak_ = 0;
  stateTransitions_ = 0;
}

SignalState SignalTemporalFilter::update(const SignalObs& obs) {
  hmmPredictAndUpdate(belief_, obs.probs);
  const SignalState prev = filtered_;
  filtered_ =
    stepAsymmetricFsm(filtered_, obs, belief_, greenStreak_, redStreak_);
  if (filtered_ != prev) {
    ++stateTransitions_;
  }
  return filtered_;
}

SignalDecision SignalTemporalFilter::decision() const {
  if (filtered_ == SignalState::Green) {
    return SignalDecision::Proceed;
  }
  return (filtered_ == SignalState::Yellow) ? SignalDecision::Yield
                                            : SignalDecision::Stop;
}

}  // namespace cybercab::signal
