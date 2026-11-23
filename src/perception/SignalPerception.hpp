#pragma once
// Pure-vision traffic light state classifier, maneuver-aware lane associator,
// and safety-asymmetric HMM/FSM temporal filter.

#include "sensors/CameraRigRenderer.hpp"
#include "world/WorldTypes.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace cybercab::signal {

enum class SignalState : uint8_t {
  Red = 0,
  Yellow = 1,
  Green = 2,
  Unknown = 3,
};

enum class SignalManeuver : uint8_t {
  Straight = 0,
  LeftTurn = 1,
};

enum class SignalDecision : uint8_t {
  Stop = 0,
  Yield = 1,
  Proceed = 2,
};

struct SignalHeadSpec {
  Vec3 worldPos{};
  int targetLane = 0;
  SignalManeuver role = SignalManeuver::Straight;
};

struct SignalObs {
  SignalState rawState = SignalState::Unknown;
  std::array<float, 4> probs{0.0f, 0.0f, 0.0f, 1.0f};
  float confidence = 0.0f;
  int headIndex = -1;
  int targetLane = 0;
  SignalManeuver role = SignalManeuver::Straight;
  bool visible = false;
};

struct SignalAssocResult {
  int chosenHeadIdx = -1;
  bool valid = false;
  SignalObs obs{};
};

struct SignalConfusionMatrix {
  static constexpr int kN = 4;
  static constexpr double kCostRedRed = 0.0, kCostRedYel = 50.0;
  static constexpr double kCostRedGrn = 1000.0, kCostRedUnk = 35.0;
  static constexpr double kCostYelRed = 10.0, kCostYelYel = 0.0;
  static constexpr double kCostYelGrn = 200.0, kCostYelUnk = 15.0;
  static constexpr double kCostGrnRed = 25.0, kCostGrnYel = 10.0;
  static constexpr double kCostGrnGrn = 0.0, kCostGrnUnk = 20.0;
  static constexpr double kCostUnkRed = 40.0, kCostUnkYel = 20.0;
  static constexpr double kCostUnkGrn = 300.0, kCostUnkUnk = 0.0;
  static constexpr double kCost[kN][kN] = {
    {kCostRedRed, kCostRedYel, kCostRedGrn, kCostRedUnk},
    {kCostYelRed, kCostYelYel, kCostYelGrn, kCostYelUnk},
    {kCostGrnRed, kCostGrnYel, kCostGrnGrn, kCostGrnUnk},
    {kCostUnkRed, kCostUnkYel, kCostUnkGrn, kCostUnkUnk},
  };
  std::array<std::array<int, kN>, kN> m_{};

  void add(SignalState gt, SignalState pred);
  int redToGreenCount() const { return m_[0][2]; }
  int total() const;
  double accuracy() const;
  double weightedCost() const;
};

class VisionSignalClassifier {
public:
  static SignalObs classifyHeadFromRgba(const std::vector<uint8_t>& rgba,
                                        int w, int h,
                                        const CameraRigRenderer& rig,
                                        int camIdx, const SignalHeadSpec& head,
                                        int headIdx);

  static std::vector<SignalObs> classifyWorldHeadsFromRig(
    const CameraRigRenderer& rig, const std::vector<SignalHeadSpec>& heads);
};

class SignalLaneAssociator {
public:
  static SignalAssocResult associate(int egoLane, SignalManeuver maneuver,
                                     const std::vector<SignalObs>& heads);
};

class SignalTemporalFilter {
public:
  static constexpr int kRedToGreenConfirmFrames = 3;
  static constexpr int kInitGreenConfirmFrames = 2;
  static constexpr float kRedHmmMinProb = 0.50f;
  static constexpr float kYelHmmMinProb = 0.45f;
  static constexpr float kGrnHmmMinProb = 0.65f;
  static constexpr float kGreenHmmMinProb = 0.72f;
  static constexpr float kUnkHmmMinProb = 0.65f;
  static constexpr float kFastRedConf = 0.45f;

  SignalTemporalFilter() { reset(); }
  void reset();
  SignalState update(const SignalObs& obs);
  SignalState state() const { return filtered_; }
  SignalDecision decision() const;
  const std::array<float, 4>& belief() const { return belief_; }
  int transitions() const { return stateTransitions_; }

private:
  std::array<float, 4> belief_{};
  SignalState filtered_ = SignalState::Unknown;
  int greenStreak_ = 0;
  int redStreak_ = 0;
  int stateTransitions_ = 0;
};

}  // namespace cybercab::signal
