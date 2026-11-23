#include "MetricsEvalHelpers.hpp"
#include "perception/SignalPerception.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace cybercab {
namespace {
using AvVisionSignalTest = testutil::SharedGpuMetricsTest;
using testutil::fmtProp;
using namespace signal;

constexpr float kHeadX = 6.18f, kHeadY = 5.40f, kLeftHeadOffZ = 3.0f;
constexpr float kApproachX = -18.0f, kApproachLatZ = 1.9f, kYawHalf = 0.5f;
constexpr float kRenderT1 = 0.1f, kRenderT2 = 0.2f;
constexpr float kConfRed = 0.94f, kConfGlitch = 0.92f, kConfGrn = 0.95f;
constexpr float kConfFastRed = 0.90f, kConfUnk = 0.90f, kOtherClasses = 3.0f;
constexpr int kLeftTurnLane = 1, kStraightLane = 0, kSplitHoldF = 4;
constexpr int kRedWarmF = 60, kGlitchHoldF = 15, kGlitchBurstF = 2;
constexpr int kRedRecoverF = 10, kMaxRgSearchF = 12, kMaxRgLatF = 5;
constexpr int kGrnWarmF = 20, kMaxGrSearchF = 6, kMaxGrLatF = 2, kOccF = 10;
constexpr double kMinVisionAcc = 0.90, kPerfectAssocAcc = 1.0;

SimConfig mkSigCfg(const char* resMode, bool night) {
  SimConfig c =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  c.pipelineVersion = "v15";
  c.initialResMode = resMode;
  if (night) {
    c.lightingMode = "night_glare";
    c.photonRccbMosaic = true;
  }
  return c;
}

std::vector<SignalHeadSpec> buildSouthHeads() {
  return {
    {{kHeadX, kHeadY, WorldEngine::kLoopHalfZ - kLeftHeadOffZ},
     kLeftTurnLane, SignalManeuver::LeftTurn},
    {{kHeadX, kHeadY, WorldEngine::kLoopHalfZ},
     kStraightLane, SignalManeuver::Straight},
  };
}

SignalState toSigState(SignalPhase p) {
  return (p == SignalPhase::Red)
           ? SignalState::Red
           : ((p == SignalPhase::Yellow) ? SignalState::Yellow
                                         : SignalState::Green);
}

SignalDecision expectedDecision(SignalPhase p) {
  return (p == SignalPhase::Red) ? SignalDecision::Stop
                                 : SignalDecision::Proceed;
}

SignalObs mkObs(SignalState st, float conf, int lane = 0,
                SignalManeuver role = SignalManeuver::Straight) {
  SignalObs o{};
  o.rawState = st;
  o.confidence = conf;
  o.targetLane = lane;
  o.role = role;
  o.visible = (st != SignalState::Unknown);
  const float rem = std::max(0.0f, 1.0f - conf) / kOtherClasses;
  o.probs = {rem, rem, rem, rem};
  o.probs[static_cast<size_t>(st)] = conf;
  return o;
}

void feedObs(SignalTemporalFilter& flt, SignalState st, float conf, int n) {
  for (int i = 0; i < n; ++i) { flt.update(mkObs(st, conf)); }
}

int framesUntilState(SignalTemporalFilter& flt, SignalState st, float conf,
                     int maxF) {
  for (int i = 1; i <= maxF; ++i) {
    if (flt.update(mkObs(st, conf)) == st) { return i; }
  }
  return 0;
}

void evalPhasePair(CameraRigRenderer& rig,
                   const std::vector<SignalHeadSpec>& heads, WorldEngine& w,
                   const SimConfig& cfg, SignalPhase l0, SignalPhase l1,
                   bool withFilter, SignalConfusionMatrix& cm, int& assocOk,
                   int& assocTot) {
  w.setSplitSignalPhase(l0, l1);
  rig.renderAllCameras(w, cfg, withFilter ? kRenderT2 : kRenderT1, nullptr);
  const auto obs =
    VisionSignalClassifier::classifyWorldHeadsFromRig(rig, heads);
  ASSERT_EQ(obs.size(), heads.size());
  for (const auto& o : obs) { EXPECT_TRUE(o.visible); }
  const auto a0 = SignalLaneAssociator::associate(
    kStraightLane, SignalManeuver::Straight, obs);
  const auto a1 = SignalLaneAssociator::associate(
    kLeftTurnLane, SignalManeuver::LeftTurn, obs);
  ASSERT_TRUE(a0.valid && a1.valid);
  cm.add(toSigState(l0), a0.obs.rawState);
  cm.add(toSigState(l1), a1.obs.rawState);
  assocTot += 2;
  if (!withFilter) {
    if (a0.obs.rawState == toSigState(l0)) { ++assocOk; }
    if (a1.obs.rawState == toSigState(l1)) { ++assocOk; }
    return;
  }
  SignalTemporalFilter f0, f1;
  for (int k = 0; k < kSplitHoldF; ++k) {
    f0.update(a0.obs);
    f1.update(a1.obs);
  }
  if (f0.decision() == expectedDecision(l0)) { ++assocOk; }
  if (f1.decision() == expectedDecision(l1)) { ++assocOk; }
  EXPECT_EQ(f0.decision(), expectedDecision(l0));
  EXPECT_EQ(f1.decision(), expectedDecision(l1));
}

TEST_F(AvVisionSignalTest,
       VisionOnlySignalClassifierAndSplitPhaseLaneAssociation) {
  const auto heads = buildSouthHeads();
  SignalConfusionMatrix cm;
  int assocOk = 0, assocTot = 0;
  const struct { const char* res; bool night; } setups[] = {
    {"half", false}, {"full", false}, {"half", true}};
  for (const auto& su : setups) {
    const SimConfig cfg = mkSigCfg(su.res, su.night);
    WorldEngine w(cfg);
    EgoState& e = w.mutableEgo();
    e.pos = {kApproachX, 0.0f, WorldEngine::kLoopHalfZ + kApproachLatZ};
    e.yaw = kPi * kYawHalf;
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, w);
    for (SignalPhase ph :
         {SignalPhase::Green, SignalPhase::Yellow, SignalPhase::Red}) {
      evalPhasePair(rig, heads, w, cfg, ph, ph, false, cm, assocOk, assocTot);
    }
    evalPhasePair(rig, heads, w, cfg, SignalPhase::Red, SignalPhase::Green,
                  true, cm, assocOk, assocTot);
    evalPhasePair(rig, heads, w, cfg, SignalPhase::Green, SignalPhase::Red,
                  true, cm, assocOk, assocTot);
    rig.destroy();
  }
  const double assocAcc =
    assocTot > 0 ? static_cast<double>(assocOk) / assocTot : 0.0;
  RecordProperty("sig_vision_acc", fmtProp(cm.accuracy()));
  RecordProperty("sig_red_to_green_cnt", std::to_string(cm.redToGreenCount()));
  RecordProperty("sig_safety_cost", fmtProp(cm.weightedCost()));
  RecordProperty("sig_split_assoc_acc", fmtProp(assocAcc));
  EXPECT_EQ(cm.redToGreenCount(), 0);
  EXPECT_GE(cm.accuracy(), kMinVisionAcc);
  EXPECT_EQ(assocAcc, kPerfectAssocAcc);
}

TEST_F(AvVisionSignalTest,
       AsymmetricHmmFsmBlocksRedToGreenGlitchesWithBoundedLatency) {
  SignalTemporalFilter flt;
  int glitchLeaks = 0;
  feedObs(flt, SignalState::Red, kConfRed, kRedWarmF);
  EXPECT_EQ(flt.decision(), SignalDecision::Stop);
  if (framesUntilState(flt, SignalState::Green, kConfGlitch, 1) > 0) {
    ++glitchLeaks;
  }
  feedObs(flt, SignalState::Red, kConfRed, kGlitchHoldF);
  if (framesUntilState(flt, SignalState::Green, kConfGlitch,
                       kGlitchBurstF) > 0) {
    ++glitchLeaks;
  }
  feedObs(flt, SignalState::Red, kConfRed, kRedRecoverF);
  EXPECT_EQ(glitchLeaks, 0);

  const int rgFrames =
    framesUntilState(flt, SignalState::Green, kConfGrn, kMaxRgSearchF);
  EXPECT_GE(rgFrames, SignalTemporalFilter::kRedToGreenConfirmFrames);
  EXPECT_LE(rgFrames, kMaxRgLatF);
  feedObs(flt, SignalState::Green, kConfGrn, kGrnWarmF);
  const int grFrames =
    framesUntilState(flt, SignalState::Red, kConfFastRed, kMaxGrSearchF);
  EXPECT_GE(grFrames, 1);
  EXPECT_LE(grFrames, kMaxGrLatF);
  SignalTemporalFilter occFlt;
  for (int i = 0; i < kOccF; ++i) {
    occFlt.update(mkObs(SignalState::Unknown, kConfUnk));
    EXPECT_NE(occFlt.decision(), SignalDecision::Proceed);
  }
  RecordProperty("sig_glitch_leaks", std::to_string(glitchLeaks));
  RecordProperty("sig_red_to_green_lat_frames", std::to_string(rgFrames));
  RecordProperty("sig_green_to_red_lat_frames", std::to_string(grFrames));
}
}  // namespace
}  // namespace cybercab
