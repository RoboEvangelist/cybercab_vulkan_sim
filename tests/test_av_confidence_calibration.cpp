#include "AvValidationMetrics.hpp"
#include "MetricsEvalHelpers.hpp"
#include "perception/ConfidenceCalibrator.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace cybercab {
namespace {
using AvConfidenceCalibTest = testutil::SharedGpuMetricsTest;
using testutil::EceAccum;
using testutil::fmtProp;
using SC = SemanticClass;

constexpr int kEvalF = 44, kWarmupF = 2, kSynthN = 80, kSynthSplit = 40;
constexpr int kSynthMod = 10, kSynthHiPass = 6, kSynthLoPass = 4;
constexpr int kCorrobTestCams = 3, kNumCams = 8;
constexpr float kMaxR = 42.f, kPedG = 3.5f, kVehG = 5.5f, kSigG = 6.0f;
constexpr float kBndR = 52.f, kMinTemp = 0.25f, kOutLo = 0.01f, kOutHi = 0.99f;
constexpr float kSigX = 6.18f, kSigY = 5.4f, kSigZBaseOff = 6.2f;
constexpr float kSigArmA = 3.2f, kSigArmB = 6.2f;
constexpr float kSigHx = 0.4f, kSigHy = 0.7f, kSigHz = 0.4f;
constexpr float kRawHi = 0.90f, kRawLo = 0.72f, kMonoLo = 0.30f;
constexpr float kMonoMid = 0.60f, kMonoHi = 0.90f, kTestSoil = 0.70f;
constexpr float kScoreSoilTest = 0.80f, kScoreCorrobTest = 0.75f;
constexpr double kMaxDayEce = 0.22, kMaxDayOce = 0.16, kMaxNightEce = 0.28;
constexpr double kMaxNightOce = 0.20, kMaxSoilEce = 0.25, kMaxSoilOce = 0.18;

struct CalibRunStats {
  double calEce = 0.0, rawEce = 0.0, calOce = 0.0, rawOce = 0.0;
};

float recoverRawScore(float calScore, const calib::CalibParams& p) {
  const float zRaw =
    (calib::logit(calScore) - p.bias) * std::max(kMinTemp, p.temperature);
  return clampf(calib::sigmoid(zRaw), kOutLo, kOutHi);
}

float classMatchGate(SemanticClass sem) {
  if (sem == SC::Pedestrian) return kPedG;
  return (sem == SC::Vehicle) ? kVehG : kSigG;
}

SimConfig mkCalibCfg(bool night, bool soil) {
  SimConfig c =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  c.pipelineVersion = "v15";
  c.initialResMode = "half";
  if (night) { c.lightingMode = "night_glare"; c.photonRccbMosaic = true; }
  if (soil) { c.cameras[0].soiled = true; }
  return c;
}

void collectGtInRange(const WorldEngine& w, const std::vector<OBB>& allSt,
                      std::vector<OBB>& gt, std::vector<OBB>& bnd) {
  const EgoState& ego = w.ego();
  auto pushRng = [&](const OBB& g) {
    const float rng = std::hypot(g.cx - ego.pos.x, g.cz - ego.pos.z);
    if (rng <= kMaxR) gt.push_back(g);
    else if (rng <= kBndR) bnd.push_back(g);
  };
  for (const auto& g : w.dynamicGroundTruthOBBs()) pushRng(g);
  for (const auto& st : allSt) if (st.semantic == SC::SignalSign) pushRng(st);
}

bool matchDetToGt(const Detection3D& d, const std::vector<OBB>& gt,
                  const std::vector<OBB>& bnd, int& bestIdx) {
  bestIdx = -1;
  float bestD = kSigG;
  for (size_t gi = 0; gi < gt.size(); ++gi) {
    const float dist =
      std::hypot(d.worldPos.x - gt[gi].cx, d.worldPos.z - gt[gi].cz);
    const float gate = std::min(bestD, classMatchGate(gt[gi].semantic));
    if (dist < gate) {
      bestD = dist;
      bestIdx = static_cast<int>(gi);
    }
  }
  if (bestIdx >= 0) return true;
  for (const auto& bg : bnd) {
    const float dist = std::hypot(d.worldPos.x - bg.cx, d.worldPos.z - bg.cz);
    if (dist < classMatchGate(bg.semantic)) return false;
  }
  return true;
}

void scoreFrameCalib(const PerceptionFrameOutput& out,
                     const std::vector<OBB>& gt, const std::vector<OBB>& bnd,
                     bool night, EceAccum& calEce, EceAccum& rawEce) {
  for (const auto& d : out.detections3D) {
    int best = -1;
    if (d.rangeM > kMaxR || !matchDetToGt(d, gt, bnd, best)) continue;
    const bool ok =
      (best >= 0 && gt[static_cast<size_t>(best)].semantic == d.semantic);
    calEce.add(d.score, ok);
    const int ci = std::clamp(d.camIndex, 0, kNumCams - 1);
    const float sp = out.soilCameras[static_cast<size_t>(ci)].meanSoilProb;
    const auto cp =
      calib::paramsForCondition(night, sp, d.corroboratingCamCount);
    rawEce.add(recoverRawScore(d.score, cp), ok);
  }
}

CalibRunStats runCalibEval(VulkanContext& ctx, bool night, bool soil,
                           const char* tag) {
  const SimConfig cfg = mkCalibCfg(night, soil);
  WorldEngine w(cfg);
  CameraRigRenderer r;
  r.init(ctx, cfg, w);
  PerceptionPipeline p(cfg);
  p.init(ctx, r);
  TrajectoryPlanner pl(cfg);
  const float dt = 1.0f / cfg.controlHz;
  auto allSt = w.staticOBBs();
  for (float zA : {kSigArmA, kSigArmB})
    allSt.push_back({"SIG_HEAD", kSigX, kSigY,
                     WorldEngine::kLoopHalfZ - kSigZBaseOff + zA, kSigHx,
                     kSigHy, kSigHz, 0.f, 0.f, 0.f, SC::SignalSign, "SIG"});
  PerceptionFrameOutput out{};
  PlannerCommand cmd{};
  EceAccum calEce, rawEce;
  for (int s = 0; s < kEvalF; ++s) {
    r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, nullptr);
    p.runFrame(ctx, r, w, dt, true, out, nullptr);
    pl.plan(w, out, cmd);
    EXPECT_FALSE(w.lastContact().occurred) << tag << " step=" << s;
    if (s >= kWarmupF) {
      std::vector<OBB> gt, bnd;
      collectGtInRange(w, allSt, gt, bnd);
      scoreFrameCalib(out, gt, bnd, night, calEce, rawEce);
    }
    w.stepAgents(dt);
    w.stepEgoAndPhysics(dt, cmd, {});
  }
  const CalibRunStats st{calEce.ece(), rawEce.ece(), calEce.overconfError(),
                         rawEce.overconfError()};
  const std::string pfx = std::string(tag) + "_";
  testing::Test::RecordProperty(pfx + "raw_ece", fmtProp(st.rawEce));
  testing::Test::RecordProperty(pfx + "cal_ece", fmtProp(st.calEce));
  testing::Test::RecordProperty(pfx + "raw_oce", fmtProp(st.rawOce));
  testing::Test::RecordProperty(pfx + "cal_oce", fmtProp(st.calOce));
  p.destroy();
  r.destroy();
  return st;
}

TEST_F(AvConfidenceCalibTest,
       TemperatureScalingFitReducesNllAndEceOnHeldOutSplit) {
  std::vector<calib::CalibSample> train, test;
  for (int i = 0; i < kSynthN; ++i) {
    const float raw = (i % 2 == 0) ? kRawHi : kRawLo;
    const bool ok = (i % 2 == 0) ? ((i % kSynthMod) < kSynthHiPass)
                                 : ((i % kSynthMod) < kSynthLoPass);
    (i < kSynthSplit ? train : test).push_back({raw, ok});
  }
  const calib::CalibParams fitP = calib::fitTemperatureNll(train);
  EceAccum eRaw, eFit;
  for (const auto& s : test) {
    eRaw.add(s.rawScore, s.correct);
    eFit.add(calib::calibrateScore(s.rawScore, fitP), s.correct);
  }
  auto cal = [](float v, const calib::CalibParams& p) {
    return calib::calibrateScore(v, p);
  };
  EXPECT_LT(calib::evalNll(test, fitP), calib::evalNll(test, {}));
  EXPECT_LT(eFit.ece(), eRaw.ece());
  EXPECT_LT(eFit.overconfError(), eRaw.overconfError());
  EXPECT_LT(cal(kMonoLo, fitP), cal(kMonoMid, fitP));
  EXPECT_LT(cal(kMonoMid, fitP), cal(kMonoHi, fitP));
  const auto cleanP = calib::paramsForCondition(false, 0.0f, 1);
  const auto soilP = calib::paramsForCondition(false, kTestSoil, 1);
  const auto corrobP = calib::paramsForCondition(false, 0.0f, kCorrobTestCams);
  EXPECT_LT(cal(kScoreSoilTest, soilP), cal(kScoreSoilTest, cleanP));
  EXPECT_GT(cal(kScoreCorrobTest, corrobP), cal(kScoreCorrobTest, cleanP));
}

TEST_F(AvConfidenceCalibTest,
       LiveV15PerConditionCalibrationImprovesDayNightAndSoiled) {
  const auto d = runCalibEval(s_ctx, false, false, "v15_day");
  const auto n = runCalibEval(s_ctx, true, false, "v15_night");
  const auto s = runCalibEval(s_ctx, false, true, "v15_soiled");
  EXPECT_LT(d.calEce, kMaxDayEce);
  EXPECT_LT(d.calOce, kMaxDayOce);
  EXPECT_LT(n.calEce, n.rawEce);
  EXPECT_LT(n.calEce, kMaxNightEce);
  EXPECT_LT(n.calOce, kMaxNightOce);
  EXPECT_LT(s.calEce, kMaxSoilEce);
  EXPECT_LT(s.calOce, kMaxSoilOce);
}
}  // namespace
}  // namespace cybercab
