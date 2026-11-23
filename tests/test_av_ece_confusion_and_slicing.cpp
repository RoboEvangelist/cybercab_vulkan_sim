#include "AvValidationMetrics.hpp"
#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace cybercab {
namespace {
using AvCalibrationSlicingTest = testutil::SharedGpuMetricsTest;
using testutil::EceAccum; using testutil::fmtProp;
using testutil::MotLaneMonitor; using testutil::SafetyConfusionMatrix;
using testutil::SliceStats;
using SC = SemanticClass;

constexpr int kEvalF = 44;
constexpr float kMaxR = 42.f, kPedG = 3.5f, kVehG = 5.5f, kSigG = 6.0f;
constexpr float kNearM = 20.f, kMidM = 35.f, kFarM = 50.f, kBndR = 52.f;
constexpr double kMaxEceNom = 0.28, kMaxOceNom = 0.22, kMaxCostNom = 36.0;
constexpr double kMinNearNom = 0.52, kMinMidNom = 0.60;
constexpr double kMinFrontNom = 0.68, kMinSig2Nom = 0.58;
constexpr double kMaxEceNight = 0.46, kMaxOceEnv = 0.30, kMaxCostNight = 45.0;
constexpr double kMinNearNight = 0.45, kMinMidNight = 0.45;
constexpr double kMinFrontNight = 0.60, kMinSig2Night = 0.55;
constexpr double kMinRecNight = 0.45, kMinPrecEnv = 0.65, kMaxEceSoil = 0.30;
constexpr double kMaxCostSoil = 50.0, kMinNearSoil = 0.38, kMinMidSoil = 0.38;
constexpr double kMinFrontSoil = 0.55, kMinSig2Soil = 0.60, kMinRecSoil = 0.42;

SimConfig mkSliceCfg(const char* v, const char* m, bool night, bool soil) {
  SimConfig c =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  c.pipelineVersion = v; c.initialResMode = m;
  if (night) { c.lightingMode = "night_glare"; c.photonRccbMosaic = true; }
  if (soil) { c.cameras[0].soiled = true; }
  return c;
}

double runCalibrationSliceCase(VulkanContext& ctx, const char* v, const char* m,
                               bool night, bool soil, const char* tag) {
  const SimConfig cfg = mkSliceCfg(v, m, night, soil); WorldEngine w(cfg);
  CameraRigRenderer r; r.init(ctx, cfg, w);
  PerceptionPipeline p(cfg); p.init(ctx, r); TrajectoryPlanner pl(cfg);
  const float dt = 1.0f / cfg.controlHz;
  auto allSt = w.staticOBBs();
  for (float zA : {3.2f, 6.2f})
    allSt.push_back({"SIG_HEAD", 6.18f, 5.4f,
                     WorldEngine::kLoopHalfZ - 6.2f + zA, 0.4f, 0.7f, 0.4f,
                     0.0f, 0.0f, 0.0f, SC::SignalSign, "SIG"});
  PerceptionFrameOutput out{}; PlannerCommand cmd{};
  EceAccum ece; SafetyConfusionMatrix cm; MotLaneMonitor mot;
  SliceStats nearS, midS, farS, frontS, sideS;
  double recSum = 0.0, precSum = 0.0; int nEv = 0, misT = 0;
  for (int s = 0; s < kEvalF; ++s) {
    r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, nullptr);
    p.runFrame(ctx, r, w, dt, true, out, nullptr); pl.plan(w, out, cmd);
    EXPECT_FALSE(w.lastContact().occurred) << tag << " step=" << s;
    if (s < 2) { w.stepAgents(dt); w.stepEgoAndPhysics(dt, cmd, {}); continue; }
    ++nEv; recSum += out.eval.detRecall; precSum += out.eval.detPrecision;
    misT += out.eval.misclassifiedTrucks; mot.update(w, out.tracks);
    const EgoState& ego = w.ego();
    const auto dyn = w.dynamicGroundTruthOBBs();
    std::vector<OBB> gtCm, gtBnd, gtFar;
    for (const auto& g : dyn) {
      const float rng = std::hypot(g.cx - ego.pos.x, g.cz - ego.pos.z);
      if (rng <= kMaxR) gtCm.push_back(g);
      else if (rng <= kBndR) gtBnd.push_back(g);
      if (rng <= kFarM) gtFar.push_back(g);
    }
    for (const auto& st : allSt) {
      if (st.semantic != SC::SignalSign) continue;
      const float rng = std::hypot(st.cx - ego.pos.x, st.cz - ego.pos.z);
      if (rng <= kMaxR) gtCm.push_back(st);
      else if (rng <= kBndR) gtBnd.push_back(st);
    }
    std::vector<uint8_t> gtUsed(gtCm.size(), 0);
    for (const auto& d : out.detections3D) {
      if (d.rangeM > kMaxR) continue;
      int best = -1; float bestD = kSigG;
      for (size_t gi = 0; gi < gtCm.size(); ++gi) {
        const float g = (gtCm[gi].semantic == SC::Pedestrian) ? kPedG
                      : (gtCm[gi].semantic == SC::Vehicle) ? kVehG : kSigG;
        const float dist = std::hypot(d.worldPos.x - gtCm[gi].cx,
                                      d.worldPos.z - gtCm[gi].cz);
        if (dist < std::min(bestD, g)) {
          bestD = dist; best = static_cast<int>(gi);
        }
      }
      if (best < 0) {
        bool bnd = false;
        for (const auto& bg : gtBnd) {
          const float g = (bg.semantic == SC::Pedestrian) ? kPedG
                        : (bg.semantic == SC::Vehicle) ? kVehG : kSigG;
          if (std::hypot(d.worldPos.x - bg.cx, d.worldPos.z - bg.cz) < g) {
            bnd = true; break;
          }
        }
        if (bnd) continue;
      }
      const bool ok = (best >= 0 && gtCm[best].semantic == d.semantic);
      ece.add(d.score, ok);
      const int pc = SafetyConfusionMatrix::clsIdx(d.semantic);
      if (best >= 0) {
        gtUsed[best] = 1;
        cm.add(SafetyConfusionMatrix::clsIdx(gtCm[best].semantic), pc);
      } else { cm.add(0, pc); }
      SliceStats& cg = (d.camIndex <= 2) ? frontS : sideS;
      if (ok) ++cg.tp; else ++cg.fp;
    }
    for (size_t gi = 0; gi < gtCm.size(); ++gi)
      if (!gtUsed[gi])
        cm.add(SafetyConfusionMatrix::clsIdx(gtCm[gi].semantic), 0);
    for (const auto& g : gtFar) {
      const float rng = std::hypot(g.cx - ego.pos.x, g.cz - ego.pos.z);
      SliceStats& ds = (rng < kNearM) ? nearS : ((rng < kMidM) ? midS : farS);
      const float gate = (g.semantic == SC::Pedestrian) ? kPedG : kVehG;
      bool hit = false;
      for (const auto& d : out.detections3D)
        if (d.semantic == g.semantic &&
            std::hypot(d.worldPos.x - g.cx, d.worldPos.z - g.cz) < gate) {
          hit = true; break;
        }
      if (hit) ++ds.tp; else ++ds.fn;
    }
    w.stepAgents(dt); w.stepEgoAndPhysics(dt, cmd, {});
  }
  const std::string pfx = std::string(tag) + "_";
  const double cost = cm.costPerGt();
  testing::Test::RecordProperty(pfx + "ece", fmtProp(ece.ece()));
  testing::Test::RecordProperty(pfx + "oce", fmtProp(ece.overconfError()));
  testing::Test::RecordProperty(pfx + "safety_cost", fmtProp(cost));
  testing::Test::RecordProperty(pfx + "near_rec", fmtProp(nearS.recall()));
  testing::Test::RecordProperty(pfx + "mid_rec", fmtProp(midS.recall()));
  testing::Test::RecordProperty(pfx + "far_rec", fmtProp(farS.recall()));
  testing::Test::RecordProperty(pfx + "front_prec",
                                fmtProp(frontS.precision()));
  testing::Test::RecordProperty(pfx + "sigma2_cov",
                                fmtProp(mot.sigma2Cov()));
  EXPECT_EQ(misT, 0) << tag;
  const double maxEce =
    night ? kMaxEceNight : (soil ? kMaxEceSoil : kMaxEceNom);
  const double maxOce = (night || soil) ? kMaxOceEnv : kMaxOceNom;
  const double maxCost =
    night ? kMaxCostNight : (soil ? kMaxCostSoil : kMaxCostNom);
  const double minNear =
    night ? kMinNearNight : (soil ? kMinNearSoil : kMinNearNom);
  const double minMid =
    night ? kMinMidNight : (soil ? kMinMidSoil : kMinMidNom);
  const double minFront =
    night ? kMinFrontNight : (soil ? kMinFrontSoil : kMinFrontNom);
  const double minSig2 =
    night ? kMinSig2Night : (soil ? kMinSig2Soil : kMinSig2Nom);
  EXPECT_LT(ece.ece(), maxEce) << tag;
  EXPECT_LT(ece.overconfError(), maxOce) << tag;
  EXPECT_LT(cost, maxCost) << tag;
  EXPECT_GE(nearS.recall(), minNear) << tag;
  EXPECT_GE(midS.recall(), minMid) << tag;
  EXPECT_GE(frontS.precision(), minFront) << tag;
  EXPECT_GE(mot.sigma2Cov(), minSig2) << tag;
  if (night || soil) {
    EXPECT_GE(recSum / nEv, night ? kMinRecNight : kMinRecSoil) << tag;
    EXPECT_GE(precSum / nEv, kMinPrecEnv) << tag;
  }
  p.destroy(); r.destroy(); return cost;
}

TEST_F(AvCalibrationSlicingTest,
       EceReliabilitySafetyConfusionAndMultiAxisSlices) {
  EceAccum perf, over;
  for (int i = 0; i < 9; ++i) { perf.add(0.90f, true); } perf.add(0.90f, false);
  EXPECT_NEAR(perf.ece(), 0.0, 1e-6);
  EXPECT_NEAR(perf.overconfError(), 0.0, 1e-6);
  for (int i = 0; i < 6; ++i) over.add(0.90f, true);
  for (int i = 0; i < 4; ++i) over.add(0.90f, false);
  EXPECT_NEAR(over.ece(), 0.30, 1e-4);
  EXPECT_NEAR(over.overconfError(), 0.30, 1e-4);
  using CM = SafetyConfusionMatrix;
  CM pedBg, vehBg, vehPed;
  pedBg.add(CM::clsIdx(SC::Pedestrian), 0);
  vehBg.add(CM::clsIdx(SC::Vehicle), 0);
  vehPed.add(CM::clsIdx(SC::Vehicle), CM::clsIdx(SC::Pedestrian));
  EXPECT_GT(pedBg.weightedCost(), vehBg.weightedCost());
  EXPECT_GT(vehBg.weightedCost(), vehPed.weightedCost());
  runCalibrationSliceCase(s_ctx, "v14", "half", false, false, "v14_half");
  const double v15HalfCost =
    runCalibrationSliceCase(s_ctx, "v15", "half", false, false, "v15_half");
  runCalibrationSliceCase(s_ctx, "v15", "full", false, false, "v15_full");
  runCalibrationSliceCase(s_ctx, "v15", "cybercab_binned", false, false,
                          "v15_cybercab_binned");
  runCalibrationSliceCase(s_ctx, "v15", "half", true, false, "v15_night");
  const double soiledCost =
    runCalibrationSliceCase(s_ctx, "v15", "half", false, true, "v15_soiled");
  EXPECT_GT(soiledCost, v15HalfCost);
}
}  // namespace
}  // namespace cybercab
