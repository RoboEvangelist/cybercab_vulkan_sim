#include "MetricsEvalHelpers.hpp"
#include "core/GpuProfiler.hpp"
#include "perception/BEVFormer.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace cybercab {
namespace {
using AllResValidationTest = testutil::SharedGpuMetricsTest;
using testutil::fmtProp;
using Clk = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;
using SC = SemanticClass;

constexpr int kProfSlots = 64, kWarm = 4, kFullF = 36, kNatF = 10;
constexpr int kHazStd = 48, kHazNat = 30, kSirenNat = 24;
constexpr int kMinHits = 4, kMinCor = 2, kMinObs = 100;
constexpr float kEvalR = 42.0f, kDetR = 44.0f, kBndR = 48.0f, kPrG = 5.0f;
constexpr float kSlackM = 0.5f, kVoxTol = 0.05f, kPadM = 1.80f;
constexpr double kMinRec = 0.50, kMinPrec = 0.65, kMinV14Ap = 0.36;
constexpr double kMinV15Ap = 0.40, kMaxRmse = 2.8, kMinVoxP = 0.50;
constexpr double kBudgetMs = 27.8;

struct EvalRig {
  VulkanContext& ctx; SimConfig cfg; WorldEngine w; CameraRigRenderer r;
  PerceptionPipeline p; TrajectoryPlanner pl;
  EvalRig(VulkanContext& c, const char* v, const char* m, const char* haz = "")
    : ctx(c), cfg(mkCfg(v, m)), w(cfg), p(cfg), pl(cfg) {
    if (std::strcmp(haz, "jaywalk") == 0) w.triggerJaywalkerCutIn();
    else if (std::strcmp(haz, "occluded") == 0) w.triggerOccludedDartOut();
    else if (std::strcmp(haz, "hardbrake") == 0) w.triggerLeadHardBrake();
    else if (std::strcmp(haz, "siren") == 0) w.triggerEmergencySiren();
    r.init(ctx, cfg, w); p.init(ctx, r);
  }
  ~EvalRig() { p.destroy(); r.destroy(); }
  static SimConfig mkCfg(const char* v, const char* m) {
    SimConfig c =
      SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    c.pipelineVersion = v; c.initialResMode = m; return c;
  }
  void step(int s, GpuProfiler* pr, PerceptionFrameOutput& o,
            PlannerCommand& cmd) {
    const float dt = 1.0f / cfg.controlHz;
    r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, pr);
    p.runFrame(ctx, r, w, dt, true, o, pr); pl.plan(w, o, cmd);
  }
  void adv(const PlannerCommand& cmd) {
    const float dt = 1.0f / cfg.controlHz; w.stepAgents(dt);
    w.stepEgoAndPhysics(dt, cmd, {}); EXPECT_FALSE(w.lastContact().occurred);
  }
};

void evalQuality(VulkanContext& ctx, const char* ver, const char* mode,
                 int evalF, bool profRun) {
  EvalRig er(ctx, ver, mode); const SimConfig& cfg = er.cfg;
  GpuProfiler prof; if (profRun) prof.init(ctx, kProfSlots);
  GpuProfiler* pr = profRun ? &prof : nullptr;
  PerceptionFrameOutput out{}; PlannerCommand cmd{}; testutil::PrAccum acc{};
  std::vector<OBB> gt, reqV, bndV;
  double rec = 0, prec = 0, rmse = 0, gpuS = 0, walS = 0, ap = 0, f1 = 0;
  float maxErr = 0.0f, maxAge = 0.0f; size_t dTot = 0u, dOk = 0u;
  int nEv = 0, misT = 0, maxH = 0, maxCd = 0, maxCt = 0;
  for (int s = 0; s < kWarm + evalF; ++s) {
    const auto t0 = Clk::now(); er.step(s, pr, out, cmd);
    const double walMs = Ms(Clk::now() - t0).count();
    if (s == 0 && profRun) {
      EXPECT_GT(out.bevSummary.observedCells, kMinObs);
      EXPECT_EQ(out.gpuStageMs.count("1c_v15_freq_split"), 1u);
      EXPECT_EQ(out.gpuStageMs.count("5_v15_static_mem"), 1u);
      EXPECT_EQ(out.gpuStageMs.count("6_v15_gaussian_splat"), 1u);
    }
    if (s < kWarm) { er.adv(cmd); continue; }
    rec += out.eval.detRecall; prec += out.eval.detPrecision;
    rmse += out.eval.trackPosRmseM; misT += out.eval.misclassifiedTrucks;
    gpuS += out.eval.totalPipelineMs; walS += walMs;
    for (const auto& d : out.detections3D) {
      const size_t ci = static_cast<size_t>(d.camIndex);
      ASSERT_TRUE(d.camIndex >= 0 && ci < cfg.cameras.size());
      const float lm = std::min(cfg.rtdetrMaxRangeM, cfg.cameras[ci].maxRangeM);
      EXPECT_GT(d.rangeM, 0.0f); EXPECT_LE(d.rangeM, lm + kSlackM);
      maxCd = std::max(maxCd, d.corroboratingCamCount);
    }
    for (const auto& t : out.tracks) {
      maxH = std::max(maxH, t.hits); EXPECT_TRUE(std::isfinite(t.posSigmaM));
      maxAge = std::max(maxAge, t.timeSinceUpdateSec);
      maxCt = std::max(maxCt, t.corroboratingCamCount);
    }
    er.w.collectDynamicGroundTruthOBBs(gt); reqV.clear(); bndV.clear();
    const Mat4 wIso = BEVFormer::worldFromIsoEgo(er.w.ego());
    for (const auto& v : out.activeVoxels) {
      const Vec3 ex = wIso.transformPoint({v.egoX, v.egoY, v.wy});
      maxErr = std::max(maxErr, std::hypot(v.wx - ex.x, v.wz - ex.z));
      if (v.sem != SC::Vehicle && v.sem != SC::Pedestrian) continue;
      ++dTot; for (const auto& b : gt)
        if (testutil::isPointInOBB(b, v.wx, v.wz, kPadM)) { ++dOk; break; }
    }
    const EgoState& e = er.w.ego();
    for (const auto& g : gt) {
      if (g.semantic != SC::Vehicle) continue;
      const float d = std::hypot(g.cx - e.pos.x, g.cz - e.pos.z);
      if (d <= kBndR) (d <= kEvalR ? reqV : bndV).push_back(g);
    }
    testutil::accumulatePrCls(SC::Vehicle, out.detections3D, reqV, bndV, kPrG,
      kDetR, acc);
    ++nEv; er.adv(cmd);
  }
  testutil::summarizePrCurve(acc, ap, f1);
  const double voxP = dTot > 0u ? static_cast<double>(dOk) / dTot : 0.0;
  const std::string pfx = std::string(ver) + "_" + mode + "_";
  testing::Test::RecordProperty(pfx + "ap", fmtProp(ap));
  testing::Test::RecordProperty(pfx + "rmse", fmtProp(rmse / nEv));
  testing::Test::RecordProperty(pfx + "vox_p", fmtProp(voxP));
  if (profRun) {
    testing::Test::RecordProperty(pfx + "gpu_ms", fmtProp(gpuS / nEv));
    testing::Test::RecordProperty(pfx + "wall_ms", fmtProp(walS / nEv));
    EXPECT_LT(gpuS / nEv, kBudgetMs);
  }
  const double minAp =
    (std::strcmp(ver, "v15") == 0) ? kMinV15Ap : kMinV14Ap;
  EXPECT_GT(rec / nEv, kMinRec) << pfx; EXPECT_GT(prec / nEv, kMinPrec) << pfx;
  EXPECT_GT(ap, minAp) << pfx; EXPECT_LT(rmse / nEv, kMaxRmse) << pfx;
  EXPECT_EQ(misT, 0) << pfx; EXPECT_LT(maxErr, kVoxTol) << pfx;
  EXPECT_GT(voxP, kMinVoxP) << pfx; EXPECT_GE(maxH, kMinHits) << pfx;
  EXPECT_GE(maxCd, kMinCor) << pfx; EXPECT_GE(maxCt, kMinCor) << pfx;
  EXPECT_LE(maxAge, cfg.btMaxTimeLostSec) << pfx;
  if (profRun) prof.destroy();
}

void runHazard(VulkanContext& ctx, const char* ver, const char* mode,
               const char* haz, int steps) {
  EvalRig er(ctx, ver, mode, haz);
  PerceptionFrameOutput out{}; PlannerCommand cmd{};
  bool seenSiren = false; int misT = 0;
  for (int s = 0; s < steps; ++s) {
    er.step(s, nullptr, out, cmd);
    if (cmd.behaviorState == "YIELDING_SIREN") seenSiren = true;
    misT += out.eval.misclassifiedTrucks; er.adv(cmd);
  }
  if (std::strcmp(haz, "hardbrake") == 0) { EXPECT_EQ(misT, 0); }
  if (std::strcmp(haz, "siren") == 0) { EXPECT_TRUE(seenSiren); }
}

TEST_F(AllResValidationTest, V14AllFourResolutionsAndV15FullPrVoxelCorrob) {
  evalQuality(s_ctx, "v14", "half", kFullF, false);
  evalQuality(s_ctx, "v14", "full", kFullF, false);
  evalQuality(s_ctx, "v14", "cybercab_binned", kFullF, false);
  evalQuality(s_ctx, "v14", "cybercab", kNatF, false);
  evalQuality(s_ctx, "v15", "full", kFullF, true);
}

TEST_F(AllResValidationTest,
       AllFourResolutionsClosedLoopHazardsZeroCollisions) {
  for (const char* m : {"half", "full", "cybercab_binned", "cybercab"}) {
    const bool isNat = (std::strcmp(m, "cybercab") == 0);
    const int nHaz = isNat ? kHazNat : kHazStd;
    runHazard(s_ctx, "v15", m, "jaywalk", nHaz);
    runHazard(s_ctx, "v15", m, "occluded", nHaz);
    runHazard(s_ctx, "v15", m, "hardbrake", nHaz);
    runHazard(s_ctx, "v15", m, "siren", isNat ? kSirenNat : kHazStd);
  }
  for (const char* m : {"full", "cybercab_binned"}) {
    runHazard(s_ctx, "v14", m, "jaywalk", kHazStd);
    runHazard(s_ctx, "v14", m, "occluded", kHazStd);
    runHazard(s_ctx, "v14", m, "hardbrake", kHazStd);
  }
}
}  // namespace
}  // namespace cybercab
