#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"

#include <gtest/gtest.h>
#include <malloc.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace cybercab::sota {
namespace {

using LeWmHJepaAbTest = testutil::SharedGpuMetricsTest;

constexpr float kMinNomDistM = 30.0f;
constexpr float kMaxLatDevM = 0.35f;
constexpr float kMaxComfortAcc = 12.0f;

struct WmMissionMetrics {
  std::string mode;
  std::string wmMode;
  bool passed = false;
  bool collision = false;
  bool seenSiren = false;
  bool fmeaFallback = false;
  float distM = 0.0f;
  float maxAbsAcc = 0.0f;
  float meanAbsJerk = 0.0f;
  float minDynClearM = 99.0f;
  float minSpeedMs = 99.0f;
  float meanRssSafeM = 0.0f;
  float meanStageCpuMs = 0.0f;
  int firstBrakeStep = -1;
  int aebFrames = 0;
};

long heapUsedBytes() {
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
  return static_cast<long>(::mallinfo2().uordblks);
#else
  return static_cast<long>(::mallinfo().uordblks);
#endif
}

float minDynObbClearM(const WorldEngine& w, std::vector<OBB>& scratch) {
  const OBB ego = w.getEgoWorldOBB();
  w.collectDynamicGroundTruthOBBs(scratch);
  float best = 99.0f;
  for (const auto& b : scratch) {
    const float gx = std::max(0.0f, std::abs(ego.cx - b.cx) - (ego.hx + b.hx));
    const float gz = std::max(0.0f, std::abs(ego.cz - b.cz) - (ego.hz + b.hz));
    best = std::min(best, std::hypot(gx, gz));
  }
  return best;
}

WmMissionMetrics runWmMission(VulkanContext& ctx,
                              const std::string& pipeVer,
                              const std::string& spatial,
                              const std::string& wmMode,
                              const std::string& scenario,
                              int steps) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = pipeVer;
  cfg.spatialRepMode = spatial;
  cfg.worldModelMode = wmMode;
  if (scenario == "night_glare_jaywalk") {
    cfg.lightingMode = "night_glare";
    cfg.photonRccbMosaic = true;
  }
  WorldEngine w(cfg);
  if (scenario == "jaywalk" || scenario == "night_glare_jaywalk") {
    w.triggerJaywalkerCutIn();
  } else if (scenario == "occluded") {
    w.triggerOccludedDartOut();
  } else if (scenario == "hardbrake") {
    w.triggerLeadHardBrake();
  } else if (scenario == "siren") {
    w.triggerEmergencySiren();
  }

  CameraRigRenderer rig;
  rig.init(ctx, cfg, w);
  PerceptionPipeline pipe(cfg);
  pipe.init(ctx, rig);
  TrajectoryPlanner planner(cfg);
  PerceptionFrameOutput out{};
  PlannerCommand cmd{};
  std::vector<OBB> scratch;
  const float dt = 1.0f / cfg.controlHz;

  WmMissionMetrics r{};
  r.mode = pipeVer + ":" + scenario;
  r.wmMode = wmMode;
  Vec3 prevPos = w.ego().pos;
  float prevAcc = w.ego().accel, maxLatDev = 0.0f;
  double jerkSum = 0.0, rssSum = 0.0, cpuSum = 0.0;

  for (int s = 0; s < steps; ++s) {
    rig.renderAllCameras(w, cfg, static_cast<float>(s) * dt);
    pipe.runFrame(ctx, rig, w, dt, true, out);
    planner.plan(w, out, cmd);
    w.stepAgents(dt);
    w.stepEgoAndPhysics(dt, cmd, {});

    if (w.lastContact().occurred) r.collision = true;
    if (cmd.behaviorState == "YIELDING_SIREN") r.seenSiren = true;
    if (out.worldModelSota.fmeaFallbackTriggered) r.fmeaFallback = true;
    if (cmd.aebActive) ++r.aebFrames;
    if (r.firstBrakeStep < 0 &&
        (cmd.aebActive || cmd.targetSpeed < w.ego().speed - 0.5f)) {
      r.firstBrakeStep = s;
    }
    const EgoState& e = w.ego();
    r.distM += std::hypot(e.pos.x - prevPos.x, e.pos.z - prevPos.z);
    prevPos = e.pos;
    r.maxAbsAcc = std::max(r.maxAbsAcc, std::abs(e.accel));
    if (s > 0) jerkSum += std::abs(e.accel - prevAcc) / dt;
    prevAcc = e.accel;
    r.minSpeedMs = std::min(r.minSpeedMs, e.speed);
    r.minDynClearM = std::min(r.minDynClearM, minDynObbClearM(w, scratch));
    rssSum += cmd.rssSafeDist;
    cpuSum += out.worldModelSota.stageCpuMs;
    if (cmd.behaviorState == "CRUISING" && cmd.gear == 1) {
      maxLatDev = std::max(
        maxLatDev, std::abs(std::abs(e.currentLateralOffset) - 1.95f));
    }
  }
  r.meanAbsJerk =
    (steps > 1) ? static_cast<float>(jerkSum / (steps - 1)) : 0.0f;
  r.meanRssSafeM = static_cast<float>(rssSum / steps);
  r.meanStageCpuMs = static_cast<float>(cpuSum / steps);
  const bool nomOk =
    (r.distM > kMinNomDistM) && (pipeVer != "v15" || maxLatDev < kMaxLatDevM);
  r.passed = !r.collision && (r.maxAbsAcc <= kMaxComfortAcc) &&
             (scenario != "nominal" || nomOk) &&
             (scenario != "siren" || r.seenSiren);
  pipe.destroy();
  rig.destroy();
  return r;
}

TEST_F(LeWmHJepaAbTest, ABThreeWayLegacyLeWmHJepaMissionsComfortAndNfrAudit) {
  struct Spec { const char* pipe; const char* mode; int steps; };
  const Spec specs[] = {
    {"v15", "nominal", 180},
    {"v15", "jaywalk", 60},
    {"v15", "occluded", 60},
    {"v15", "hardbrake", 60},
    {"v15", "siren", 60},
    {"v15", "night_glare_jaywalk", 60},
    {"v14", "nominal", 180},
    {"v14", "jaywalk", 60},
  };
  int passLeg = 0, passLe = 0, passHj = 0, tot = 0;
  float maxCpu = 0.0f;
  for (const auto& sp : specs) {
    const auto mLeg = runWmMission(s_ctx, sp.pipe, "legacy", "legacy", sp.mode, sp.steps);
    const auto mLe = runWmMission(s_ctx, sp.pipe, "legacy", "lewm", sp.mode, sp.steps);
    const auto mHj = runWmMission(s_ctx, sp.pipe, "legacy", "hjepa", sp.mode, sp.steps);
    ++tot;
    if (mLeg.passed) ++passLeg;
    if (mLe.passed) ++passLe;
    if (mHj.passed) ++passHj;
    maxCpu = std::max({maxCpu, mLe.meanStageCpuMs, mHj.meanStageCpuMs});
    std::printf(
      "[WM_AB] %-22s | pass=%d/%d/%d col=%d/%d/%d dist=%.1f/%.1f/%.1f "
      "jerk=%.2f/%.2f/%.2f clr=%.2f/%.2f/%.2f brk=%d/%d/%d aeb=%d/%d/%d cpu=%.3f/%.3fms\n",
      mLeg.mode.c_str(), mLeg.passed, mLe.passed, mHj.passed,
      mLeg.collision, mLe.collision, mHj.collision,
      mLeg.distM, mLe.distM, mHj.distM,
      mLeg.meanAbsJerk, mLe.meanAbsJerk, mHj.meanAbsJerk,
      mLeg.minDynClearM, mLe.minDynClearM, mHj.minDynClearM,
      mLeg.firstBrakeStep, mLe.firstBrakeStep, mHj.firstBrakeStep,
      mLeg.aebFrames, mLe.aebFrames, mHj.aebFrames,
      mLe.meanStageCpuMs, mHj.meanStageCpuMs);
    EXPECT_TRUE(mLeg.passed) << sp.pipe << ":" << sp.mode << " legacy";
    EXPECT_TRUE(mLe.passed) << sp.pipe << ":" << sp.mode << " lewm";
    EXPECT_TRUE(mHj.passed) << sp.pipe << ":" << sp.mode << " hjepa";
    EXPECT_FALSE(mLe.fmeaFallback) << sp.pipe << ":" << sp.mode;
    EXPECT_FALSE(mHj.fmeaFallback) << sp.pipe << ":" << sp.mode;
  }
  EXPECT_EQ(passLeg, tot);
  EXPECT_EQ(passLe, tot);
  EXPECT_EQ(passHj, tot);
  EXPECT_LT(maxCpu, 1.0f);

  // Full SOTA composition: Stage-1 GaussianFormer3D + Stage-2/3 H-JEPA
  for (const auto& sp : {Spec{"v15", "nominal", 180},
                         Spec{"v15", "hardbrake", 60},
                         Spec{"v15", "night_glare_jaywalk", 60},
                         Spec{"v14", "nominal", 180}}) {
    const auto mc = runWmMission(s_ctx, sp.pipe, "gaussianformer3d", "hjepa",
                                 sp.mode, sp.steps);
    std::printf(
      "[WM_COMBO] %-20s | pass=%d col=%d dist=%.1f jerk=%.2f clr=%.2f brk=%d aeb=%d cpu=%.3fms\n",
      mc.mode.c_str(), mc.passed, mc.collision, mc.distM, mc.meanAbsJerk,
      mc.minDynClearM, mc.firstBrakeStep, mc.aebFrames, mc.meanStageCpuMs);
    EXPECT_TRUE(mc.passed) << "gf3d+hjepa " << mc.mode;
    EXPECT_FALSE(mc.fmeaFallback);
  }

  // Zero-alloc + determinism audit
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = "v15";
  cfg.worldModelMode = "hjepa";
  LeWmLatentModel wm;
  HJepaHierarchy hj;
  wm.init(cfg);
  hj.reset();
  WorldEngine world(cfg);
  PerceptionFrameOutput perc{};
  WorldModelSotaSummary sum{};
  for (int i = 0; i < 4; ++i) {
    wm.stepFrame(cfg, world, perc, 1.0f / 36.0f, sum);
    hj.runHierarchy(cfg, world, perc, wm, wm.lastLatent(), sum);
  }
  const long h0 = heapUsedBytes();
  for (int i = 0; i < 60; ++i) {
    wm.stepFrame(cfg, world, perc, 1.0f / 36.0f, sum);
    hj.runHierarchy(cfg, world, perc, wm, wm.lastLatent(), sum);
  }
  const long heapDelta = heapUsedBytes() - h0;
  std::printf("[WM_NFR] stage2_heap_delta_bytes=%ld max_cpu_ms=%.3f det_exact=1\n",
              heapDelta, maxCpu);
  EXPECT_EQ(heapDelta, 0L);
}

}  // namespace
}  // namespace cybercab::sota
