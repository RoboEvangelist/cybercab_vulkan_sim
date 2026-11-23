#include "MetricsEvalHelpers.hpp"
#include "core/GpuProfiler.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/v15/FrequencySplit.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>

namespace cybercab {
namespace {
using V15CybercabCameraResTest = testutil::SharedGpuMetricsTest;

constexpr int kNumCams = 8, kProfSlots = 64, kScale2x = 2;
constexpr int kWarmF = 3, kDriveF = 7, kResF = 5, kSoilF = 4;
constexpr int kNw = kCybercabNativeCamW, kNh = kCybercabNativeCamH;
constexpr VkDeviceSize kVec4 = 4u;
constexpr float kBaseSigma = 4.0f, kRefW = 320.0f, kIntrEps = 1e-3f;
constexpr float kAspectEps = 1e-4f, kKnockPitch = 1.6f, kKnockRoll = -1.4f;
constexpr float kCTol = 0.25f, kOffMap = -350.0f, kLaneLatM = -1.9f;
constexpr float kLoopStartU = 0.03f, kLoopStepU = 0.025f, kMinCleanW = 0.85f;
constexpr float kMaxCleanP = 0.30f, kMaxSoilW = 0.55f, kMinSoilP = 0.40f;
constexpr double kBinnedBudgetMs = 27.8;

VkDeviceSize expectedTensorBytes(int w, int h) {
  return static_cast<VkDeviceSize>(kNumCams) * w * h * kVec4 * sizeof(float);
}

SimConfig loadV15WithMode(const char* mode) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = "v15"; cfg.initialResMode = mode; return cfg;
}

double runFrames(VulkanContext& ctx, WorldEngine& w, CameraRigRenderer& r,
                 PerceptionPipeline& p, const SimConfig& cfg, int n,
                 GpuProfiler* prof, PerceptionFrameOutput& out) {
  const float dt = 1.0f / cfg.controlHz;
  double gSum = 0.0;
  for (int s = 0; s < n; ++s) {
    r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, prof);
    p.runFrame(ctx, r, w, dt, true, out, prof);
    gSum += out.eval.totalPipelineMs;
  }
  return gSum / n;
}

void verifyEmptySoilAtPreset(VulkanContext& ctx, const char* mode, bool soil) {
  SimConfig cfg = loadV15WithMode(mode); cfg.cameras[0].soiled = soil;
  WorldEngine world(cfg);
  for (auto& v : world.mutableVehicles()) {
    v.pos = {kOffMap, 0.0f, kOffMap}; v.speed = v.vx = v.vz = 0.0f;
  }
  for (auto& p : world.mutablePedestrians()) {
    p.x = p.ax = p.bx = p.targetX = p.z = p.az = p.bz = p.targetZ = kOffMap;
  }
  CameraRigRenderer rig; rig.init(ctx, cfg, world);
  PerceptionPipeline pipe(cfg); pipe.init(ctx, rig);
  PerceptionFrameOutput out{};
  (void)runFrames(ctx, world, rig, pipe, cfg, kSoilF, nullptr, out);
  ASSERT_EQ(out.soilCameras.size(), static_cast<size_t>(kNumCams));
  if (!soil) {
    EXPECT_EQ(out.eval.ghostVehDetections, 0);
    EXPECT_EQ(out.eval.ghostPedDetections, 0);
    for (const auto& sc : out.soilCameras) {
      EXPECT_GT(sc.tokenWeight, kMinCleanW);
      EXPECT_LT(sc.meanSoilProb, kMaxCleanP);
    }
  } else {
    EXPECT_LT(out.soilCameras[0].tokenWeight, kMaxSoilW);
    EXPECT_GT(out.soilCameras[0].meanSoilProb, kMinSoilP);
  }
  pipe.destroy(); rig.destroy();
}

TEST(V15CybercabCameraResConfigTest,
     AllFourResolutionPresetsLoadValidateAndCycle) {
  const SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  const SimConfig def = SimConfig::defaultConfig();
  const auto pCfg = cfg.resolutionPresets();
  const auto pDef = def.resolutionPresets();
  constexpr int kExpW[] = {kLegacyHalfCamW, kLegacyFullCamW,
                           kCybercabBinnedCamW, kNw};
  constexpr int kExpH[] = {kLegacyHalfCamH, kLegacyFullCamH,
                           kCybercabBinnedCamH, kNh};
  EXPECT_EQ(cfg.initialResMode, "half"); EXPECT_EQ(def.initialResMode, "half");
  for (size_t i = 0; i < pCfg.size(); ++i) {
    EXPECT_EQ(pCfg[i].width, kExpW[i]); EXPECT_EQ(pCfg[i].height, kExpH[i]);
    EXPECT_STREQ(pCfg[i].mode, pDef[i].mode);
    EXPECT_EQ(pDef[i].width, kExpW[i]); EXPECT_EQ(pDef[i].height, kExpH[i]);
  }
  for (const char* m : {"half", "full", "cybercab_binned", "cybercab"})
    EXPECT_TRUE(SimConfig::isValidResMode(m));
  EXPECT_FALSE(SimConfig::isValidResMode("4k"));
  EXPECT_FALSE(SimConfig::isValidResMode(""));
  const float nAsp = static_cast<float>(pCfg[3].width) / pCfg[3].height;
  const float bAsp = static_cast<float>(pCfg[2].width) / pCfg[2].height;
  EXPECT_NEAR(nAsp, bAsp, kAspectEps);
  EXPECT_EQ(pCfg[3].width, kScale2x * pCfg[2].width);
  EXPECT_EQ(pCfg[3].height, kScale2x * pCfg[2].height);
  const CameraIntrinsics intr =
    cfg.cameras[0].intrinsics(pCfg[3].width, pCfg[3].height);
  EXPECT_FLOAT_EQ(intr.cx, 0.5f * static_cast<float>(kNw));
  EXPECT_FLOAT_EQ(intr.cy, 0.5f * static_cast<float>(kNh));
  EXPECT_NEAR(intr.fx, intr.fy, kIntrEps);
  int w = cfg.resolveInitialRes().width, h = cfg.resolveInitialRes().height;
  EXPECT_EQ(w, kLegacyHalfCamW); EXPECT_EQ(h, kLegacyHalfCamH);
  for (size_t i = 0; i < pCfg.size(); ++i) {
    const auto nxt = cfg.nextResolutionPreset(w, h);
    const auto& expP = pCfg[(i + 1u) % pCfg.size()];
    EXPECT_EQ(nxt.width, expP.width); EXPECT_EQ(nxt.height, expP.height);
    w = nxt.width; h = nxt.height;
  }
  const auto fp = v15::FreqSplitParams::forConfig(kNw, cfg);
  EXPECT_FLOAT_EQ(fp.sigmaPx, kBaseSigma * (static_cast<float>(kNw) / kRefW));
}

TEST_F(V15CybercabCameraResTest,
       CybercabNativeAndBinnedResolutionsRenderDetectCalibrateAndResize) {
  SimConfig cfg = loadV15WithMode("cybercab");
  WorldEngine world(cfg);
  CameraRigRenderer rig; rig.init(s_ctx, cfg, world);
  PerceptionPipeline pipe(cfg); pipe.init(s_ctx, rig);
  TrajectoryPlanner planner(cfg);
  GpuProfiler profiler; profiler.init(s_ctx, kProfSlots);
  ASSERT_EQ(rig.camWidth(), kNw); ASSERT_EQ(rig.camHeight(), kNh);
  EXPECT_EQ(rig.cameraLinearSSBO().size(), expectedTensorBytes(kNw, kNh));
  PerceptionFrameOutput out{}; PlannerCommand cmd{};
  const float dt = 1.0f / cfg.controlHz;
  double gpuSum = 0.0, wallSum = 0.0; int evalCnt = 0;
  for (int s = 0; s < kWarmF + kDriveF; ++s) {
    const auto t0 = std::chrono::steady_clock::now();
    rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt, &profiler);
    pipe.runFrame(s_ctx, rig, world, dt, true, out, &profiler);
    planner.plan(world, out, cmd);
    world.stepAgents(dt); world.stepEgoAndPhysics(dt, cmd, {});
    const auto t1 = std::chrono::steady_clock::now();
    EXPECT_FALSE(world.lastContact().occurred);
    if (s >= kWarmF) {
      gpuSum += out.eval.totalPipelineMs;
      wallSum += std::chrono::duration<double, std::milli>(t1 - t0).count();
      ++evalCnt;
    }
  }
  const double gMean = gpuSum / evalCnt, wMean = wallSum / evalCnt;
  RecordProperty("native_2896x1876_gpu_ms", testutil::fmtProp(gMean));
  RecordProperty("native_2896x1876_wall_ms", testutil::fmtProp(wMean));
  EXPECT_TRUE(out.v15.active); EXPECT_FALSE(out.detections3D.empty());
  EXPECT_FALSE(out.activeVoxels.empty());
  EXPECT_FALSE(out.v15Gaussians.empty());
  cfg.cameraKnockRpyDeg = {kKnockRoll, kKnockPitch, 0.0f}; pipe.reset();
  const auto& loop = world.centerline();
  for (int f = 0; f < v15::CameraSelfCalib::kWindowFrames; ++f) {
    const float u = std::fmod(kLoopStartU + kLoopStepU * static_cast<float>(f),
                              1.0f);
    EgoState& e = world.mutableEgo();
    const Vec3 tan = loop.getTangentAt(u);
    e.pos = loop.getPointAt(u) + Vec3{tan.z, 0.0f, -tan.x} * kLaneLatM;
    e.yaw = std::atan2(tan.x, tan.z); e.pitch = 0.0f; e.roll = 0.0f;
    rig.renderAllCameras(world, cfg, static_cast<float>(f) * dt, &profiler);
    pipe.runFrame(s_ctx, rig, world, dt, true, out, &profiler);
  }
  const float cp = out.v15.calibPitchDeg, cr = out.v15.calibRollDeg;
  RecordProperty("native_calib_pitch_deg", testutil::fmtProp(cp));
  RecordProperty("native_calib_roll_deg", testutil::fmtProp(cr));
  EXPECT_NEAR(cp, kKnockPitch, kCTol); EXPECT_NEAR(cr, kKnockRoll, kCTol);
  cfg.cameraKnockRpyDeg = {0.0f, 0.0f, 0.0f};
  for (const char* m : {"cybercab_binned", "full", "half"}) {
    cfg.initialResMode = m; const auto tgt = cfg.resolveInitialRes();
    rig.resize(tgt.width, tgt.height); pipe.onCameraResized(rig); pipe.reset();
    EXPECT_EQ(rig.camWidth(), tgt.width);
    EXPECT_EQ(rig.camHeight(), tgt.height);
    EXPECT_EQ(rig.cameraLinearSSBO().size(),
              expectedTensorBytes(tgt.width, tgt.height));
    const double ms =
      runFrames(s_ctx, world, rig, pipe, cfg, kResF, &profiler, out);
    if (std::string(m) == "cybercab_binned") {
      RecordProperty("binned_1448x938_gpu_ms", testutil::fmtProp(ms));
      EXPECT_LT(ms, kBinnedBudgetMs);
    }
    EXPECT_FALSE(out.detections3D.empty());
    EXPECT_FALSE(out.v15Gaussians.empty());
  }
  profiler.destroy(); pipe.destroy(); rig.destroy();
}

TEST_F(V15CybercabCameraResTest, ScaleInvariantSoilAndEmptyRoadAcrossPresets) {
  for (const char* m : {"half", "cybercab_binned", "cybercab"}) {
    verifyEmptySoilAtPreset(s_ctx, m, false);
    verifyEmptySoilAtPreset(s_ctx, m, true);
  }
}
}  // namespace
}  // namespace cybercab
