#include "config/SimConfig.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "sensors/PhotonModel.hpp"
#include "world/WorldEngine.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cybercab {
namespace {
constexpr float kShadowR = 0.012f, kShadowG = 0.015f, kShadowB = 0.020f;
constexpr float kMinLuma = 0.005f;
constexpr float kFullWellElectrons = 40000.0f, kReadNoiseElectrons = 3.0f;
constexpr float kDarkMeanDn = 10.0f, kMaxVarRatio = 0.55f;
constexpr int kNumCams = 8;
constexpr float kMaxDispClamp = 1.0f, kMinGlareRad = 1.5f;
constexpr float kMinShadowRad = 0.001f, kMaxShadowBandRad = 0.25f;
constexpr float kMinShadowMeanRad = 0.005f;
constexpr int kClosedLoopSteps = 60;
constexpr double kBudgetMs = 27.8;

TEST(V15HdrRccbTest, LightingModeValidationAndRccbDarkShadowSnrGain) {
  const auto badPath =
    std::filesystem::temp_directory_path() / "v15_bad_lighting.yaml";
  std::ofstream(badPath) << "simulation:\n  lighting_mode: \"twilight\"\n";
  EXPECT_THROW(SimConfig::loadFromYaml(badPath.string()), std::runtime_error);
  std::filesystem::remove(badPath);

  const float clearRad =
    photon::rccbClearRadiance(kShadowR, kShadowG, kShadowB);
  EXPECT_GT(clearRad, photon::kRccbClearGain * kMinLuma);
  const auto np =
    photon::noiseParamsFromPhysical(kFullWellElectrons, kReadNoiseElectrons);
  const float rgbVar = photon::noiseVarianceDn(kDarkMeanDn, np);
  const float rccbVar = photon::rccbEffectiveNoiseVarianceDn(kDarkMeanDn, np);
  EXPECT_LT(rccbVar, rgbVar * kMaxVarRatio);
}

TEST(V15HdrRccbTest, NightGlareHdrLinearPreservesHighlightsAndShadowDetail) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = "v15";
  cfg.lightingMode = "night_glare";
  cfg.photonRccbMosaic = true;
  cfg.photonNoiseEnabled = false;
  WorldEngine world(cfg);
  VulkanContext ctx;
  ctx.init(false, nullptr);
  CameraRigRenderer rig;
  rig.init(ctx, cfg, world);
  rig.renderAllCameras(world, cfg, 0.0f, nullptr);

  const size_t nPx = static_cast<size_t>(kNumCams) *
                     static_cast<size_t>(rig.camWidth()) *
                     static_cast<size_t>(rig.camHeight());
  std::vector<Vec4> disp(nPx), lin(nPx);
  rig.cameraTensorSSBO().downloadVector(disp);
  rig.cameraLinearSSBO().downloadVector(lin);

  float maxDisp = 0.0f, maxLin = 0.0f;
  double shadowSum = 0.0;
  size_t shadowCnt = 0u;
  for (size_t i = 0; i < nPx; ++i) {
    maxDisp = std::max({maxDisp, disp[i].x, disp[i].y, disp[i].z});
    const float lx = lin[i].x * photon::kFullWellRadiance;
    const float ly = lin[i].y * photon::kFullWellRadiance;
    const float lz = lin[i].z * photon::kFullWellRadiance;
    const float lMax = std::max({lx, ly, lz});
    maxLin = std::max(maxLin, lMax);
    if (lMax > kMinShadowRad && lMax < kMaxShadowBandRad) {
      shadowSum += lMax;
      ++shadowCnt;
    }
  }
  const double meanShadow =
    shadowCnt > 0u ? (shadowSum / static_cast<double>(shadowCnt)) : 0.0;
  RecordProperty("night_max_lin_rad", std::to_string(maxLin));
  RecordProperty("night_mean_shadow_rad", std::to_string(meanShadow));
  EXPECT_LE(maxDisp, kMaxDispClamp);
  EXPECT_GT(maxLin, kMinGlareRad);
  EXPECT_GT(shadowCnt, 0u);
  EXPECT_GT(meanShadow, static_cast<double>(kMinShadowMeanRad));

  rig.destroy();
  ctx.cleanup();
}

TEST(V15HdrRccbTest, NightGlareShadowPedestrianClosedLoopZeroCollision) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = "v15";
  cfg.lightingMode = "night_glare";
  cfg.photonRccbMosaic = true;
  WorldEngine world(cfg);
  world.triggerJaywalkerCutIn();
  VulkanContext ctx;
  ctx.init(false, nullptr);
  CameraRigRenderer rig;
  rig.init(ctx, cfg, world);
  PerceptionPipeline perc(cfg);
  perc.init(ctx, rig);
  TrajectoryPlanner planner(cfg);
  GpuProfiler profiler;
  profiler.init(ctx, static_cast<uint32_t>(kClosedLoopSteps));

  PerceptionFrameOutput out{};
  PlannerCommand cmd{};
  const float dt = 1.0f / cfg.controlHz;
  bool collision = false;
  double maxGpuMs = 0.0;
  for (int s = 0; s < kClosedLoopSteps; ++s) {
    rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt, &profiler);
    perc.runFrame(ctx, rig, world, dt, cfg.occEnableShadows, out, &profiler);
    planner.plan(world, out, cmd);
    world.stepAgents(dt);
    world.stepEgoAndPhysics(dt, cmd, {});
    if (world.lastContact().occurred) collision = true;
    maxGpuMs = std::max(maxGpuMs, out.eval.totalPipelineMs);
  }
  RecordProperty("night_glare_max_gpu_ms", std::to_string(maxGpuMs));
  EXPECT_FALSE(collision);
  EXPECT_LT(maxGpuMs, kBudgetMs);

  profiler.destroy();
  perc.destroy();
  rig.destroy();
  ctx.cleanup();
}

}  // namespace
}  // namespace cybercab
