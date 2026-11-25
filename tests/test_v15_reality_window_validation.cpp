#include "MetricsEvalHelpers.hpp"
#include "app/V15HudOverlay.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/v15/GaussianWorldModel.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace cybercab {
namespace {
using namespace v15;

using V15RealityWindowValidationTest = testutil::SharedGpuMetricsTest;

struct RgbaPixel {
  int r = 0;
  int g = 0;
  int b = 0;
  int a = 0;
};

RgbaPixel samplePixel(const std::vector<uint8_t>& rgba, int u, int v) {
  const int cu = std::clamp(u, 0, GaussianWorldModel::kRealityW - 1);
  const int cv = std::clamp(v, 0, GaussianWorldModel::kRealityH - 1);
  const size_t idx =
    (static_cast<size_t>(cv) * GaussianWorldModel::kRealityW + cu) * 4u;
  return {rgba[idx + 0u], rgba[idx + 1u], rgba[idx + 2u], rgba[idx + 3u]};
}

int maxPatchDiffFromBaseline(const std::vector<uint8_t>& rgba,
                             const std::vector<uint8_t>& baseRgba,
                             int u, int v, int radiusPx,
                             RgbaPixel& peakPixelOut) {
  int bestDiff = -1;
  for (int dv = -radiusPx; dv <= radiusPx; ++dv) {
    for (int du = -radiusPx; du <= radiusPx; ++du) {
      const RgbaPixel p = samplePixel(rgba, u + du, v + dv);
      const RgbaPixel b = samplePixel(baseRgba, u + du, v + dv);
      const int diff =
        std::abs(p.r - b.r) + std::abs(p.g - b.g) + std::abs(p.b - b.b);
      if (diff > bestDiff) {
        bestDiff = diff;
        peakPixelOut = p;
      }
    }
  }
  return bestDiff;
}

TEST_F(V15RealityWindowValidationTest,
       LiveWorldToPixelSemanticAlignmentLegacyAndGf3d) {
  SimConfig baseCfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  baseCfg.pipelineVersion = "v15";
  const CameraSpec frontCam = baseCfg.cameras[0];
  const CameraIntrinsics intr = frontCam.intrinsics(
    GaussianWorldModel::kRealityW, GaussianWorldModel::kRealityH);

  std::vector<uint8_t> emptyRgba;
  Vec3 vehWorld{};
  Vec3 pedWorld{};
  {
    baseCfg.spatialRepMode = "legacy";
    WorldEngine emptyWorld(baseCfg);
    emptyWorld.mutableVehicles().clear();
    emptyWorld.mutablePedestrians().clear();
    const Mat4 wFromIso = BEVFormer::worldFromIsoEgo(emptyWorld.ego());
    vehWorld = wFromIso.transformPoint({13.0f, 1.5f, 0.72f});
    pedWorld = wFromIso.transformPoint({9.0f, -1.4f, 0.85f});

    CameraRigRenderer rig;
    rig.init(s_ctx, baseCfg, emptyWorld);
    PerceptionPipeline pipe(baseCfg);
    pipe.init(s_ctx, rig);
    PerceptionFrameOutput out{};
    const float dt = 1.0f / baseCfg.controlHz;
    for (int f = 0; f < 4; ++f) {
      rig.renderAllCameras(emptyWorld, baseCfg, static_cast<float>(f) * dt);
      pipe.runFrame(s_ctx, rig, emptyWorld, dt, true, out);
    }
    emptyRgba = out.v15RealityRgba;
    ASSERT_EQ(emptyRgba.size(), v15_hud::kRealityExpectedBytes);
    pipe.destroy();
    rig.destroy();
  }

  for (const char* mode : {"legacy", "gaussianformer3d"}) {
    SimConfig cfg = baseCfg;
    cfg.spatialRepMode = mode;
    WorldEngine world(cfg);
    world.mutableVehicles().clear();
    world.mutablePedestrians().clear();

    AIVehicle lead{};
    lead.id = "val_sedan";
    lead.type = "sedan";
    lead.pos = {vehWorld.x, 0.0f, vehWorld.z};
    lead.yaw = world.ego().yaw;
    lead.speed = 0.0f;
    lead.baseSpeed = 0.0f;
    lead.color = {0.23f, 0.51f, 0.96f};
    world.mutableVehicles().push_back(lead);

    Pedestrian ped{};
    ped.id = "val_ped";
    ped.x = pedWorld.x;
    ped.z = pedWorld.z;
    ped.baseSpeed = 0.0f;
    ped.jaywalking = true;
    world.mutablePedestrians().push_back(ped);

    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    PerceptionPipeline pipe(cfg);
    pipe.init(s_ctx, rig);
    PerceptionFrameOutput out{};
    const float dt = 1.0f / cfg.controlHz;
    for (int f = 0; f < 4; ++f) {
      rig.renderAllCameras(world, cfg, static_cast<float>(f) * dt);
      pipe.runFrame(s_ctx, rig, world, dt, true, out);
    }
    ASSERT_EQ(out.v15RealityRgba.size(), v15_hud::kRealityExpectedBytes);
    EXPECT_EQ(out.gaussianFormer3D.active,
              std::string(mode) == "gaussianformer3d");

    const Mat4 camFromWorld =
      (BEVFormer::worldFromIsoEgo(world.ego()) *
       frontCam.extrinsicEgoFromCamISO()).inverseRigid();
    const Vec3 vehCam = camFromWorld.transformPoint(vehWorld);
    const Vec3 pedCam = camFromWorld.transformPoint(pedWorld);
    ASSERT_GT(vehCam.z, 1.0f);
    ASSERT_GT(pedCam.z, 1.0f);
    const int vehU = static_cast<int>(
      std::lround(intr.cx + intr.fx * vehCam.x / vehCam.z));
    const int vehV = static_cast<int>(
      std::lround(intr.cy + intr.fy * vehCam.y / vehCam.z));
    const int pedU = static_cast<int>(
      std::lround(intr.cx + intr.fx * pedCam.x / pedCam.z));
    const int pedV = static_cast<int>(
      std::lround(intr.cy + intr.fy * pedCam.y / pedCam.z));

    EXPECT_GT(vehU, 10);
    EXPECT_LT(vehU, GaussianWorldModel::kRealityW - 10);
    EXPECT_GT(pedU, 10);
    EXPECT_LT(pedU, GaussianWorldModel::kRealityW - 10);

    RgbaPixel vehPix{};
    const int vehDiff = maxPatchDiffFromBaseline(
      out.v15RealityRgba, emptyRgba, vehU, vehV, 12, vehPix);
    const std::string vehDiag =
      "mode=" + std::string(mode) + " tracks=" + std::to_string(out.tracks.size()) +
      " vehUV=(" + std::to_string(vehU) + "," + std::to_string(vehV) + ")" +
      " vehPix=(" + std::to_string(vehPix.r) + "," + std::to_string(vehPix.g) +
      "," + std::to_string(vehPix.b) + ")";
    EXPECT_GT(vehDiff, 40) << vehDiag;
    EXPECT_GT(vehPix.r, vehPix.g) << vehDiag;
    EXPECT_GT(vehPix.r, vehPix.b) << vehDiag;

    RgbaPixel pedPix{};
    const int pedDiff = maxPatchDiffFromBaseline(
      out.v15RealityRgba, emptyRgba, pedU, pedV, 12, pedPix);
    const std::string pedDiag =
      "mode=" + std::string(mode) +
      " pedUV=(" + std::to_string(pedU) + "," + std::to_string(pedV) + ")" +
      " pedPix=(" + std::to_string(pedPix.r) + "," + std::to_string(pedPix.g) +
      "," + std::to_string(pedPix.b) + ")";
    EXPECT_GT(pedDiff, 40) << pedDiag;
    EXPECT_GT(pedPix.g, pedPix.r) << pedDiag;
    EXPECT_GT(pedPix.g, pedPix.b) << pedDiag;

    pipe.destroy();
    rig.destroy();
  }
}

TEST_F(V15RealityWindowValidationTest,
       OffAxisRotatedEllipsoidGpuParityAndFailureCases) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  const CameraIntrinsics intr = cfg.cameras[0].intrinsics(
    GaussianWorldModel::kRealityW, GaussianWorldModel::kRealityH);

  GaussianWorldModel gw;
  gw.initSplatGpu(s_ctx);

  const Gaussian3DGpu offAxisRot{
    {2.4f, 0.4f, 11.0f, 0.96f},
    {0.70f, 0.14f, 0.16f, 0.0f},
    {0.0f, 0.0f, 0.55f, 1.0f},
    {0.98f, 0.05f, 0.05f, 1.0f}};
  Vec2 uv{};
  float cov[3]{};
  float depth = 0.0f;
  GaussianWorldModel::projectEwa2D(
    offAxisRot, Mat4::identity(), intr, uv, cov, depth);
  const float theta = 0.5f * std::atan2(2.0f * cov[1], cov[0] - cov[2]);
  const Vec2 majorDir{std::cos(theta), std::sin(theta)};
  const Vec2 minorDir{-std::sin(theta), std::cos(theta)};

  gw.setGaussiansForTest({offAxisRot});
  VkCommandBuffer cmd = s_ctx.beginSingleTimeCommands();
  gw.dispatchRealitySplat(cmd, Mat4::identity(), intr);
  s_ctx.endSingleTimeCommands(cmd);

  std::vector<uint8_t> rgba;
  gw.downloadRealityViewRGBA(rgba);
  ASSERT_EQ(rgba.size(), v15_hud::kRealityExpectedBytes);

  int bestRed = -1, bestU = -1, bestV = -1;
  for (int y = 0; y < GaussianWorldModel::kRealityH; ++y) {
    for (int x = 0; x < GaussianWorldModel::kRealityW; ++x) {
      const RgbaPixel p = samplePixel(rgba, x, y);
      const int redExcess = p.r - p.b;
      if (redExcess > bestRed) {
        bestRed = redExcess;
        bestU = x;
        bestV = y;
      }
    }
  }
  EXPECT_GT(bestRed, 150);
  EXPECT_LE(std::abs(bestU - static_cast<int>(std::lround(uv.x))), 2);
  EXPECT_LE(std::abs(bestV - static_cast<int>(std::lround(uv.y))), 2);

  const RgbaPixel majorSample = samplePixel(
    rgba, static_cast<int>(std::lround(uv.x + majorDir.x * 10.0f)),
    static_cast<int>(std::lround(uv.y + majorDir.y * 10.0f)));
  const RgbaPixel minorSample = samplePixel(
    rgba, static_cast<int>(std::lround(uv.x + minorDir.x * 10.0f)),
    static_cast<int>(std::lround(uv.y + minorDir.y * 10.0f)));
  EXPECT_GT(majorSample.r - majorSample.b, minorSample.r - minorSample.b + 40);

  // Empty world -> clean dark gradient without foreground splats.
  gw.setGaussiansForTest({});
  cmd = s_ctx.beginSingleTimeCommands();
  gw.dispatchRealitySplat(cmd, Mat4::identity(), intr);
  s_ctx.endSingleTimeCommands(cmd);
  gw.downloadRealityViewRGBA(rgba);
  ASSERT_EQ(rgba.size(), v15_hud::kRealityExpectedBytes);
  for (size_t i = 0; i < rgba.size(); i += 4u) {
    EXPECT_EQ(rgba[i + 3u], 255u);
    EXPECT_LT(rgba[i], 60u);
  }

  // Degenerate / behind-camera / NaN / Inf Gaussians must be safely ignored,
  // while valid foreground Gaussians in the same list render normally.
  const float kNan = std::numeric_limits<float>::quiet_NaN();
  const float kInf = std::numeric_limits<float>::infinity();
  const Gaussian3DGpu validBlue{
    {0.0f, 0.0f, 8.0f, 0.96f},
    {0.8f, 0.8f, 0.4f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f},
    {0.05f, 0.15f, 0.98f, 1.0f}};
  gw.setGaussiansForTest({
    {{0.0f, 0.0f, -5.0f, 0.98f}, {1.2f, 1.2f, 0.5f, 0.0f},
     {0.0f, 0.0f, 0.0f, 0.0f}, {0.98f, 0.05f, 0.05f, 1.0f}},
    {{0.0f, 0.0f, 0.20f, 0.98f}, {1.2f, 1.2f, 0.5f, 0.0f},
     {0.0f, 0.0f, 0.0f, 0.0f}, {0.98f, 0.05f, 0.05f, 1.0f}},
    {{0.0f, 0.0f, 6.0f, 0.98f}, {0.0f, 1.0f, 1.0f, 0.0f},
     {0.0f, 0.0f, 0.0f, 0.0f}, {0.98f, 0.05f, 0.05f, 1.0f}},
    {{kNan, 0.0f, 6.0f, 0.98f}, {1.0f, 1.0f, 1.0f, 0.0f},
     {0.0f, 0.0f, kInf, 0.0f}, {0.98f, 0.05f, 0.05f, 1.0f}},
    validBlue,
  });
  cmd = s_ctx.beginSingleTimeCommands();
  gw.dispatchRealitySplat(cmd, Mat4::identity(), intr);
  s_ctx.endSingleTimeCommands(cmd);
  gw.downloadRealityViewRGBA(rgba);
  const RgbaPixel ctr = samplePixel(
    rgba, GaussianWorldModel::kRealityW / 2, GaussianWorldModel::kRealityH / 2);
  EXPECT_GT(ctr.b, 200);
  EXPECT_LT(ctr.r, 50);

  gw.destroy();
}

TEST(V15RealityWindowHudTest, NeutralAlertTagsGf3dAndNanSanitization) {
  static_assert(std::is_trivially_copyable_v<v15_hud::V15HudTelemetryLines>,
                "V15 HUD telemetry struct must stay trivially copyable");
  EgoState ego{};
  ego.gear = 0;
  PlannerCommand cmd{};
  cmd.gear = 0;
  cmd.behaviorState = "CRUISE";
  cmd.aebActive = true;

  PerceptionFrameOutput perc{};
  perc.v15.active = true;
  perc.gaussianFormer3D.active = true;
  perc.gaussianFormer3D.gaussianCount = 512u;
  perc.v15Gaussians.resize(1024u);

  const auto neutralAeb = v15_hud::formatV15HudTelemetry(ego, perc, cmd);
  const std::string hAeb = neutralAeb.header.data();
  EXPECT_NE(hAeb.find("GEAR:N"), std::string::npos);
  EXPECT_NE(hAeb.find("GAUSS:1024"), std::string::npos);
  EXPECT_NE(hAeb.find("GF3D:512"), std::string::npos);
  EXPECT_NE(hAeb.find("| AEB"), std::string::npos);
  EXPECT_EQ(hAeb.find("| MRM"), std::string::npos);

  cmd.gear = -1;
  cmd.behaviorState = "SENSOR_SOILED_MRM";
  const auto revMrm = v15_hud::formatV15HudTelemetry(ego, perc, cmd);
  const std::string hMrm = revMrm.header.data();
  EXPECT_NE(hMrm.find("GEAR:R"), std::string::npos);
  EXPECT_NE(hMrm.find("| MRM"), std::string::npos);

  const float kNan = std::numeric_limits<float>::quiet_NaN();
  const float kInf = std::numeric_limits<float>::infinity();
  ego.currentLateralOffset = kNan;
  perc.v15.calibPitchDeg = kInf;
  perc.v15.calibRollDeg = -kInf;
  perc.v15.preferredLaneOffsetM = kNan;
  perc.v15.rearClearanceM = kNan;
  perc.v15.sirenConfidence = kNan;
  perc.v15.sirenDopplerHz = kInf;
  perc.v15.latentNovelty = kNan;
  perc.v15.surpriseSigmaDelta = kNan;
  perc.v15.staticSkipRatio = kNan;
  perc.v15.tokenSparsityRatio = kInf;
  perc.v15.kvCacheReuseRatio = kNan;
  perc.v15.effectiveSpeedupX = -kInf;

  const auto san = v15_hud::formatV15HudTelemetry(ego, perc, cmd);
  for (const char* line : {san.header.data(), san.calib.data(),
                           san.audio.data(), san.compute.data()}) {
    const std::string s(line);
    EXPECT_EQ(s.find("nan"), std::string::npos) << s;
    EXPECT_EQ(s.find("NaN"), std::string::npos) << s;
    EXPECT_EQ(s.find("inf"), std::string::npos) << s;
    EXPECT_EQ(s.find("Inf"), std::string::npos) << s;
  }
  EXPECT_NE(std::string(san.compute.data()).find("SPD:1.00x"),
            std::string::npos);
}

}  // namespace
}  // namespace cybercab
