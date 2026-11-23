#include "app/V15HudOverlay.hpp"
#include "config/SimConfig.hpp"
#include "core/TraceLogger.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "world/WorldEngine.hpp"

#include <gtest/gtest.h>
#include <json/json.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace cybercab {
namespace {
constexpr int kReverseGear = -1, kDriveGear = 1;
constexpr int kTraceStepV14 = 0, kTraceStepV15 = 1;
constexpr float kTraceTimeV14Sec = 0.0f, kTraceTimeV15Sec = 0.1f;
constexpr float kTestLatOffsetM = -1.92f, kTestPrefLaneM = -1.95f;
constexpr float kTestPitchDeg = 0.35f, kTestRollDeg = -0.18f;
constexpr float kTestSirenConf = 0.88f, kTestSirenBearRad = -0.45f;
constexpr float kTestSirenDoppHz = 42.0f, kTestRearClearM = 9.4f;
constexpr float kTestNovelty = 0.27f, kTestSigmaDelta = 0.14f;
constexpr float kTestSkipRatio = 0.65f, kTestSparsityRatio = 0.72f;
constexpr float kTestKvRatio = 0.58f, kTestSpeedupX = 2.40f;
constexpr float kFrameDtSec = 1.0f / 36.0f, kJsonEps = 1e-4f;
constexpr uint8_t kMinNonZeroRgbVal = 4u;

Json::Value readFirstJsonLine(const std::filesystem::path& path) {
  std::ifstream ifs(path);
  EXPECT_TRUE(ifs.good());
  std::string line;
  EXPECT_TRUE(static_cast<bool>(std::getline(ifs, line)));
  Json::CharReaderBuilder rb;
  Json::Value root;
  std::string errs;
  std::istringstream iss(line);
  EXPECT_TRUE(Json::parseFromStream(rb, iss, &root, &errs)) << errs;
  return root;
}

TEST(V15TelemetryAndHudTest, TraceLoggerV15SchemaAndV14Equivalence) {
  const SimConfig cfg = SimConfig::defaultConfig();
  const auto tmpDir = std::filesystem::temp_directory_path();
  const auto pV14 = tmpDir / "v15_trace_equiv_v14.jsonl";
  const auto pV15 = tmpDir / "v15_trace_equiv_v15.jsonl";
  EgoState ego{};
  ego.gear = kReverseGear;
  ego.currentLateralOffset = kTestLatOffsetM;
  PlannerCommand cmd{};
  PerceptionFrameOutput perc{};
  {
    TraceLogger log14;
    ASSERT_TRUE(log14.open(pV14.string()));
    log14.logFrame(kTraceStepV14, kTraceTimeV14Sec, cfg, ego, cmd, perc, {});
  }
  perc.v15 = {true, true, kTestSirenConf, kTestSirenBearRad, kTestSirenDoppHz,
              kTestNovelty, kTestSigmaDelta, kTestSkipRatio,
              kTestSparsityRatio, kTestKvRatio, kTestSpeedupX, kTestPrefLaneM,
              kTestRearClearM, kTestPitchDeg, kTestRollDeg};
  perc.v15Gaussians.resize(2u);
  perc.v15RealityRgba.resize(v15_hud::kRealityExpectedBytes, 0u);
  {
    TraceLogger log15;
    ASSERT_TRUE(log15.open(pV15.string()));
    log15.logFrame(kTraceStepV15, kTraceTimeV15Sec, cfg, ego, cmd, perc, {});
  }
  const Json::Value j14 = readFirstJsonLine(pV14);
  const Json::Value j15 = readFirstJsonLine(pV15);
  std::filesystem::remove(pV14);
  std::filesystem::remove(pV15);

  EXPECT_FALSE(j14.isMember("v15"));
  ASSERT_TRUE(j15.isMember("v15"));
  const Json::Value& v15 = j15["v15"];
  EXPECT_TRUE(v15["active"].asBool());
  EXPECT_EQ(v15["gear"].asInt(), kReverseGear);
  EXPECT_NEAR(v15["lat_offset_m"].asFloat(), kTestLatOffsetM, kJsonEps);
  EXPECT_NEAR(
    v15["preferred_lane_offset_m"].asFloat(), kTestPrefLaneM, kJsonEps);
  EXPECT_NEAR(v15["calib_pitch_deg"].asFloat(), kTestPitchDeg, kJsonEps);
  EXPECT_NEAR(v15["calib_roll_deg"].asFloat(), kTestRollDeg, kJsonEps);
  EXPECT_TRUE(v15["siren_detected"].asBool());
  EXPECT_NEAR(v15["siren_confidence"].asFloat(), kTestSirenConf, kJsonEps);
  EXPECT_NEAR(v15["siren_bearing_rad"].asFloat(), kTestSirenBearRad, kJsonEps);
  EXPECT_NEAR(v15["siren_doppler_hz"].asFloat(), kTestSirenDoppHz, kJsonEps);
  EXPECT_NEAR(v15["rear_clearance_m"].asFloat(), kTestRearClearM, kJsonEps);
  EXPECT_NEAR(v15["latent_novelty"].asFloat(), kTestNovelty, kJsonEps);
  EXPECT_NEAR(
    v15["surprise_sigma_delta"].asFloat(), kTestSigmaDelta, kJsonEps);
  EXPECT_NEAR(v15["static_skip_ratio"].asFloat(), kTestSkipRatio, kJsonEps);
  EXPECT_NEAR(
    v15["token_sparsity_ratio"].asFloat(), kTestSparsityRatio, kJsonEps);
  EXPECT_NEAR(v15["kv_cache_reuse_ratio"].asFloat(), kTestKvRatio, kJsonEps);
  EXPECT_NEAR(v15["effective_speedup_x"].asFloat(), kTestSpeedupX, kJsonEps);
  EXPECT_EQ(v15["gaussian_count"].asInt(), 2);
  EXPECT_EQ(v15["reality_view_bytes"].asInt(),
            static_cast<int>(v15_hud::kRealityExpectedBytes));
}

TEST(V15TelemetryAndHudTest, V15HudOverlayFormatsDriveReverseAndSiren) {
  static_assert(std::is_trivially_copyable_v<v15_hud::V15HudTelemetryLines>,
                "V15 HUD formatter struct must remain zero-allocation POD");
  EgoState ego{};
  ego.gear = kDriveGear;
  ego.currentLateralOffset = kTestLatOffsetM;
  PlannerCommand cmd{};
  cmd.gear = kDriveGear;
  PerceptionFrameOutput perc{};
  perc.v15 = {true, false, 0.0f, 0.0f, 0.0f, kTestNovelty, kTestSigmaDelta,
              kTestSkipRatio, kTestSparsityRatio, kTestKvRatio, kTestSpeedupX,
              kTestPrefLaneM, kTestRearClearM, kTestPitchDeg, kTestRollDeg};
  perc.v15Gaussians.resize(3u);
  const auto drive = v15_hud::formatV15HudTelemetry(ego, perc, cmd);
  const std::string hD = drive.header.data();
  const std::string cD = drive.calib.data();
  const std::string aD = drive.audio.data();
  const std::string sD = drive.compute.data();
  EXPECT_NE(hD.find("GEAR:D"), std::string::npos);
  EXPECT_NE(hD.find("GAUSS:3"), std::string::npos);
  EXPECT_NE(cD.find("CALIB P:"), std::string::npos);
  EXPECT_NE(aD.find("SIREN:CLEAR"), std::string::npos);
  EXPECT_NE(sD.find("SPD:2.40x"), std::string::npos);

  ego.gear = kReverseGear;
  cmd.gear = kReverseGear;
  perc.v15.sirenDetected = true;
  perc.v15.sirenConfidence = kTestSirenConf;
  const auto rev = v15_hud::formatV15HudTelemetry(ego, perc, cmd);
  EXPECT_NE(std::string(rev.header.data()).find("GEAR:R"), std::string::npos);
  EXPECT_NE(
    std::string(rev.audio.data()).find("SIREN:YIELD"), std::string::npos);
}

TEST(V15TelemetryAndHudTest, V15PipelineExportsRealityViewAndGaussians) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = "v15";
  WorldEngine world(cfg);
  VulkanContext ctx;
  ctx.init(false, nullptr);
  CameraRigRenderer rig;
  rig.init(ctx, cfg, world);
  PerceptionPipeline pipe(cfg);
  pipe.init(ctx, rig);
  PerceptionFrameOutput out{};
  for (int i = 0; i < 2; ++i) {
    rig.renderAllCameras(world, cfg, static_cast<float>(i) * kFrameDtSec);
    pipe.runFrame(ctx, rig, world, kFrameDtSec, true, out);
  }
  EXPECT_TRUE(out.v15.active);
  EXPECT_FALSE(out.v15Gaussians.empty());
  EXPECT_EQ(out.v15RealityRgba.size(), v15_hud::kRealityExpectedBytes);
  size_t nonZeroRgb = 0u;
  const auto& rgba = out.v15RealityRgba;
  for (size_t i = 0; i + v15_hud::kRealityRgbaChannels <= rgba.size();
       i += v15_hud::kRealityRgbaChannels) {
    if (rgba[i] > kMinNonZeroRgbVal || rgba[i + 1u] > kMinNonZeroRgbVal ||
        rgba[i + 2u] > kMinNonZeroRgbVal) {
      ++nonZeroRgb;
    }
  }
  EXPECT_GT(nonZeroRgb, 0u);
  pipe.destroy();
  rig.destroy();
  ctx.cleanup();
}

}  // namespace
}  // namespace cybercab
