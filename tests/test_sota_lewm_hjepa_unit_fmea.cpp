#include "app/V15HudOverlay.hpp"
#include "config/SimConfig.hpp"
#include "core/TraceLogger.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/sota/HJepaHierarchy.hpp"
#include "perception/sota/LeWmLatentModel.hpp"
#include "perception/sota/LeWmTypes.hpp"
#include "planning/sota/WorldModelGuidance.hpp"

#include <gtest/gtest.h>
#include <json/json.h>
#include <malloc.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace cybercab::sota {
namespace {

long getHeapUsedBytes() {
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
  return static_cast<long>(::mallinfo2().uordblks);
#else
  return static_cast<long>(::mallinfo().uordblks);
#endif
}

TEST(LeWmHJepaUnitTest, ConfigValidationAndYamlLoading) {
  const SimConfig yamlCfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  EXPECT_EQ(yamlCfg.worldModelMode, "legacy");
  EXPECT_FALSE(yamlCfg.isLeWorldModel());
  EXPECT_FALSE(yamlCfg.isHJepa());
  EXPECT_FALSE(yamlCfg.usesSotaWorldModel());
  EXPECT_TRUE(validateWorldModelSotaParams(yamlCfg.wmSota));
  EXPECT_TRUE(SimConfig::isValidWorldModelMode("legacy"));
  EXPECT_TRUE(SimConfig::isValidWorldModelMode("lewm"));
  EXPECT_TRUE(SimConfig::isValidWorldModelMode("hjepa"));
  EXPECT_FALSE(SimConfig::isValidWorldModelMode("invalid_wm"));

  WorldModelSotaParams bad = yamlCfg.wmSota;
  bad.latentDim = 2u;
  EXPECT_FALSE(validateWorldModelSotaParams(bad));
  bad = yamlCfg.wmSota;
  bad.cemElites = bad.cemCandidates + 1u;
  EXPECT_FALSE(validateWorldModelSotaParams(bad));
  bad = yamlCfg.wmSota;
  bad.stepDtSec = -0.1f;
  EXPECT_FALSE(validateWorldModelSotaParams(bad));
  bad = yamlCfg.wmSota;
  bad.energyCostWeight = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(validateWorldModelSotaParams(bad));

  SimConfig badCfg = yamlCfg;
  badCfg.wmSota.latentDim = 0u;
  LeWmLatentModel model;
  model.init(badCfg);
  EXPECT_NE(model.initFaultMask() & kWmFaultInvalidConfig, 0u);
}

TEST(LeWmHJepaUnitTest, WeightBinaryIoRoundTripAndCorruptionFmea) {
  const auto p = std::filesystem::temp_directory_path() / "cybercab_lewm_w.bin";
  std::filesystem::remove(p);
  const LeWmWeights defW = LeWmWeights::defaultAnalytical();
  EXPECT_TRUE(defW.isFiniteValid());
  ASSERT_TRUE(defW.saveBinary(p.string()));

  LeWmWeights loaded{};
  EXPECT_TRUE(LeWmWeights::loadBinary(p.string(), loaded));
  EXPECT_EQ(loaded.computeChecksum(), defW.checksum);
  EXPECT_FLOAT_EQ(loaded.macroW[4][9], defW.macroW[4][9]);

  // FMEA: corrupted checksum byte
  {
    std::ifstream ifs(p, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
    ASSERT_GT(bytes.size(), 32u);
    bytes[20] ^= static_cast<char>(0x5Au);
    std::ofstream ofs(p, std::ios::binary | std::ios::trunc);
    ofs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  LeWmWeights guard = defW;
  EXPECT_FALSE(LeWmWeights::loadBinary(p.string(), guard));

  // FMEA: corrupted magic header
  {
    std::vector<char> junk(32u, 0x11);
    std::ofstream ofs(p, std::ios::binary | std::ios::trunc);
    ofs.write(junk.data(), static_cast<std::streamsize>(junk.size()));
  }
  EXPECT_FALSE(LeWmWeights::loadBinary(p.string(), guard));

  // FMEA: NaN weight rejection + missing file fallback in init()
  LeWmWeights nanW = defW;
  nanW.dynA[0][0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(nanW.isFiniteValid());
  EXPECT_FALSE(nanW.saveBinary(p.string()));
  std::filesystem::remove(p);

  SimConfig cfg = SimConfig::defaultConfig();
  cfg.worldModelMode = "lewm";
  cfg.wmSota.weightsPath = p.string();
  LeWmLatentModel model;
  model.init(cfg);
  EXPECT_FALSE(model.weightsLoadedFromFile());
  EXPECT_NE(model.initFaultMask() & kWmFaultWeightLoad, 0u);
}

TEST(LeWmHJepaUnitTest, SigRegIsotropyMetricsAndCollapseFmea) {
  SimConfig cfg = SimConfig::defaultConfig();
  cfg.worldModelMode = "lewm";
  LeWmLatentModel model;
  model.init(cfg);

  SigRegMetrics variedM{};
  for (int step = 0; step < 20; ++step) {
    LatentVec16 z{};
    for (size_t i = 0; i < kLeWmLatentDim; ++i) {
      z[i] = std::sin(0.55f * static_cast<float>(step + static_cast<int>(i) * 3));
    }
    variedM = model.updateSigReg(z, 1.0f);
  }
  EXPECT_FALSE(variedM.collapsed);
  EXPECT_GT(variedM.meanSqNorm, 0.1f);
  EXPECT_TRUE(std::isfinite(variedM.sigregLoss));

  model.reset();
  SigRegMetrics colM{};
  const LatentVec16 zConst{0.4f, -0.4f, 0.0f, 0.0f};
  for (int i = 0; i < 12; ++i) colM = model.updateSigReg(zConst, 1.0f);
  EXPECT_TRUE(colM.collapsed);

  // Varying ego speed while forced collapsed projW=0 must latch kWmFaultLatentCollapse.
  const auto zeroPath =
    std::filesystem::temp_directory_path() / "cybercab_lewm_zero.bin";
  LeWmWeights zeroW = LeWmWeights::defaultAnalytical();
  zeroW.projW = {};
  zeroW.sigregMean = {};
  ASSERT_TRUE(zeroW.saveBinary(zeroPath.string()));
  cfg.wmSota.weightsPath = zeroPath.string();
  model.init(cfg);
  ASSERT_TRUE(model.weightsLoadedFromFile());

  WorldEngine w(cfg);
  PerceptionFrameOutput perc{};
  WorldModelSotaSummary sum{};
  for (int k = 0; k < 10; ++k) {
    w.mutableEgo().speed = 2.0f + 0.8f * static_cast<float>(k);
    model.stepFrame(cfg, w, perc, 1.0f / 36.0f, sum);
  }
  EXPECT_NE(sum.fmeaFaultMask & kWmFaultLatentCollapse, 0u);
  EXPECT_TRUE(sum.fmeaFallbackTriggered);
  std::filesystem::remove(zeroPath);
}

TEST(LeWmHJepaUnitTest, ActionConditionedRolloutDivergenceAndNanInputFmea) {
  SimConfig cfg = SimConfig::defaultConfig();
  cfg.worldModelMode = "lewm";
  LeWmLatentModel model;
  model.init(cfg);

  WorldEngine w(cfg);
  PerceptionFrameOutput perc{};
  Track3D lead{};
  lead.trackId = 7;
  lead.semantic = SemanticClass::Vehicle;
  lead.worldPos = {w.ego().pos.x + 12.0f, 0.7f, w.ego().pos.z};
  lead.worldVel = {2.0f, 0.0f, 0.0f};
  lead.halfExtents = {0.96f, 0.72f, 2.18f};
  perc.tracks.push_back(lead);

  LatentVec16 z{};
  uint32_t mask = 0u;
  ASSERT_TRUE(model.encodeScene(cfg, w, perc, z, mask));
  EXPECT_GT(z[4], 0.35f);
  EXPECT_GT(z[5], 0.25f);

  float eBrake = 0.0f, eAccel = 0.0f;
  ASSERT_TRUE(model.rolloutEnergy(z, {-4.0f, 0.0f}, 6u, 0.25f, eBrake));
  ASSERT_TRUE(model.rolloutEnergy(z, {2.5f, 0.0f}, 6u, 0.25f, eAccel));
  EXPECT_LT(eBrake, eAccel);

  // Predict divergence / invalid action guards
  LatentVec16 nxt{};
  EXPECT_FALSE(model.predictNext(z, {std::numeric_limits<float>::infinity(), 0.0f}, 0.25f, nxt));
  EXPECT_FALSE(model.predictNext(z, {0.0f, 0.0f}, 0.0f, nxt));
  LatentVec16 badZ = z;
  badZ[2] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(model.predictNext(badZ, {0.0f, 0.0f}, 0.25f, nxt));

  // NaN input injection into stepFrame -> FMEA safe fallback
  w.mutableEgo().speed = std::numeric_limits<float>::quiet_NaN();
  WorldModelSotaSummary sum{};
  model.stepFrame(cfg, w, perc, 1.0f / 36.0f, sum);
  EXPECT_NE(sum.fmeaFaultMask & kWmFaultNanInput, 0u);
  EXPECT_TRUE(sum.fmeaFallbackTriggered);
  EXPECT_FLOAT_EQ(guidedSpeedScale(cfg, perc), 1.0f);
}

TEST(LeWmHJepaUnitTest, HJepaAllFiveMacroSubgoalsCemDeterminismAndZeroAlloc) {
  SimConfig cfg = SimConfig::defaultConfig();
  cfg.worldModelMode = "hjepa";
  LeWmLatentModel model;
  model.init(cfg);
  EXPECT_STREQ(macroSubgoalName(0), "KEEP_LANE");
  EXPECT_STREQ(macroSubgoalName(4), "STOP_HOLD");
  EXPECT_STREQ(macroSubgoalName(99), "UNKNOWN");

  // 1. Clear right lane -> KeepLane
  LatentVec16 zClear{0.9f, -1.0f};
  auto mClear = HJepaHierarchy::planMacroSubgoal(cfg, EgoState{}, zClear, model, false, false);
  EXPECT_EQ(mClear.code, MacroSubgoal::KeepLane);

  // 2. Slow right-lane block + clear left lane -> OvertakeLeft
  LatentVec16 zBlock{};
  zBlock[0] = 0.85f; zBlock[1] = -1.0f; zBlock[4] = 0.72f; zBlock[5] = 0.55f; zBlock[6] = 0.65f; zBlock[11] = 1.0f;
  auto mOv = HJepaHierarchy::planMacroSubgoal(cfg, EgoState{}, zBlock, model, false, false);
  EXPECT_EQ(mOv.code, MacroSubgoal::OvertakeLeft);

  // 3. Slow right-lane block + ONCOMING left-lane vehicle -> refuses OvertakeLeft!
  LatentVec16 zOncom = zBlock;
  zOncom[9] = 0.95f;
  auto mSafe = HJepaHierarchy::planMacroSubgoal(cfg, EgoState{}, zOncom, model, false, false);
  EXPECT_NE(mSafe.code, MacroSubgoal::OvertakeLeft);

  // 4. In left lane, right lane clear -> ReturnRight
  LatentVec16 zLeft{0.9f, 1.0f};
  auto mRet = HJepaHierarchy::planMacroSubgoal(cfg, EgoState{}, zLeft, model, false, false);
  EXPECT_EQ(mRet.code, MacroSubgoal::ReturnRight);

  // 5. Siren or high epistemic glare -> YieldSlow; Red light / imminent VRU -> StopHold
  LatentVec16 zGlare{0.75f, -1.0f}; zGlare[14] = 0.95f;
  auto mYield = HJepaHierarchy::planMacroSubgoal(cfg, EgoState{}, zGlare, model, false, true);
  EXPECT_EQ(mYield.code, MacroSubgoal::YieldSlow);
  LatentVec16 zStop{0.65f, -1.0f}; zStop[12] = 0.90f; zStop[13] = 0.95f;
  auto mStop = HJepaHierarchy::planMacroSubgoal(cfg, EgoState{}, zStop, model, true, false);
  EXPECT_EQ(mStop.code, MacroSubgoal::StopHold);

  // CEM monotonicity across iterations
  uint32_t fault = 0u;
  const CemPlanResult cem = HJepaHierarchy::optimizeActionCem(
    cfg, EgoState{}, zBlock, mOv, model, -1.95f, 1.0f, fault);
  EXPECT_TRUE(cem.valid);
  EXPECT_EQ(fault, 0u);
  EXPECT_LE(cem.finalBestCost, cem.iter0BestCost + 1e-4f);

  // Determinism + Zero hot-path heap allocation check
  WorldEngine w(cfg);
  PerceptionFrameOutput perc{};
  HJepaHierarchy h1, h2;
  h1.reset();
  h2.reset();
  WorldModelSotaSummary s1{}, s2{};
  for (int i = 0; i < 4; ++i) {
    model.stepFrame(cfg, w, perc, 1.0f / 36.0f, s1);
    h1.runHierarchy(cfg, w, perc, model, model.lastLatent(), s1);
  }
  const long h0 = getHeapUsedBytes();
  for (int i = 0; i < 60; ++i) {
    model.stepFrame(cfg, w, perc, 1.0f / 36.0f, s1);
    h1.runHierarchy(cfg, w, perc, model, model.lastLatent(), s1);
  }
  EXPECT_EQ(getHeapUsedBytes() - h0, 0L);

  model.init(cfg);
  for (int i = 0; i < 64; ++i) {
    model.stepFrame(cfg, w, perc, 1.0f / 36.0f, s2);
    h2.runHierarchy(cfg, w, perc, model, model.lastLatent(), s2);
  }
  EXPECT_FLOAT_EQ(s1.bestRolloutEnergy, s2.bestRolloutEnergy);
  EXPECT_FLOAT_EQ(s1.recommendedLatM, s2.recommendedLatM);
  EXPECT_EQ(s1.macroSubgoalCode, s2.macroSubgoalCode);
}

TEST(LeWmHJepaUnitTest, TraceLoggerSchemaOmissionAndHudTags) {
  static_assert(std::is_trivially_copyable_v<v15_hud::V15HudTelemetryLines>,
                "V15HudTelemetryLines must remain trivially copyable");
  SimConfig cfg = SimConfig::defaultConfig();
  EgoState ego{};
  PlannerCommand cmd{};
  PerceptionFrameOutput perc{};

  const std::string jLegacy =
    TraceLogger::formatFrameJson(0, 0.0f, cfg, ego, cmd, perc, {});
  Json::CharReaderBuilder rb;
  Json::Value rootLeg;
  std::string errs;
  std::istringstream issLeg(jLegacy);
  ASSERT_TRUE(Json::parseFromStream(rb, issLeg, &rootLeg, &errs)) << errs;
  EXPECT_FALSE(rootLeg.isMember("world_model_sota"));
  static_assert(std::is_trivially_copyable_v<v15_hud::WorldModelHudLines>);
  const auto hudLeg = v15_hud::formatV15HudTelemetry(ego, perc, cmd);
  EXPECT_EQ(hudLeg.worldModel[0], '\0');
  const auto wmLeg = v15_hud::formatWorldModelHudLines(perc);
  EXPECT_NE(std::string(wmLeg.line1.data()).find("REP:BEV2D | WM:LEGACY"), std::string::npos);

  cfg.worldModelMode = "hjepa";
  perc.v15.active = true;
  perc.worldModelSota.active = true;
  perc.worldModelSota.hierarchical = true;
  perc.worldModelSota.macroSubgoalCode = static_cast<int>(MacroSubgoal::KeepLane);
  const auto hudHj = v15_hud::formatV15HudTelemetry(ego, perc, cmd);
  EXPECT_NE(std::string(hudHj.header.data()).find("| HJEPA"), std::string::npos);
  EXPECT_NE(std::string(hudHj.worldModel.data()).find("WM:HJEPA KEEP_LANE"), std::string::npos);
  const auto wmHj = v15_hud::formatWorldModelHudLines(perc);
  EXPECT_NE(std::string(wmHj.line1.data()).find("WM:HJEPA (KEEP_LANE)"), std::string::npos);
  EXPECT_NE(std::string(wmHj.line2.data()).find("SURP:"), std::string::npos);

  perc.worldModelSota.hierarchical = false;
  perc.gaussianFormer3D.active = true;
  perc.gaussianFormer3D.gaussianCount = 512;
  const auto hudLe = v15_hud::formatV15HudTelemetry(ego, perc, cmd);
  EXPECT_NE(std::string(hudLe.header.data()).find("| LEWM"), std::string::npos);
  EXPECT_NE(std::string(hudLe.worldModel.data()).find("WM:LEWM LAT:"), std::string::npos);
  const auto wmGf = v15_hud::formatWorldModelHudLines(perc);
  EXPECT_NE(std::string(wmGf.line1.data()).find("REP:GF3D(512) | WM:LEWM"), std::string::npos);

  perc.worldModelSota.fmeaFallbackTriggered = true;
  const auto hudFb = v15_hud::formatV15HudTelemetry(ego, perc, cmd);
  EXPECT_NE(std::string(hudFb.header.data()).find("| WM:FB"), std::string::npos);
  EXPECT_NE(std::string(hudFb.worldModel.data()).find("WM:LEWM[FB]"), std::string::npos);
  const auto wmFb = v15_hud::formatWorldModelHudLines(perc);
  EXPECT_NE(std::string(wmFb.line1.data()).find("[FB]"), std::string::npos);

  const std::string jWm =
    TraceLogger::formatFrameJson(1, 0.1f, cfg, ego, cmd, perc, {});
  Json::Value rootWm;
  std::istringstream issWm(jWm);
  ASSERT_TRUE(Json::parseFromStream(rb, issWm, &rootWm, &errs)) << errs;
  ASSERT_TRUE(rootWm.isMember("world_model_sota"));
  EXPECT_EQ(rootWm["world_model_sota"]["mode"].asString(), "hjepa");
  EXPECT_TRUE(rootWm["world_model_sota"]["fmea_fallback"].asBool());

  perc.worldModelSota.predSurprise = std::numeric_limits<float>::quiet_NaN();
  perc.worldModelSota.sigregLoss = std::numeric_limits<float>::infinity();
  const auto wmNaN = v15_hud::formatWorldModelHudLines(perc);
  EXPECT_EQ(std::string(wmNaN.line2.data()).find("nan"), std::string::npos);
  EXPECT_EQ(std::string(wmNaN.line2.data()).find("inf"), std::string::npos);
}

}  // namespace
}  // namespace cybercab::sota
