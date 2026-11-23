#include "config/SimConfig.hpp"
#include "core/VulkanPipeline.hpp"
#include "planning/v15/AutoregressivePolicy.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cybercab {
namespace {
constexpr float kOffMap = -350.0f, kLane0 = -1.95f, kAdjSep = 3.90f;
constexpr float kCfgC = 2.0f, kCfgMin = 1.75f, kCfgMax = 2.20f, kEps = 1e-4f;
constexpr float kTightLat = -1.2f, kWideLat = 2.8f, kTruckFwd = 10.0f;
constexpr float kCurveU = 0.34f, kCurveV = 15.0f, kMinCurvV = 6.0f;
constexpr float kLoLatA = 2.0f, kHiLatA = 8.0f, kAddRss = 1.5f, kGaussR = 0.65f;
constexpr float kLoSig = 0.15f, kHiSig = 0.75f, kModOcc = 0.50f, kModFwd = 4.0f;
constexpr float kStopLo = 0.40f, kStopHi = 0.65f, kPedFwd = 3.0f, kAppV = 6.0f;
constexpr float kSirenLo = 0.20f, kSirenMid = 0.45f, kSirenHi = 0.80f;
constexpr float kLeadEx = 5.0f, kLeadVx = 4.5f, kStopBarOff = -6.0f;
constexpr float kSignX = 22.0f, kSignZ = -45.95f, kBlkFwd = 5.0f;
constexpr float kOvertakeFwd = 14.0f, kSoilP = 0.85f, kSoilW = 0.20f;
constexpr float kRearOk = 12.0f, kRearNo = 3.0f;
const Vec3 kTruckHe{1.15f, 1.55f, 4.40f}, kCarHe{0.96f, 0.72f, 2.18f};
const Vec3 kPedHe{0.4f, 0.9f, 0.4f};
const Vec4 kDbgCol{0.2f, 0.9f, 0.4f, 0.9f};

SimConfig baseV15Cfg() {
  SimConfig c =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  c.pipelineVersion = "v15";
  return c;
}
PlannerCommand runPlan(const SimConfig& c, WorldEngine& w,
                       const PerceptionFrameOutput& p) {
  PlannerCommand cmd{};
  v15::AutoregressivePolicy::plan(c, w, p, cmd);
  return cmd;
}
void parkAway(WorldEngine& w) {
  for (auto& v : w.mutableVehicles()) {
    v.pos = {kOffMap, 0.0f, kOffMap}; v.speed = v.vx = v.vz = 0.0f;
  }
  for (auto& p : w.mutablePedestrians()) {
    p.x = p.ax = p.bx = p.targetX = kOffMap;
    p.z = p.az = p.bz = p.targetZ = kOffMap;
  }
}
Track3D mkVeh(int id, float x, float z, float vx, float yaw, Vec3 he = kCarHe) {
  Track3D t{};
  t.trackId = id; t.semantic = SemanticClass::Vehicle; t.score = 0.95f;
  t.hits = 3; t.yaw = yaw; t.halfExtents = he;
  t.worldPos = {x, he.y, z}; t.worldVel = {vx, 0.0f, 0.0f};
  return t;
}

TEST(V15ParamsAndActionsTest, V15PolicyAndSimParamsProduceExactEffects) {
  SimConfig cfg = baseV15Cfg();
  V15PolicyConfig pol = cfg.v15Policy;
  pol.laneCenterAbsM = kCfgC; pol.minLaneAbsM = kCfgMin;
  pol.maxLaneAbsM = kCfgMax;
  EXPECT_FLOAT_EQ(
    v15::AutoregressivePolicy::clampToMiddleOfLane(kTightLat, pol), -kCfgMin);
  EXPECT_FLOAT_EQ(
    v15::AutoregressivePolicy::clampToMiddleOfLane(kWideLat, pol), kCfgMax);
  WorldEngine world(cfg); parkAway(world);
  const EgoState& ego = world.ego();
  const auto trk = mkVeh(1, ego.pos.x + kTruckFwd, ego.pos.z - kAdjSep,
                         0.0f, ego.yaw, kTruckHe);
  pol = cfg.v15Policy; pol.maxTruckShyM = 0.0f;
  EXPECT_NEAR(v15::AutoregressivePolicy::computeTruckShyOffset(
                kLane0, ego, {trk}, pol), kLane0, kEps);
  pol.maxTruckShyM = cfg.v15Policy.maxTruckShyM;
  EXPECT_LT(v15::AutoregressivePolicy::computeTruckShyOffset(
              kLane0, ego, {trk}, pol), kLane0);
  const Vec3 cPt = world.centerline().getPointAt(kCurveU);
  const Vec3 cTan = world.centerline().getTangentAt(kCurveU);
  world.mutableEgo().pos = cPt + Vec3{cTan.z, 0.0f, -cTan.x} * kLane0;
  world.mutableEgo().yaw = std::atan2(cTan.x, cTan.z);
  cfg.targetCruiseSpeedMs = kCurveV; cfg.v15Policy.minCurveSpeedMs = kMinCurvV;
  PerceptionFrameOutput perc{}; perc.v15.active = true;
  cfg.v15Policy.maxLatAccelMs2 = kLoLatA;
  const PlannerCommand lo = runPlan(cfg, world, perc);
  cfg.v15Policy.maxLatAccelMs2 = kHiLatA;
  const PlannerCommand hi = runPlan(cfg, world, perc);
  EXPECT_GE(lo.targetSpeed, kMinCurvV - kEps);
  EXPECT_LT(lo.targetSpeed, hi.targetSpeed);
  WorldEngine wBase(cfg); parkAway(wBase);
  perc.forwardEpistemicSigma = kLoSig;
  const PlannerCommand rLo = runPlan(cfg, wBase, perc);
  perc.forwardEpistemicSigma = kHiSig; cfg.rssDBufferM += kAddRss;
  const PlannerCommand rHi = runPlan(cfg, wBase, perc);
  EXPECT_GT(rHi.rssSafeDist, rLo.rssSafeDist + kAddRss);
}

TEST(V15ParamsAndActionsTest, SafetyParamsControlSirenAebAutoshiftGaussian) {
  SimConfig cfg = baseV15Cfg();
  WorldEngine world(cfg); parkAway(world);
  PerceptionFrameOutput perc{}; perc.v15.active = true;
  perc.v15.sirenDetected = true; perc.v15.sirenConfidence = kSirenLo;
  EXPECT_EQ(runPlan(cfg, world, perc).behaviorState, "CRUISING");
  perc.v15.sirenConfidence = kSirenMid;
  const PlannerCommand sMid = runPlan(cfg, world, perc);
  EXPECT_EQ(sMid.behaviorState, "YIELDING_SIREN");
  EXPECT_LE(sMid.targetSpeed, cfg.v15Policy.sirenYieldSpeedMs);
  perc.v15.sirenConfidence = kSirenHi;
  EXPECT_FLOAT_EQ(runPlan(cfg, world, perc).targetSpeed, 0.0f);

  perc.v15.sirenDetected = false;
  const EgoState& ego = world.ego();
  const float bX = ego.pos.x + ego.centerOffsetForward + ego.hz;
  const int pedI = static_cast<int>(SemanticClass::Pedestrian);
  perc.v15Gaussians = {{{bX + kModFwd, kGaussR, ego.pos.z, kModOcc},
                         {kGaussR, kGaussR, kGaussR, static_cast<float>(pedI)},
                         {0.0f, 0.0f, 0.0f, 0.0f}, kDbgCol}};
  cfg.v15Policy.gaussianSafetyStopProb = kStopHi;
  EXPECT_GT(runPlan(cfg, world, perc).targetSpeed, 0.0f);
  cfg.v15Policy.gaussianSafetyStopProb = kStopLo;
  EXPECT_FLOAT_EQ(runPlan(cfg, world, perc).targetSpeed, 0.0f);

  perc.v15Gaussians.clear(); world.mutableEgo().speed = 0.0f;
  perc.tracks = {
    mkVeh(1, ego.pos.x + kBlkFwd, ego.pos.z, 0.0f, ego.yaw),
    mkVeh(2, ego.pos.x + kBlkFwd, ego.pos.z - kAdjSep, 0.0f, ego.yaw)};
  perc.v15.rearClearanceM = kRearOk;
  const PlannerCommand rev = runPlan(cfg, world, perc);
  EXPECT_EQ(rev.gear, -1);
  EXPECT_FLOAT_EQ(rev.targetSpeed, cfg.v15Policy.autoshiftReverseSpeedMs);
  perc.v15.rearClearanceM = kRearNo;
  const PlannerCommand abt = runPlan(cfg, world, perc);
  EXPECT_EQ(abt.gear, 1); EXPECT_FLOAT_EQ(abt.targetSpeed, 0.0f);
}

TEST(V15ParamsAndActionsTest, AllDrivingActionStatesTriggerOnExactConditions) {
  SimConfig cfg = baseV15Cfg();
  WorldEngine world(cfg); parkAway(world);
  PerceptionFrameOutput perc{}; perc.v15.active = true;
  const float yaw = world.ego().yaw;
  const float bX =
    world.ego().pos.x + world.ego().centerOffsetForward + world.ego().hz;
  EXPECT_EQ(runPlan(cfg, world, perc).behaviorState, "CRUISING");
  perc.tracks = {mkVeh(1, bX + kOvertakeFwd, world.ego().pos.z, 0.0f, yaw)};
  EXPECT_EQ(runPlan(cfg, world, perc).behaviorState, "LANE_CHANGE_OVERTAKE");
  const PlannerCommand ov = runPlan(cfg, world, perc);
  perc.tracks = {
    mkVeh(2, bX + ov.rssSafeDist + kLeadEx, world.ego().pos.z, kLeadVx, yaw),
    mkVeh(3, bX + kBlkFwd, world.ego().pos.z - kAdjSep, kLeadVx, yaw)};
  EXPECT_EQ(runPlan(cfg, world, perc).behaviorState, "FOLLOWING_LEAD");
  Track3D ped{};
  ped.trackId = 4; ped.semantic = SemanticClass::Pedestrian; ped.score = 0.95f;
  ped.hits = 3; ped.halfExtents = kPedHe;
  ped.worldPos = {bX + kPedFwd, 0.9f, world.ego().pos.z};
  perc.tracks = {ped}; world.mutableEgo().speed = kAppV;
  EXPECT_EQ(runPlan(cfg, world, perc).behaviorState, "AEB_EMERGENCY_BRAKE");
  cfg.aebEnabled = false;
  EXPECT_EQ(runPlan(cfg, world, perc).behaviorState, "YIELDING_PEDESTRIAN");
  cfg.aebEnabled = true; perc.tracks.clear();
  world.mutableEgo().pos.x = WorldEngine::kStopLineX + kStopBarOff;
  world.setForcedRedSignal(true);
  EXPECT_EQ(runPlan(cfg, world, perc).behaviorState, "RED_LIGHT_HOLD");
  world.setForcedRedSignal(false);
  world.mutableEgo().pos = {kSignX, 0.0f, kSignZ};
  world.mutableEgo().yaw = -kPi * 0.5f;
  world.mutableEgo().passedStopSignCooldown = 0.0f;
  EXPECT_EQ(runPlan(cfg, world, perc).behaviorState, "STOP_SIGN_HOLD");
  for (int i = 0; i < 3; ++i) {
    perc.soilCameras[i].meanSoilProb = kSoilP;
    perc.soilCameras[i].tokenWeight = kSoilW;
  }
  EXPECT_EQ(runPlan(cfg, world, perc).behaviorState, "SENSOR_SOILED_MRM");
  cfg.driveMode = "manual"; perc.soilCameras = {};
  world.mutableEgo().speed = 0.0f;
  EXPECT_EQ(runPlan(cfg, world, perc).behaviorState, "MANUAL_DRIVE");
}

TEST(V15ParamsAndActionsTest, PolicyYamlLoadsAllSectionsAndRejectsBadBounds) {
  const auto ok =
    V15PolicyConfig::loadFromYaml(resolveAssetPath("config/v15_policy.yaml"));
  EXPECT_GT(ok.maxLatAccelMs2, 0.0f);
  EXPECT_GT(ok.gaussianOccCostGain, 0.0f);
  const std::vector<std::pair<const char*, const char*>> bads = {
    {"lane_centering:\n  min_lane_abs_m: 3.0\n", "lane"},
    {"autoregressive_rollout:\n  max_lat_accel_ms2: 0.0\n", "lat"},
    {"lane_hysteresis:\n  obb_hull_sample_scale: 0.0\n", "obb"},
    {"safety_shield:\n  min_hard_aeb_dist_m: 0.0\n", "aeb"},
    {"latent_predictor:\n  surprise_scale: 0.0\n", "sup"},
    {"gaussian_comfort:\n  speed_backoff_ratio: 1.5\n", "spd"}};
  for (const auto& [yml, tag] : bads) {
    const auto p =
      std::filesystem::temp_directory_path() / (std::string("v15_b_") + tag);
    std::ofstream(p) << yml;
    EXPECT_THROW(V15PolicyConfig::loadFromYaml(p.string()), std::runtime_error);
    std::filesystem::remove(p);
  }
}
}  // namespace
}  // namespace cybercab
