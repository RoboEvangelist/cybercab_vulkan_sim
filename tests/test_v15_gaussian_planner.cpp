#include "config/SimConfig.hpp"
#include "core/VulkanPipeline.hpp"
#include "planning/v15/AutoregressivePolicy.hpp"
#include "planning/v15/GaussianRolloutScorer.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cybercab {
namespace {
constexpr float kOffMapPosM = -350.0f, kOffMapPedM = -320.0f;
constexpr float kLane0LatM = -1.95f, kLane1LatM = 1.95f;
constexpr float kAdjacentLaneSepM = 3.90f, kLane0BlockFwdM = 7.5f;
constexpr float kLane1BlockFwdM = 16.0f, kClosePedFwdM = 4.2f;
constexpr float kApproachSpeedMs = 6.0f, kCloseObsMaxM = 6.0f;
constexpr float kVehCenterYM = 0.70f, kVehOpacity = 0.94f;
constexpr float kVehScaleXM = 0.85f, kVehScaleZM = 1.40f;
constexpr float kPedCenterYM = 0.75f, kPedOpacity = 0.96f;
constexpr float kPedScaleM = 0.55f, kClosePedScaleM = 0.65f;
constexpr float kDynCrossFwdM = 12.0f, kDynStartRightM = 4.2f;
constexpr float kDynCrossVelZMs = -4.2f, kDynVehScaleXM = 0.95f;
constexpr float kDynLookaheadM = 22.0f, kStopHorizonM = 24.0f;
constexpr float kMinBlockedGaussCost = 20.0f, kMaxClearGaussFrac = 0.2f;
constexpr float kEdgePedFwdM = 11.0f, kEdgePedOffZM = 0.55f;
constexpr float kNudgeLookaheadM = 20.0f, kEvalRssDistM = 4.0f;
constexpr float kEvalSlowEgoSpeedMs = 0.4f, kEvalLeadFwdM = 2.5f;
constexpr float kEvalCorrEndM = 4.5f, kEvalLeadVelXMs = 1.8f;
constexpr float kEvalInitObsM = 40.0f, kEvalInitSpeedMs = 8.0f;
constexpr float kEvalMinTrackSpeedMs = 1.0f;
const Vec4 kVehDebugColor{0.9f, 0.3f, 0.2f, 0.95f};
const Vec4 kPedDebugColor{0.2f, 0.9f, 0.4f, 0.95f};

float semF(SemanticClass sc) {
  return static_cast<float>(static_cast<int>(sc));
}

void parkTrafficAway(WorldEngine& world) {
  for (auto& v : world.mutableVehicles()) {
    v.pos = {kOffMapPosM, 0.0f, kOffMapPosM};
    v.speed = v.vx = v.vz = 0.0f;
  }
  for (auto& p : world.mutablePedestrians()) {
    p.x = kOffMapPedM; p.z = kOffMapPedM;
    p.ax = p.bx = p.targetX = p.x; p.az = p.bz = p.targetZ = p.z;
  }
}

TEST(V15GaussianPlannerTest, UnboxedGaussianObstacleTriggersSwitchOrStop) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = "v15";
  WorldEngine world(cfg);
  parkTrafficAway(world);
  const EgoState& ego = world.ego();
  const float bumpX = ego.pos.x + ego.centerOffsetForward + ego.hz;
  const float pedS = semF(SemanticClass::Pedestrian);
  const float vehS = semF(SemanticClass::Vehicle);

  PerceptionFrameOutput perc{};
  perc.v15.active = true;
  perc.v15Gaussians = {{{bumpX + kLane0BlockFwdM, kVehCenterYM, ego.pos.z,
                          kVehOpacity},
                         {kVehScaleXM, kVehCenterYM, kVehScaleZM, vehS},
                         {0.0f, 0.0f, ego.yaw, 0.0f}, kVehDebugColor}};
  PlannerCommand cmd{};
  v15::AutoregressivePolicy::plan(cfg, world, perc, cmd);
  EXPECT_TRUE(cmd.candidateBlocked.at(0));
  EXPECT_FALSE(cmd.candidateBlocked.at(1));
  EXPECT_EQ(cmd.bestCandIdx, 1);

  const V15Gaussian3D lane1Veh = {
    {bumpX + kLane1BlockFwdM, kVehCenterYM, ego.pos.z - kAdjacentLaneSepM,
     kVehOpacity},
    {kVehScaleXM, kVehCenterYM, kVehScaleZM, vehS},
    {0.0f, 0.0f, ego.yaw, 0.0f}, kVehDebugColor};
  perc.v15Gaussians.push_back(lane1Veh);
  v15::AutoregressivePolicy::plan(cfg, world, perc, cmd);
  EXPECT_TRUE(cmd.candidateBlocked.at(0));
  EXPECT_TRUE(cmd.candidateBlocked.at(1));

  perc.v15Gaussians = {
    lane1Veh,
    {{bumpX + kClosePedFwdM, kPedCenterYM, ego.pos.z, kPedOpacity},
     {kClosePedScaleM, kPedCenterYM, kClosePedScaleM, pedS},
     {0.0f, 0.0f, 0.0f, 0.0f}, kPedDebugColor}};
  world.mutableEgo().speed = kApproachSpeedMs;
  v15::AutoregressivePolicy::plan(cfg, world, perc, cmd);
  EXPECT_TRUE(cmd.aebActive);
  EXPECT_FLOAT_EQ(cmd.targetSpeed, 0.0f);
  EXPECT_LT(cmd.minObstacleDist, kCloseObsMaxM);
  EXPECT_EQ(cmd.behaviorState, "AEB_EMERGENCY_BRAKE");
}

TEST(V15GaussianPlannerTest, Dynamic4DGaussianFutureCrossingPenalized) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = "v15";
  WorldEngine world(cfg);
  parkTrafficAway(world);
  const EgoState& ego = world.ego();
  const float u0 = world.findClosestLoopU(ego.pos);
  const float vehS = semF(SemanticClass::Vehicle);
  const std::vector<V15Gaussian3D> dynGs = {
    {{ego.pos.x + kDynCrossFwdM, kVehCenterYM, ego.pos.z + kDynStartRightM,
      kVehOpacity},
     {kDynVehScaleXM, kVehCenterYM, kVehScaleZM, vehS},
     {0.0f, kDynCrossVelZMs, ego.yaw, 1.0f}, kVehDebugColor}};
  const auto cLane0 = v15::selectComfortSubRollout(
    kLane0LatM, ego.currentLateralOffset, kLane0LatM, ego, world.centerline(),
    u0, kDynLookaheadM, kStopHorizonM, dynGs, cfg.v15Policy);
  const auto cLane1 = v15::selectComfortSubRollout(
    kLane1LatM, ego.currentLateralOffset, kLane1LatM, ego, world.centerline(),
    u0, kDynLookaheadM, kStopHorizonM, dynGs, cfg.v15Policy);
  EXPECT_GT(cLane0.gaussianCost, kMinBlockedGaussCost);
  EXPECT_LT(cLane1.gaussianCost, cLane0.gaussianCost * kMaxClearGaussFrac);
}

TEST(V15GaussianPlannerTest, MultiRolloutNudgesAndJerkPenaltiesBehaveCleanly) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  WorldEngine world(cfg);
  const EgoState& ego = world.ego();
  const float u0 = world.findClosestLoopU(ego.pos);
  const std::vector<V15Gaussian3D> edgeGs = {
    {{ego.pos.x + kEdgePedFwdM, kPedCenterYM, ego.pos.z + kEdgePedOffZM,
      kPedOpacity},
     {kPedScaleM, kPedCenterYM, kPedScaleM, semF(SemanticClass::Pedestrian)},
     {0.0f, 0.0f, 0.0f, 0.0f}, kPedDebugColor}};
  const auto ch = v15::selectComfortSubRollout(
    kLane0LatM, kLane0LatM, kLane0LatM, ego, world.centerline(), u0,
    kNudgeLookaheadM, kStopHorizonM, edgeGs, cfg.v15Policy);
  EXPECT_GE(std::abs(ch.targetLatM), cfg.v15Policy.minLaneAbsM);
  EXPECT_LE(std::abs(ch.targetLatM), cfg.v15Policy.maxLaneAbsM);
  EXPECT_GT(ch.targetLatM, kLane0LatM);

  const auto smooth = v15::selectComfortSubRollout(
    kLane0LatM, kLane0LatM, kLane0LatM, ego, world.centerline(), u0,
    kNudgeLookaheadM, kStopHorizonM, {}, cfg.v15Policy);
  const auto jumped = v15::selectComfortSubRollout(
    kLane0LatM, kLane0LatM, kLane1LatM, ego, world.centerline(), u0,
    kNudgeLookaheadM, kStopHorizonM, {}, cfg.v15Policy);
  EXPECT_LT(smooth.comfortJerkCost, jumped.comfortJerkCost);
}

TEST(V15GaussianPlannerTest, DynamicLeadGaussianPullingAwayDoesNotPinZero) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  WorldEngine world(cfg);
  EgoState ego = world.ego();
  ego.speed = kEvalSlowEgoSpeedMs;
  const float bumpX = ego.pos.x + ego.centerOffsetForward + ego.hz;
  const std::vector<V15Gaussian3D> leadGs = {
    {{bumpX + kEvalLeadFwdM, kVehCenterYM, ego.pos.z, kVehOpacity},
     {kVehScaleXM, kVehCenterYM, kVehScaleZM, semF(SemanticClass::Vehicle)},
     {kEvalLeadVelXMs, 0.0f, ego.yaw, 1.0f}, kVehDebugColor}};
  const std::vector<Vec3> corrPts = {
    {bumpX, kVehCenterYM, ego.pos.z},
    {bumpX + kEvalLeadFwdM, kVehCenterYM, ego.pos.z},
    {bumpX + kEvalCorrEndM, kVehCenterYM, ego.pos.z}};
  const std::vector<float> corrArc = {0.0f, kEvalLeadFwdM, kEvalCorrEndM};
  float minObs = kEvalInitObsM, desiredSpd = kEvalInitSpeedMs;
  bool aeb = false;
  std::string behavior = "CRUISING";
  v15::evaluateCorridorGaussians(
    leadGs, corrPts, corrArc, kEvalRssDistM, ego, cfg.v15Policy, minObs,
    desiredSpd, aeb, behavior);
  EXPECT_GT(desiredSpd, kEvalMinTrackSpeedMs);
  EXPECT_FALSE(aeb);
  EXPECT_EQ(behavior, "FOLLOWING_LEAD");
}

TEST(V15GaussianPlannerTest, GaussianComfortConfigBoundsValidated) {
  auto checkBad = [](const std::string& yaml, const std::string& tag) {
    const auto p =
      std::filesystem::temp_directory_path() / ("v15_gc_" + tag + ".yaml");
    std::ofstream(p) << yaml;
    EXPECT_THROW(V15PolicyConfig::loadFromYaml(p.string()), std::runtime_error);
    std::filesystem::remove(p);
  };
  checkBad("gaussian_comfort:\n  gaussian_occ_thresh: 0.0\n", "occ0");
  checkBad("gaussian_comfort:\n  gaussian_safety_stop_prob: 1.0\n", "stop1");
  checkBad("gaussian_comfort:\n  speed_backoff_ratio: 1.2\n", "spd");
}
}  // namespace
}  // namespace cybercab
