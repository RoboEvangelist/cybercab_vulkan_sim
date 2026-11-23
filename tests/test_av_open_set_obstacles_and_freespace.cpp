#include "AvValidationMetrics.hpp"
#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "planning/v15/StaticVoxelGuard.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <string>

namespace cybercab {
namespace {
using AvOpenSetObstacleTest = testutil::SharedGpuMetricsTest;
using testutil::fmtProp;
using testutil::isPointInOBB;
using SC = SemanticClass;

constexpr int kSimSteps = 130;
constexpr int kWarmupF = 2;
constexpr float kOff = -350.0f;
constexpr float kInitSpd = 8.5f;
constexpr float kObsPadM = 2.2f;
constexpr float kEgoStartX = -36.0f;
constexpr float kEgoStartLatZ = 1.9f;
constexpr float kYawHalf = 0.5f;
constexpr float kObsPosX = -14.0f;
constexpr float kObsLatZ = 1.9f;
constexpr float kMinVoxConf = 0.50f;
constexpr float kMaxFsErrM = 2.5f;
constexpr float kMinVoxYM = 0.30f;
constexpr float kMaxVoxYM = 1.95f;
constexpr float kMaxRoadLatM = 3.2f;
constexpr float kStopGapPadM = 0.5f;
constexpr float kStopMaxSpdMs = 1.2f;
constexpr float kPassMinLatM = 0.5f;
constexpr float kPassBehindObsM = 2.0f;
constexpr float kMinSafeClearM = 0.05f;
constexpr float kInitDistSentinel = 99.0f;
constexpr float kCargoHx = 0.85f;
constexpr float kCargoHy = 0.60f;
constexpr float kCargoHz = 0.85f;
const Vec3 kCargoColor{0.84f, 0.48f, 0.12f};
constexpr float kBarrierHx = 1.10f;
constexpr float kBarrierHy = 0.55f;
constexpr float kBarrierHz = 0.55f;
const Vec3 kBarrierColor{0.56f, 0.60f, 0.65f};

void clearRoadUsers(WorldEngine& w) {
  for (auto& vh : w.mutableVehicles()) {
    vh.pos = {kOff, 0.0f, kOff};
    vh.speed = 0.0f;
    vh.vx = 0.0f;
    vh.vz = 0.0f;
  }
  for (auto& p : w.mutablePedestrians()) {
    p.x = kOff;
    p.ax = kOff;
    p.bx = kOff;
    p.targetX = kOff;
    p.z = kOff;
    p.az = kOff;
    p.bz = kOff;
    p.targetZ = kOff;
  }
}

float computeObbClearanceM(const OBB& ego, const UnknownObstacleSpec& obs) {
  const float gapX = std::max(0.0f, std::abs(ego.cx - obs.cx) -
                                      (ego.hx + obs.hx));
  const float gapZ = std::max(0.0f, std::abs(ego.cz - obs.cz) -
                                      (ego.hz + obs.hz));
  return std::hypot(gapX, gapZ);
}

void measureObstacleVoxels(const WorldEngine& w, const OBB& obsObb,
                           float obsFrontX, const PerceptionFrameOutput& out,
                           int& obsVox, float& minFsErrM) {
  for (const auto& vx : out.activeVoxels) {
    if (vx.confidence < kMinVoxConf ||
        !isPointInOBB(obsObb, vx.wx, vx.wz, kObsPadM) ||
        vx.wy < kMinVoxYM || vx.wy > kMaxVoxYM) {
      continue;
    }
    ++obsVox;
    if (std::abs(v15::voxelRoadLateralM(w, vx)) <= kMaxRoadLatM) {
      minFsErrM = std::min(minFsErrM, std::abs(vx.wx - obsFrontX));
    }
  }
}

void runUnknownObstacleCase(VulkanContext& ctx, const char* resMode,
                            const UnknownObstacleSpec& obs, const char* tag) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = "v15";
  cfg.initialResMode = resMode;
  WorldEngine w(cfg);
  clearRoadUsers(w);
  EgoState& e = w.mutableEgo();
  e.pos = {kEgoStartX, 0.0f, WorldEngine::kLoopHalfZ + kEgoStartLatZ};
  e.yaw = kPi * kYawHalf;
  e.speed = kInitSpd;
  w.addUnknownRoadObstacle(obs);

  CameraRigRenderer r;
  r.init(ctx, cfg, w);
  PerceptionPipeline p(cfg);
  p.init(ctx, r);
  TrajectoryPlanner pl(cfg);
  const float dt = 1.0f / cfg.controlHz;
  const OBB obsObb{obs.id, obs.cx, obs.cy, obs.cz, obs.hx, obs.hy, obs.hz,
                   obs.yaw, 0.0f, 0.0f, SC::Static, ""};
  const float obsFrontX = obs.cx - obs.hx;

  PerceptionFrameOutput out{};
  PlannerCommand cmd{};
  int obsVox = 0;
  float minFsErrM = kInitDistSentinel;
  float minClearM = kInitDistSentinel;

  for (int s = 0; s < kSimSteps; ++s) {
    r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, nullptr);
    p.runFrame(ctx, r, w, dt, true, out, nullptr);
    pl.plan(w, out, cmd);
    EXPECT_FALSE(w.lastContact().occurred) << tag << " step=" << s;
    minClearM =
      std::min(minClearM, computeObbClearanceM(w.getEgoWorldOBB(), obs));
    if (s >= kWarmupF) {
      measureObstacleVoxels(w, obsObb, obsFrontX, out, obsVox, minFsErrM);
    }
    w.stepAgents(dt);
    w.stepEgoAndPhysics(dt, cmd, {});
  }

  const EgoState& ef = w.ego();
  const bool stoppedSafe = (ef.pos.x + ef.hz < obsFrontX + kStopGapPadM) &&
                           (ef.speed < kStopMaxSpdMs);
  const bool changedLane = (ef.currentLateralOffset > kPassMinLatM) &&
                           (ef.pos.x > obs.cx - kPassBehindObsM);
  const std::string pfx = std::string(tag) + "_";
  testing::Test::RecordProperty(pfx + "vox", std::to_string(obsVox));
  testing::Test::RecordProperty(pfx + "fs_err_m", fmtProp(minFsErrM));
  testing::Test::RecordProperty(pfx + "min_clear_m", fmtProp(minClearM));
  EXPECT_GT(obsVox, 0) << tag;
  EXPECT_LE(minFsErrM, kMaxFsErrM) << tag;
  EXPECT_GT(minClearM, kMinSafeClearM) << tag;
  EXPECT_TRUE(stoppedSafe || changedLane)
    << tag << " finalX=" << ef.pos.x << " lat=" << ef.currentLateralOffset
    << " spd=" << ef.speed;
  p.destroy();
  r.destroy();
}

TEST_F(AvOpenSetObstacleTest,
       OpenSetUnknownRoadObstaclesFreespaceIoUAndClosedLoopSafety) {
  const float obsZ = WorldEngine::kLoopHalfZ + kObsLatZ;
  const UnknownObstacleSpec cargo{
    "FALLEN_CARGO", kObsPosX, kCargoHy, obsZ, kCargoHx, kCargoHy, kCargoHz,
    0.0f, kCargoColor, true};
  const UnknownObstacleSpec barrier{
    "JERSEY_BARRIER", kObsPosX, kBarrierHy, obsZ, kBarrierHx, kBarrierHy,
    kBarrierHz, 0.0f, kBarrierColor, true};
  runUnknownObstacleCase(s_ctx, "half", cargo, "cargo_half");
  runUnknownObstacleCase(s_ctx, "full", cargo, "cargo_full");
  runUnknownObstacleCase(s_ctx, "half", barrier, "barrier_half");
}

}  // namespace
}  // namespace cybercab
