#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/v15/CameraSelfCalib.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "planning/v15/AutoregressivePolicy.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace cybercab {
namespace {

using V15AutoshiftAndCalibTest = testutil::SharedGpuMetricsTest;

constexpr float kOffMapPosM = -350.0f, kOffMapPedM = -320.0f;
constexpr float kLane1OffsetM = -3.90f, kBlockFwdM = 8.5f;
constexpr float kClearRearM = 18.0f, kMinReverseDeltaXM = 0.20f;
constexpr float kRearPedDistM = 2.5f, kZeroSpeedEps = 1e-4f;
constexpr int kReverseSteps = 22, kResumeSteps = 20;
constexpr float kKnockRollDeg = -1.8f, kKnockPitchDeg = 2.0f;
constexpr float kKnockRecovTolDeg = 0.45f, kCruiseLatM = -1.9f;
constexpr float kLoopStepU = 0.025f, kLoopStartU = 0.03f;

void parkWorldEntities(WorldEngine& world) {
  for (auto& v : world.mutableVehicles()) {
    v.pos = {kOffMapPosM, 0.0f, kOffMapPosM};
    v.speed = v.vx = v.vz = 0.0f;
  }
  for (auto& p : world.mutablePedestrians()) {
    p.x = kOffMapPedM; p.z = kOffMapPedM;
    p.ax = p.bx = p.targetX = p.x; p.az = p.bz = p.targetZ = p.z;
  }
}

Track3D makeStaticVeh(int id, float x, float z, float yaw) {
  Track3D t{};
  t.trackId = id; t.semantic = SemanticClass::Vehicle;
  t.score = 0.95f; t.hits = 3; t.yaw = yaw;
  t.halfExtents = {0.96f, 0.72f, 2.18f};
  t.worldPos = {x, t.halfExtents.y, z};
  return t;
}

TEST_F(V15AutoshiftAndCalibTest, AutoshiftReverseAndRearPedestrianAbort) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = "v15";
  WorldEngine world(cfg);
  parkWorldEntities(world);
  world.mutableEgo().speed = 0.0f;
  const float dt = 1.0f / cfg.controlHz;
  const float initX = world.ego().pos.x;
  const float lane0Z = world.ego().pos.z;
  const float lane1Z = lane0Z + kLane1OffsetM;

  PerceptionFrameOutput perc{};
  perc.v15.active = true;
  perc.v15.rearClearanceM = kClearRearM;
  perc.tracks = {
    makeStaticVeh(1, initX + kBlockFwdM, lane0Z, world.ego().yaw),
    makeStaticVeh(2, initX + kBlockFwdM, lane1Z, world.ego().yaw)};

  PlannerCommand cmd{};
  bool seenReverse = false, collision = false;
  for (int s = 0; s < kReverseSteps; ++s) {
    v15::AutoregressivePolicy::plan(cfg, world, perc, cmd);
    world.stepEgoAndPhysics(dt, cmd, {});
    if (cmd.gear == -1 && cmd.behaviorState == "AUTOSHIFT_REVERSE") {
      seenReverse = true;
    }
    if (world.lastContact().occurred) collision = true;
  }
  const float reverseDeltaX = world.ego().pos.x - initX;
  RecordProperty("autoshift_reverse_delta_x_m",
                 testutil::fmtProp(reverseDeltaX));
  EXPECT_TRUE(seenReverse);
  EXPECT_FALSE(collision);
  EXPECT_LT(reverseDeltaX, -kMinReverseDeltaXM);

  // Rear pedestrian enters safety zone -> immediate abort back to Drive + stop.
  perc.v15.rearClearanceM = kRearPedDistM;
  v15::AutoregressivePolicy::plan(cfg, world, perc, cmd);
  EXPECT_EQ(cmd.gear, 1);
  EXPECT_FLOAT_EQ(cmd.targetSpeed, 0.0f);

  // Adjacent lane clears -> resume forward Drive motion cleanly.
  perc.tracks.pop_back();
  world.stepEgoAndPhysics(dt, cmd, {});
  for (int s = 0; s < kResumeSteps; ++s) {
    v15::AutoregressivePolicy::plan(cfg, world, perc, cmd);
    world.stepEgoAndPhysics(dt, cmd, {});
  }
  EXPECT_EQ(cmd.gear, 1);
  EXPECT_GT(cmd.targetSpeed, kZeroSpeedEps);
  EXPECT_GT(world.ego().speed, kZeroSpeedEps);
}

TEST_F(V15AutoshiftAndCalibTest, LiveCameraKnockSelfCalibRecoversPitchRoll) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = "v15";
  cfg.photonNoiseEnabled = true;
  cfg.cameraKnockRpyDeg = {kKnockRollDeg, kKnockPitchDeg, 0.0f};

  WorldEngine world(cfg);
  CameraRigRenderer rig;
  rig.init(s_ctx, cfg, world);
  PerceptionPipeline pipe(cfg);
  pipe.init(s_ctx, rig);

  PerceptionFrameOutput out{};
  const auto& loop = world.centerline();
  const float dt = 1.0f / cfg.controlHz;
  for (int f = 0; f < v15::CameraSelfCalib::kWindowFrames; ++f) {
    const float u = std::fmod(kLoopStartU + kLoopStepU * static_cast<float>(f),
                              1.0f);
    EgoState& e = world.mutableEgo();
    const Vec3 tan = loop.getTangentAt(u);
    e.pos = loop.getPointAt(u) + Vec3{tan.z, 0.0f, -tan.x} * kCruiseLatM;
    e.yaw = std::atan2(tan.x, tan.z);
    e.pitch = 0.0f;
    e.roll = 0.0f;
    rig.renderAllCameras(world, cfg, static_cast<float>(f) * dt, nullptr);
    pipe.runFrame(s_ctx, rig, world, dt, true, out);
  }
  RecordProperty("knock_recov_pitch_deg",
                 testutil::fmtProp(out.v15.calibPitchDeg));
  RecordProperty("knock_recov_roll_deg",
                 testutil::fmtProp(out.v15.calibRollDeg));
  EXPECT_NEAR(out.v15.calibPitchDeg, kKnockPitchDeg, kKnockRecovTolDeg);
  EXPECT_NEAR(out.v15.calibRollDeg, kKnockRollDeg, kKnockRecovTolDeg);

  pipe.destroy();
  rig.destroy();
}

}  // namespace
}  // namespace cybercab
