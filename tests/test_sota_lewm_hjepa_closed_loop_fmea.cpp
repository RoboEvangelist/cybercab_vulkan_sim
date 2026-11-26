#include "MetricsEvalHelpers.hpp"
#include "core/TraceLogger.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/sota/LeWmTypes.hpp"
#include "planning/TrajectoryPlanner.hpp"

#include <gtest/gtest.h>
#include <json/json.h>
#include <filesystem>
#include <sstream>
#include <string>

namespace cybercab::sota {
namespace {

using LeWmHJepaGpuTest = testutil::SharedGpuMetricsTest;

SimConfig makeWmCfg(const std::string& pipeVer,
                    const std::string& wmMode,
                    const std::string& spatialMode = "legacy",
                    const std::string& weightsPath = "") {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = pipeVer;
  cfg.worldModelMode = wmMode;
  cfg.spatialRepMode = spatialMode;
  cfg.wmSota.weightsPath = weightsPath;
  return cfg;
}

TEST_F(LeWmHJepaGpuTest, YamlDefaultLegacyRemainsBitwiseInvariant) {
  const SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  EXPECT_EQ(cfg.worldModelMode, "legacy");
  EXPECT_FALSE(cfg.usesSotaWorldModel());

  WorldEngine world(cfg);
  CameraRigRenderer rig;
  rig.init(s_ctx, cfg, world);
  PerceptionPipeline pipe(cfg);
  pipe.init(s_ctx, rig);

  EXPECT_EQ(pipe.worldModelSota(), nullptr);
  EXPECT_EQ(pipe.hjepaHierarchy(), nullptr);
  PerceptionFrameOutput out{};
  rig.renderAllCameras(world, cfg, 0.0f);
  pipe.runFrame(s_ctx, rig, world, 1.0f / cfg.controlHz, true, out);
  EXPECT_FALSE(out.worldModelSota.active);

  const std::string js = TraceLogger::formatFrameJson(
    0, 0.0f, cfg, world.ego(), PlannerCommand{}, out, {});
  Json::CharReaderBuilder rb;
  Json::Value root;
  std::string errs;
  std::istringstream iss(js);
  ASSERT_TRUE(Json::parseFromStream(rb, iss, &root, &errs)) << errs;
  EXPECT_FALSE(root.isMember("world_model_sota"));

  pipe.destroy();
  rig.destroy();
}

TEST_F(LeWmHJepaGpuTest,
       ClosedLoopV15V14AndGf3dCompositionZeroCollisionsAndWeightParity) {
  const auto binPath =
    std::filesystem::temp_directory_path() / "cybercab_lewm_gpu_parity.bin";
  ASSERT_TRUE(LeWmWeights::defaultAnalytical().saveBinary(binPath.string()));

  auto runShortLoop = [&](const SimConfig& cfg, int steps, EgoState& finalEgo,
                          WorldModelSotaSummary& lastWm) {
    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    PerceptionPipeline pipe(cfg);
    pipe.init(s_ctx, rig);
    TrajectoryPlanner planner(cfg);
    PerceptionFrameOutput out{};
    PlannerCommand cmd{};
    const float dt = 1.0f / cfg.controlHz;
    for (int s = 0; s < steps; ++s) {
      rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt);
      pipe.runFrame(s_ctx, rig, world, dt, true, out);
      planner.plan(world, out, cmd);
      world.stepAgents(dt);
      world.stepEgoAndPhysics(dt, cmd, {});
      EXPECT_FALSE(world.lastContact().occurred);
    }
    finalEgo = world.ego();
    lastWm = out.worldModelSota;
    pipe.destroy();
    rig.destroy();
  };

  EgoState egoAna{}, egoBin{};
  WorldModelSotaSummary wmAna{}, wmBin{};
  runShortLoop(makeWmCfg("v15", "lewm"), 20, egoAna, wmAna);
  runShortLoop(makeWmCfg("v15", "lewm", "legacy", binPath.string()), 20,
               egoBin, wmBin);
  std::filesystem::remove(binPath);

  EXPECT_TRUE(wmAna.active);
  EXPECT_FALSE(wmAna.hierarchical);
  EXPECT_FALSE(wmAna.weightsLoadedFromFile);
  EXPECT_TRUE(wmBin.weightsLoadedFromFile);
  EXPECT_FALSE(wmAna.fmeaFallbackTriggered);
  EXPECT_FALSE(wmBin.fmeaFallbackTriggered);
  EXPECT_FLOAT_EQ(egoAna.pos.x, egoBin.pos.x);
  EXPECT_FLOAT_EQ(egoAna.pos.z, egoBin.pos.z);
  EXPECT_FLOAT_EQ(egoAna.speed, egoBin.speed);
  EXPECT_FLOAT_EQ(wmAna.predSurprise, wmBin.predSurprise);

  EgoState egoMix{}, egoV14{};
  WorldModelSotaSummary wmMix{}, wmV14{};
  runShortLoop(makeWmCfg("v15", "hjepa", "gaussianformer3d"), 24, egoMix, wmMix);
  EXPECT_TRUE(wmMix.active);
  EXPECT_TRUE(wmMix.hierarchical);
  EXPECT_GE(wmMix.macroSubgoalCode, 0);
  EXPECT_LE(wmMix.macroSubgoalCode, 4);
  EXPECT_FALSE(wmMix.fmeaFallbackTriggered);

  runShortLoop(makeWmCfg("v14", "hjepa"), 24, egoV14, wmV14);
  EXPECT_TRUE(wmV14.active);
  EXPECT_TRUE(wmV14.hierarchical);
  EXPECT_FALSE(wmV14.fmeaFallbackTriggered);
}

TEST_F(LeWmHJepaGpuTest, DegradedSoilFrontMrmAndEmptyRoadZeroHallucinations) {
  // 1. Degraded forward cameras (0, 1, 2) -> FMEA SENSOR_SOILED_MRM still triggers cleanly with 0 collisions.
  for (const char* wm : {"lewm", "hjepa"}) {
    SimConfig cfg = makeWmCfg("v15", wm);
    cfg.cameras[0].soiled = true;
    cfg.cameras[1].soiled = true;
    cfg.cameras[2].soiled = true;
    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    PerceptionPipeline pipe(cfg);
    pipe.init(s_ctx, rig);
    TrajectoryPlanner planner(cfg);
    PerceptionFrameOutput out{};
    PlannerCommand cmd{};
    bool seenMrm = false;
    const float dt = 1.0f / cfg.controlHz;
    for (int s = 0; s < 30; ++s) {
      rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt);
      pipe.runFrame(s_ctx, rig, world, dt, true, out);
      planner.plan(world, out, cmd);
      world.stepAgents(dt);
      world.stepEgoAndPhysics(dt, cmd, {});
      EXPECT_FALSE(world.lastContact().occurred) << wm;
      if (cmd.behaviorState == "SENSOR_SOILED_MRM") seenMrm = true;
    }
    EXPECT_TRUE(seenMrm) << wm;
    pipe.destroy();
    rig.destroy();
  }

  // 2. Empty road -> 0 ghosts, 0 false AEB, zero FMEA collapse despite static road scene.
  for (const char* wm : {"lewm", "hjepa"}) {
    SimConfig cfg = makeWmCfg("v15", wm);
    WorldEngine world(cfg);
    world.mutableVehicles().clear();
    world.mutablePedestrians().clear();
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    PerceptionPipeline pipe(cfg);
    pipe.init(s_ctx, rig);
    TrajectoryPlanner planner(cfg);
    PerceptionFrameOutput out{};
    PlannerCommand cmd{};
    const float dt = 1.0f / cfg.controlHz;
    for (int s = 0; s < 16; ++s) {
      rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt);
      pipe.runFrame(s_ctx, rig, world, dt, true, out);
      planner.plan(world, out, cmd);
      world.stepEgoAndPhysics(dt, cmd, {});
      EXPECT_FALSE(cmd.aebActive) << wm << " step=" << s;
      EXPECT_EQ(out.eval.ghostVehDetections, 0) << wm;
      EXPECT_EQ(out.eval.ghostPedDetections, 0) << wm;
    }
    EXPECT_FALSE(out.worldModelSota.fmeaFallbackTriggered) << wm;
    pipe.destroy();
    rig.destroy();
  }
}

}  // namespace
}  // namespace cybercab::sota
