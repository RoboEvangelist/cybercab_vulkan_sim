#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace cybercab::sota {
namespace {

using LeWmHJepaOpenSetTest = testutil::SharedGpuMetricsTest;

constexpr float kOffMapM = -350.0f;

struct OpenSetCfg {
  const char* label;
  const char* spatial;
  const char* wm;
};

TEST_F(LeWmHJepaOpenSetTest, BarrierAndCargoDownstreamSafetyAcrossWorldModels) {
  const float obsZ = WorldEngine::kLoopHalfZ + 1.9f;
  const UnknownObstacleSpec obstacles[] = {
    {"CARGO", -14.0f, 0.60f, obsZ, 0.85f, 0.60f, 0.85f, 0.0f, {0.84f, 0.48f, 0.12f}, true},
    {"BARRIER", -14.0f, 0.55f, obsZ, 1.10f, 0.55f, 0.55f, 0.0f, {0.56f, 0.60f, 0.65f}, true},
  };
  const OpenSetCfg cfgs[] = {
    {"legacy", "legacy", "legacy"},
    {"lewm", "legacy", "lewm"},
    {"hjepa", "legacy", "hjepa"},
    {"gf3d+hjepa", "gaussianformer3d", "hjepa"},
  };
  for (const auto& obs : obstacles) {
    const OBB obsObb{obs.id, obs.cx, obs.cy, obs.cz, obs.hx, obs.hy, obs.hz,
                     0.0f, 0.0f, 0.0f, SemanticClass::Static, ""};
    for (const auto& mc : cfgs) {
      SimConfig cfg =
        SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
      cfg.pipelineVersion = "v15";
      cfg.spatialRepMode = mc.spatial;
      cfg.worldModelMode = mc.wm;
      WorldEngine w(cfg);
      for (auto& vh : w.mutableVehicles()) {
        vh.pos = {kOffMapM, 0.0f, kOffMapM};
        vh.speed = vh.vx = vh.vz = 0.0f;
      }
      for (auto& p : w.mutablePedestrians()) {
        p.x = p.ax = p.bx = p.targetX = kOffMapM;
        p.z = p.az = p.bz = p.targetZ = kOffMapM;
      }
      w.addUnknownRoadObstacle(obs);
      EgoState& e0 = w.mutableEgo();
      e0.pos = {-36.0f, 0.0f, obsZ};
      e0.yaw = kPi * 0.5f;
      e0.speed = 8.5f;
      e0.currentLateralOffset = -1.9f;

      CameraRigRenderer rig; rig.init(s_ctx, cfg, w);
      PerceptionPipeline pipe(cfg); pipe.init(s_ctx, rig);
      TrajectoryPlanner planner(cfg);
      PerceptionFrameOutput out{};
      PlannerCommand cmd{};
      const float dt = 1.0f / cfg.controlHz;
      int cols = 0, obsVox = 0;
      float minClr = 99.0f;
      for (int s = 0; s < 130; ++s) {
        rig.renderAllCameras(w, cfg, static_cast<float>(s) * dt);
        pipe.runFrame(s_ctx, rig, w, dt, true, out);
        planner.plan(w, out, cmd);
        if (w.lastContact().occurred) ++cols;
        const OBB eo = w.getEgoWorldOBB();
        minClr = std::min(minClr, std::hypot(
          std::max(0.0f, std::abs(eo.cx - obs.cx) - (eo.hx + obs.hx)),
          std::max(0.0f, std::abs(eo.cz - obs.cz) - (eo.hz + obs.hz))));
        if (s >= 2) {
          for (const auto& vx : out.activeVoxels) {
            if (vx.confidence >= 0.50f &&
                testutil::isPointInOBB(obsObb, vx.wx, vx.wz, 2.2f) &&
                vx.wy >= 0.30f && vx.wy <= 1.95f) {
              ++obsVox;
            }
          }
        }
        w.stepAgents(dt);
        w.stepEgoAndPhysics(dt, cmd, {});
      }
      const EgoState& ef = w.ego();
      const bool avoided = ((ef.pos.x + ef.hz < (obs.cx - obs.hx) + 0.5f) && ef.speed < 1.2f) ||
                           (ef.currentLateralOffset > 0.5f && ef.pos.x > obs.cx - 2.0f);
      std::printf("[SOTA_OPENSET] %-7s %-10s | col=%d vox=%d clear=%.2fm avoided=%d end=(%.1f,%.2f,%.2f)\n",
                  obs.id.c_str(), mc.label, cols, obsVox, minClr, avoided ? 1 : 0,
                  ef.pos.x, ef.currentLateralOffset, ef.speed);
      EXPECT_EQ(cols, 0) << obs.id << " " << mc.label;
      EXPECT_TRUE(avoided) << obs.id << " " << mc.label;
      EXPECT_GT(minClr, 0.05f) << obs.id << " " << mc.label;
      EXPECT_GT(obsVox, 0) << obs.id << " " << mc.label;
      pipe.destroy();
      rig.destroy();
    }
  }
}

}  // namespace
}  // namespace cybercab::sota
