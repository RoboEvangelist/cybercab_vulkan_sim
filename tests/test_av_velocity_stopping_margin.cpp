#include "AvValidationMetrics.hpp"
#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <string>

namespace cybercab {
namespace {
using AvVelocityRangeTest = testutil::SharedGpuMetricsTest;
using testutil::fmtProp;
using testutil::rssStopDistM;
using SC = SemanticClass;

constexpr int kVelF = 90;
constexpr float kLeadGapM = 44.0f, kLeadSpd = 2.0f, kU0 = 0.01f;
constexpr float kLat0 = -1.9f, kTrkGate = 6.0f, kOff = -350.0f;

void runVelocityRangeCase(VulkanContext& ctx, const char* v, float spdMs,
                          const char* spdTag) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = v; cfg.initialResMode = "half";
  cfg.targetCruiseSpeedMs = spdMs;
  WorldEngine w(cfg);
  for (size_t i = 1; i < w.mutableVehicles().size(); ++i) {
    auto& vh = w.mutableVehicles()[i]; vh.pos = {kOff, 0.f, kOff};
    vh.speed = vh.vx = vh.vz = 0.0f;
  }
  for (auto& g : w.mutablePedestrians())
    g.x = g.ax = g.bx = g.targetX = g.z = g.az = g.bz = g.targetZ = kOff;

  const float len = w.centerline().getLength();
  EgoState& e = w.mutableEgo(); e.loopProgressU = kU0;
  e.currentLateralOffset = kLat0; e.speed = spdMs;
  e.pos = w.laneCurve(0).getPointAt(kU0);
  const Vec3 eTan = w.laneCurve(0).getTangentAt(kU0);
  e.yaw = std::atan2(eTan.x, eTan.z);

  auto& trk = w.mutableVehicles().at(0); trk.lane = 0;
  trk.u = std::fmod(kU0 + kLeadGapM / len, 1.0f);
  trk.speed = trk.baseSpeed = kLeadSpd;
  trk.pos = w.laneCurve(0).getPointAt(trk.u);
  const Vec3 lTan = w.laneCurve(0).getTangentAt(trk.u);
  trk.yaw = std::atan2(lTan.x, lTan.z);
  trk.vx = lTan.x * trk.speed; trk.vz = lTan.z * trk.speed;

  CameraRigRenderer r; r.init(ctx, cfg, w);
  PerceptionPipeline p(cfg); p.init(ctx, r); TrajectoryPlanner pl(cfg);
  const float dt = 1.0f / cfg.controlHz, rssReq = rssStopDistM(spdMs, cfg);
  PerceptionFrameOutput out{}; PlannerCommand cmd{};
  float firstDetM = 0.0f, firstTrkM = 0.0f;

  for (int s = 0; s < kVelF; ++s) {
    r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, nullptr);
    p.runFrame(ctx, r, w, dt, true, out, nullptr); pl.plan(w, out, cmd);
    EXPECT_FALSE(w.lastContact().occurred) << v << " " << spdTag;
    const auto& lead = w.vehicles().at(0);
    if (firstDetM <= 0.0f) {
      for (const auto& d : out.detections3D)
        if (d.semantic == SC::Vehicle &&
            std::hypot(d.worldPos.x - lead.pos.x,
                       d.worldPos.z - lead.pos.z) < kTrkGate) {
          firstDetM = d.rangeM; break;
        }
    }
    if (firstTrkM <= 0.0f) {
      for (const auto& tr : out.tracks)
        if (tr.semantic == SC::Vehicle && tr.hits >= 2 &&
            std::hypot(tr.worldPos.x - lead.pos.x,
                       tr.worldPos.z - lead.pos.z) < kTrkGate) {
          firstTrkM = std::hypot(tr.worldPos.x - w.ego().pos.x,
                                 tr.worldPos.z - w.ego().pos.z);
          break;
        }
    }
    w.stepAgents(dt); w.stepEgoAndPhysics(dt, cmd, {});
  }
  const std::string pfx = std::string(v) + "_" + spdTag + "_";
  testing::Test::RecordProperty(pfx + "det_m", fmtProp(firstDetM));
  testing::Test::RecordProperty(pfx + "trk_m", fmtProp(firstTrkM));
  testing::Test::RecordProperty(pfx + "rss_m", fmtProp(rssReq));
  EXPECT_GT(firstDetM, rssReq) << pfx;
  EXPECT_GT(firstTrkM, rssReq) << pfx;
  p.destroy(); r.destroy();
}

TEST_F(AvVelocityRangeTest, DetectionDistanceVsEgoVelocityStoppingMargin) {
  runVelocityRangeCase(s_ctx, "v14", 5.0f, "v5");
  runVelocityRangeCase(s_ctx, "v14", 10.5f, "v10");
  runVelocityRangeCase(s_ctx, "v14", 14.0f, "v14");
  runVelocityRangeCase(s_ctx, "v15", 5.0f, "v5");
  runVelocityRangeCase(s_ctx, "v15", 10.5f, "v10");
  runVelocityRangeCase(s_ctx, "v15", 14.0f, "v14");
}
}  // namespace
}  // namespace cybercab
