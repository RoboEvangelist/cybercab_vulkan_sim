#include "AvLaneCorridorMetrics.hpp"
#include "AvValidationMetrics.hpp"
#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace cybercab {
namespace {
using AvMotLaneValidationTest = testutil::SharedGpuMetricsTest;
using testutil::evalGroundLaneIoU;
using testutil::fmtProp;
using testutil::LaneCorridorMetrics;
using testutil::MotLaneMonitor;

constexpr int kStdF = 56, kNatF = 22, kMaxIdsw = 2;
constexpr int kRedMax = 180, kGreenMax = 360;
constexpr double kMinMota = 0.45, kMaxMotp = 1.5, kMinAssoc = 0.85;
constexpr double kMinIou = 0.65, kMaxEdgeErr = 1.2;
constexpr float kStopSpd = 0.25f, kOverM = 0.05f, kMaxMargM = 6.0f;
constexpr float kGoSpd = 1.0f, kMinSigDistM = 5.0f, kMaxGreenS = 1.2f;
constexpr float kOffMap = -350.0f;

SimConfig mkPreset(const char* v, const char* m, bool night, bool soil) {
  SimConfig c =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  c.pipelineVersion = v; c.initialResMode = m;
  if (night) { c.lightingMode = "night_glare"; c.photonRccbMosaic = true; }
  if (soil) { c.cameras[0].soiled = true; }
  return c;
}

void runMotCase(VulkanContext& ctx, const char* v, const char* m, int frames,
                bool night, bool soil, const char* tag) {
  const SimConfig cfg = mkPreset(v, m, night, soil);
  WorldEngine w(cfg); CameraRigRenderer r; r.init(ctx, cfg, w);
  PerceptionPipeline p(cfg); p.init(ctx, r); TrajectoryPlanner pl(cfg);
  const float dt = 1.0f / cfg.controlHz;
  PerceptionFrameOutput out{}; PlannerCommand cmd{};
  MotLaneMonitor mot; LaneCorridorMetrics lane;
  for (int s = 0; s < frames; ++s) {
    r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, nullptr);
    p.runFrame(ctx, r, w, dt, true, out, nullptr); pl.plan(w, out, cmd);
    EXPECT_FALSE(w.lastContact().occurred) << tag << " step=" << s;
    if (s >= 2) {
      mot.update(w, out.tracks);
      const bool sampleIou =
        (s == 10 || s == 30 || s == 50 || (frames < 30 && s == 20));
      if (sampleIou) evalGroundLaneIoU(w, r, cfg, lane);
    }
    w.stepAgents(dt); w.stepEgoAndPhysics(dt, cmd, {});
  }
  const std::string pfx = std::string(tag) + "_";
  testing::Test::RecordProperty(pfx + "mota", fmtProp(mot.mota()));
  testing::Test::RecordProperty(pfx + "motp", fmtProp(mot.motp()));
  testing::Test::RecordProperty(pfx + "idsw", mot.idsw);
  testing::Test::RecordProperty(pfx + "idsw_per_100",
                                fmtProp(mot.idswPer100()));
  testing::Test::RecordProperty(pfx + "frag", mot.frag);
  testing::Test::RecordProperty(pfx + "cls_swaps", mot.clsSwaps);
  testing::Test::RecordProperty(pfx + "lane_assoc",
                                fmtProp(mot.laneAssocAcc()));
  testing::Test::RecordProperty(pfx + "lane_iou", fmtProp(lane.iou()));
  testing::Test::RecordProperty(pfx + "lane_edge_err_m",
                                fmtProp(lane.edgeErrM()));
  EXPECT_EQ(mot.dupGt, 0) << tag; EXPECT_EQ(mot.clsSwaps, 0) << tag;
  EXPECT_LE(mot.idsw, kMaxIdsw) << tag; EXPECT_GE(mot.mota(), kMinMota) << tag;
  EXPECT_LE(mot.motp(), kMaxMotp) << tag;
  EXPECT_GE(mot.laneAssocAcc(), kMinAssoc) << tag;
  EXPECT_GE(lane.iou(), kMinIou) << tag;
  EXPECT_LE(lane.edgeErrM(), kMaxEdgeErr) << tag;
  p.destroy(); r.destroy();
}

void runSignalStopAndGreenCase(VulkanContext& ctx, const char* v,
                               const char* m) {
  const SimConfig cfg = mkPreset(v, m, false, false);
  WorldEngine w(cfg); w.setForcedRedSignal(true);
  for (auto& veh : w.mutableVehicles()) {
    veh.pos = {kOffMap, 0.0f, kOffMap}; veh.speed = veh.vx = veh.vz = 0.0f;
  }
  for (auto& ped : w.mutablePedestrians())
    ped.x = ped.ax = ped.bx = ped.targetX =
      ped.z = ped.az = ped.bz = ped.targetZ = kOffMap;
  CameraRigRenderer r; r.init(ctx, cfg, w);
  PerceptionPipeline p(cfg); p.init(ctx, r); TrajectoryPlanner pl(cfg);
  const float dt = 1.0f / cfg.controlHz;
  PerceptionFrameOutput out{}; PlannerCommand cmd{};
  float redSigDetM = 0.0f, sigDetDistM = 0.0f; bool stoppedRed = false;
  for (int s = 0; s < kRedMax; ++s) {
    r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, nullptr);
    p.runFrame(ctx, r, w, dt, true, out, nullptr); pl.plan(w, out, cmd);
    for (const auto& d : out.detections3D)
      if (d.semantic == SemanticClass::SignalSign) {
        redSigDetM = std::max(redSigDetM, d.rangeM);
        sigDetDistM = std::max(sigDetDistM, d.rangeM);
      }
    EXPECT_FALSE(w.lastContact().occurred) << v << " " << m;
    if (cmd.behaviorState == "RED_LIGHT_HOLD" && w.ego().speed < kStopSpd) {
      stoppedRed = true; break;
    }
    w.stepAgents(dt); w.stepEgoAndPhysics(dt, cmd, {});
  }
  ASSERT_TRUE(stoppedRed) << v << " " << m;
  const EgoState& e = w.ego();
  const float bx = e.pos.x + std::sin(e.yaw) * (e.hz + e.centerOffsetForward);
  const float marginM = WorldEngine::kStopLineX - bx;
  EXPECT_LE(bx, WorldEngine::kStopLineX + kOverM) << v << " " << m;
  EXPECT_GE(marginM, -kOverM) << v << " " << m;
  EXPECT_LE(marginM, kMaxMargM) << v << " " << m;

  w.setForcedRedSignal(false);
  int gStep = -1, rStep = -1;
  for (int s = 0; s < kGreenMax; ++s) {
    r.renderAllCameras(w, cfg, static_cast<float>(kRedMax + s) * dt, nullptr);
    p.runFrame(ctx, r, w, dt, true, out, nullptr); pl.plan(w, out, cmd);
    for (const auto& d : out.detections3D)
      if (d.semantic == SemanticClass::SignalSign)
        sigDetDistM = std::max(sigDetDistM, d.rangeM);
    EXPECT_FALSE(w.lastContact().occurred) << v << " " << m;
    if (gStep < 0 && w.signalPhase() == SignalPhase::Green) gStep = s;
    if (gStep >= 0 && cmd.targetSpeed > kGoSpd && w.ego().speed > kGoSpd) {
      rStep = s; break;
    }
    w.stepAgents(dt); w.stepEgoAndPhysics(dt, cmd, {});
  }
  ASSERT_GE(gStep, 0) << v << " " << m;
  ASSERT_GE(rStep, gStep) << v << " " << m;
  const float greenRespS = static_cast<float>(rStep - gStep) * dt;
  const std::string pfx = std::string(v) + "_" + m + "_";
  testing::Test::RecordProperty(pfx + "stop_margin_m", fmtProp(marginM));
  testing::Test::RecordProperty(pfx + "red_sig_det_m", fmtProp(redSigDetM));
  testing::Test::RecordProperty(pfx + "sig_det_m", fmtProp(sigDetDistM));
  testing::Test::RecordProperty(pfx + "green_resp_s", fmtProp(greenRespS));
  if (std::strcmp(v, "v15") == 0) {
    EXPECT_GT(redSigDetM, kMinSigDistM) << pfx;
  }
  EXPECT_LE(greenRespS, kMaxGreenS) << pfx;
  p.destroy(); r.destroy();
}

TEST_F(AvMotLaneValidationTest,
       IdSwitchesMotaLaneAssocAndRoadIoUAcrossResAndPipelines) {
  runMotCase(s_ctx, "v15", "half", kStdF, false, false, "v15_half");
  runMotCase(s_ctx, "v15", "full", kStdF, false, false, "v15_full");
  runMotCase(s_ctx, "v15", "cybercab_binned", kStdF, false, false,
             "v15_cybercab_binned");
  runMotCase(s_ctx, "v15", "cybercab", kNatF, false, false, "v15_cybercab");
  runMotCase(s_ctx, "v14", "half", kStdF, false, false, "v14_half");
  runMotCase(s_ctx, "v14", "full", kStdF, false, false, "v14_full");
  runMotCase(s_ctx, "v14", "cybercab_binned", kStdF, false, false,
             "v14_cybercab_binned");
  runMotCase(s_ctx, "v15", "half", kStdF, true, false, "v15_night");
  runMotCase(s_ctx, "v15", "half", kStdF, false, true, "v15_soiled");
}

TEST_F(AvMotLaneValidationTest,
       SignalStopLineComplianceAndGreenTransitionLatency) {
  runSignalStopAndGreenCase(s_ctx, "v14", "half");
  runSignalStopAndGreenCase(s_ctx, "v14", "full");
  runSignalStopAndGreenCase(s_ctx, "v15", "half");
  runSignalStopAndGreenCase(s_ctx, "v15", "full");
}
}  // namespace
}  // namespace cybercab
