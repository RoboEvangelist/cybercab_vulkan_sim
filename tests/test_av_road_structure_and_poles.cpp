#include "AvLaneCorridorMetrics.hpp"
#include "AvValidationMetrics.hpp"
#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace cybercab {
namespace {
using AvRoadStructureTest = testutil::SharedGpuMetricsTest;
using testutil::evalGroundLaneIoU;
using testutil::fmtProp;
using testutil::isPointInOBB;
using testutil::LaneCorridorMetrics;
using SC = SemanticClass;

constexpr int kSegAF = 30, kSegBF = 48, kRoadF = 48, kNatRoadF = 24;
constexpr float kPoleGateM = 2.4f, kVoxPadM = 2.0f, kMinVoxConf = 0.45f;
constexpr float kMinGantrySpd = 4.0f, kGantryStartZ = 12.0f;
constexpr float kInitSpd = 8.5f, kLat0 = -1.9f, kOff = -350.0f;
constexpr double kMinAgnPrec = 0.65, kMinRoadIou = 0.80, kMaxEdgeErrM = 0.60;
constexpr double kMaxLatErrM = 0.35, kMinObsCells = 400.0, kMinRoadRec = 0.80;

SimConfig mkRoadCfg(const char* v, const char* m) {
  SimConfig c =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  c.pipelineVersion = v; c.initialResMode = m; return c;
}

void clearDynamicAgents(WorldEngine& w) {
  for (auto& vh : w.mutableVehicles()) {
    vh.pos = {kOff, 0.0f, kOff}; vh.speed = vh.vx = vh.vz = 0.0f;
  }
  for (auto& g : w.mutablePedestrians())
    g.x = g.ax = g.bx = g.targetX = g.z = g.az = g.bz = g.targetZ = kOff;
}

void runPolesAndGantryCase(VulkanContext& ctx, const char* v, const char* m) {
  const SimConfig cfg = mkRoadCfg(v, m);
  const float dt = 1.0f / cfg.controlHz;
  const Vec2 poleAnchors[] = {
    {6.5f, 37.8f}, {12.5f, -49.6f}, {-34.0f, 49.6f},
    {50.0f, 0.0f}, {62.0f, 0.0f},
  };
  int poleVox = 0, agnOk = 0, agnTot = 0, falseAeb = 0;
  float minGantrySpd = 999.0f;
  bool seenStaticGauss = !cfg.isV15();
  std::vector<OBB> dyn;
  auto accumVox = [&](const WorldEngine& w, const PerceptionFrameOutput& o) {
    w.collectDynamicGroundTruthOBBs(dyn);
    const auto& st = w.staticOBBs();
    for (const auto& vx : o.activeVoxels) {
      if (vx.sem == SC::SignalSign || vx.sem == SC::Static)
        for (const auto& pa : poleAnchors)
          if (std::hypot(vx.wx - pa.x, vx.wz - pa.y) <= kPoleGateM) {
            ++poleVox; break;
          }
      if (vx.confidence < kMinVoxConf) continue;
      ++agnTot; bool hit = false;
      for (const auto& g : dyn)
        if (isPointInOBB(g, vx.wx, vx.wz, kVoxPadM)) { hit = true; break; }
      if (!hit)
        for (const auto& b : st)
          if (isPointInOBB(b, vx.wx, vx.wz, kVoxPadM)) { hit = true; break; }
      if (hit) ++agnOk;
    }
    for (const auto& gs : o.v15Gaussians)
      if (gs.velYaw.w < 0.5f) { seenStaticGauss = true; break; }
  };
  auto runSeg = [&](WorldEngine& w, int frames, bool chkAeb, const char* sg) {
    CameraRigRenderer r; r.init(ctx, cfg, w);
    PerceptionPipeline p(cfg); p.init(ctx, r); TrajectoryPlanner pl(cfg);
    PerceptionFrameOutput out{}; PlannerCommand cmd{};
    for (int s = 0; s < frames; ++s) {
      r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, nullptr);
      p.runFrame(ctx, r, w, dt, true, out, nullptr); pl.plan(w, out, cmd);
      EXPECT_FALSE(w.lastContact().occurred) << v << " " << m << " " << sg;
      if (chkAeb) {
        if (cmd.aebActive) ++falseAeb;
        minGantrySpd = std::min(minGantrySpd, w.ego().speed);
      }
      if (s >= 2) accumVox(w, out);
      w.stepAgents(dt); w.stepEgoAndPhysics(dt, cmd, {});
    }
    p.destroy(); r.destroy();
  };
  { WorldEngine w(cfg); runSeg(w, kSegAF, false, "segA"); }
  { WorldEngine wg(cfg); clearDynamicAgents(wg);
    const float u = wg.findClosestLoopU({WorldEngine::kLoopHalfX, 0.0f,
                                         kGantryStartZ});
    EgoState& e = wg.mutableEgo(); e.loopProgressU = u;
    e.currentLateralOffset = kLat0; e.speed = kInitSpd;
    e.pos = wg.laneCurve(0).getPointAt(u);
    const Vec3 tn = wg.laneCurve(0).getTangentAt(u);
    e.yaw = std::atan2(tn.x, tn.z);
    runSeg(wg, kSegBF, true, "segB");
  }
  const double agnPrec = agnTot > 0 ? static_cast<double>(agnOk) / agnTot : 0.;
  const std::string pfx = std::string(v) + "_" + m + "_";
  testing::Test::RecordProperty(pfx + "pole_vox", poleVox);
  testing::Test::RecordProperty(pfx + "class_agn_prec", fmtProp(agnPrec));
  testing::Test::RecordProperty(pfx + "gantry_false_aeb", falseAeb);
  testing::Test::RecordProperty(pfx + "gantry_min_spd", fmtProp(minGantrySpd));
  EXPECT_GT(poleVox, 0) << pfx; EXPECT_GE(agnPrec, kMinAgnPrec) << pfx;
  EXPECT_EQ(falseAeb, 0) << pfx; EXPECT_GT(minGantrySpd, kMinGantrySpd) << pfx;
  EXPECT_TRUE(seenStaticGauss) << pfx;
}

void runRoadStructureCase(VulkanContext& ctx, const char* v, const char* m,
                          int frames) {
  const SimConfig cfg = mkRoadCfg(v, m);
  WorldEngine w(cfg); clearDynamicAgents(w);
  CameraRigRenderer r; r.init(ctx, cfg, w);
  PerceptionPipeline p(cfg); p.init(ctx, r); TrajectoryPlanner pl(cfg);
  const float dt = 1.0f / cfg.controlHz;
  const float tgtLat = cfg.isV15() ? -cfg.v15Policy.laneCenterAbsM : kLat0;
  PerceptionFrameOutput out{}; PlannerCommand cmd{}; LaneCorridorMetrics lane;
  std::vector<Vec4> s4(static_cast<size_t>(8 * cfg.featWidth *
                                           cfg.featHeight * 4));
  double latSq = 0.0, obsSum = 0.0; int nEv = 0, rTp = 0, rTot = 0;

  for (int s = 0; s < frames; ++s) {
    r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, nullptr);
    p.runFrame(ctx, r, w, dt, true, out, nullptr); pl.plan(w, out, cmd);
    EXPECT_FALSE(w.lastContact().occurred) << v << " " << m << " s=" << s;
    if (s >= 2) {
      const EgoState& e = w.ego();
      const float dl = e.currentLateralOffset - tgtLat;
      latSq += static_cast<double>(dl * dl);
      obsSum += out.bevSummary.observedCells; ++nEv;
      if (s == 10 || (frames >= 40 && s == 30)) {
        evalGroundLaneIoU(w, r, cfg, lane);
        p.backboneRtdetr().featS4SSBO().downloadVector(s4);
        const float sE = std::sin(e.yaw), cE = std::cos(e.yaw);
        const int fw = cfg.featWidth, fh = cfg.featHeight;
        const int cw = r.camWidth(), ch = r.camHeight();
        for (float fwd : {8.0f, 12.0f, 16.0f}) {
          for (float lat : {-1.2f, 0.0f, 1.2f}) {
            const Vec3 wp{e.pos.x + sE * fwd + cE * lat, 0.03f,
                          e.pos.z + cE * fwd - sE * lat};
            Vec2 px{}; float dep = 0.0f;
            if (r.projectWorldToPixel(0, wp, px, dep) &&
                px.x >= 0.0f && px.x < cw && px.y >= 0.0f && px.y < ch) {
              const int fx = std::clamp(int(px.x * fw / cw), 0, fw - 1);
              const int fy = std::clamp(int(px.y * fh / ch), 0, fh - 1);
              ++rTot;
              if (s4[static_cast<size_t>(fy * fw + fx) * 4u + 1u].w >= 0.45f)
                ++rTp;
            }
          }
        }
      }
    }
    w.stepAgents(dt); w.stepEgoAndPhysics(dt, cmd, {});
  }
  const double latErr = std::sqrt(latSq / nEv), obsMean = obsSum / nEv;
  const double roadRec = rTot > 0 ? static_cast<double>(rTp) / rTot : 0.0;
  const std::string pfx = std::string(v) + "_" + m + "_";
  testing::Test::RecordProperty(pfx + "lane_iou", fmtProp(lane.iou()));
  testing::Test::RecordProperty(pfx + "edge_err_m", fmtProp(lane.edgeErrM()));
  testing::Test::RecordProperty(pfx + "lane_center_err_m", fmtProp(latErr));
  testing::Test::RecordProperty(pfx + "bev_obs_cells", fmtProp(obsMean));
  testing::Test::RecordProperty(pfx + "road_head_rec", fmtProp(roadRec));
  EXPECT_GE(lane.iou(), kMinRoadIou) << pfx;
  EXPECT_LE(lane.edgeErrM(), kMaxEdgeErrM) << pfx;
  EXPECT_LE(latErr, kMaxLatErrM) << pfx;
  EXPECT_GE(obsMean, kMinObsCells) << pfx;
  EXPECT_GE(roadRec, kMinRoadRec) << pfx;
  p.destroy(); r.destroy();
}

TEST_F(AvRoadStructureTest,
       ClassAgnosticPolesBarriersAndGantryUnderpassSafety) {
  runPolesAndGantryCase(s_ctx, "v14", "half");
  runPolesAndGantryCase(s_ctx, "v15", "half");
  runPolesAndGantryCase(s_ctx, "v15", "full");
  runPolesAndGantryCase(s_ctx, "v15", "cybercab_binned");
}

TEST_F(AvRoadStructureTest,
       RoadStructureObservabilityAndLaneCenteringAcrossPresets) {
  runRoadStructureCase(s_ctx, "v14", "half", kRoadF);
  runRoadStructureCase(s_ctx, "v15", "half", kRoadF);
  runRoadStructureCase(s_ctx, "v15", "full", kRoadF);
  runRoadStructureCase(s_ctx, "v15", "cybercab_binned", kRoadF);
  runRoadStructureCase(s_ctx, "v15", "cybercab", kNatRoadF);
}
}  // namespace
}  // namespace cybercab
