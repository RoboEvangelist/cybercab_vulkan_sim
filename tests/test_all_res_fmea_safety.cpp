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
using AllResFmeaTest = testutil::SharedGpuMetricsTest;
using SC = SemanticClass; using PO = PerceptionFrameOutput;
using PC = PlannerCommand;
constexpr int kClnF = 3, kSoilF = 4, kMrmF = 24, kEmpF = 5;
constexpr int kBrkStd = 42, kBrkNat = 30;
constexpr float kOff = -350.f, kMinSp = 0.40f, kMaxSw = 0.55f, kMinP = 0.01f;
constexpr float kNearM = 7.f, kUStart = 0.215f, kLatM = -1.9f, kSpd0 = 9.0f;
constexpr float kGapM = 15.5f, kBrkT = 6.f;
constexpr const char* kModes[] = {
  "half", "full", "cybercab_binned", "cybercab"};
enum class FmeaSetup { kNom, kEmpty, kBrake };

struct FmeaRig {
  VulkanContext& ctx; FmeaSetup setup_; SimConfig cfg; WorldEngine w;
  CameraRigRenderer r; PerceptionPipeline p; TrajectoryPlanner pl;
  FmeaRig(VulkanContext& c, const char* v, const char* m, FmeaSetup st,
          bool s0, bool sA)
    : ctx(c), setup_(st), cfg(mk(v, m, s0, sA)), w(cfg), p(cfg), pl(cfg) {
    if (st == FmeaSetup::kEmpty) clr(0u);
    else if (st == FmeaSetup::kBrake) { clr(1u); setBrk(); }
    r.init(ctx, cfg, w); p.init(ctx, r);
  }
  ~FmeaRig() { p.destroy(); r.destroy(); }
  static SimConfig mk(const char* v, const char* m, bool s0, bool sA) {
    SimConfig c =
      SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    c.pipelineVersion = v; c.initialResMode = m;
    if (s0 || sA) c.cameras[0].soiled = true;
    if (sA) c.cameras[1].soiled = c.cameras[2].soiled = true;
    return c;
  }
  void clr(size_t k) {
    for (size_t i = k; i < w.mutableVehicles().size(); ++i) {
      auto& v = w.mutableVehicles()[i];
      v.pos = {kOff, 0.f, kOff}; v.speed = v.vx = v.vz = 0.f;
    }
    for (auto& g : w.mutablePedestrians())
      g.x = g.ax = g.bx = g.targetX = g.z = g.az = g.bz = g.targetZ = kOff;
  }
  void setBrk() {
    const float len = w.centerline().getLength();
    EgoState& e = w.mutableEgo(); e.loopProgressU = kUStart;
    e.currentLateralOffset = kLatM; e.speed = kSpd0;
    e.pos = w.laneCurve(0).getPointAt(kUStart);
    const Vec3 et = w.laneCurve(0).getTangentAt(kUStart);
    e.yaw = std::atan2(et.x, et.z);
    auto& tr = w.mutableVehicles().at(0); tr.lane = 0; tr.speed = 0.f;
    tr.hardBrakeTimer = kBrkT; tr.vx = tr.vz = 0.f;
    tr.u = std::fmod(kUStart + kGapM / len, 1.f);
    tr.pos = w.laneCurve(0).getPointAt(tr.u);
    const Vec3 tt = w.laneCurve(0).getTangentAt(tr.u);
    tr.yaw = std::atan2(tt.x, tt.z);
  }
  void step(int s, PO& o, PC& cmd) {
    const float dt = 1.f / cfg.controlHz;
    r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, nullptr);
    p.runFrame(ctx, r, w, dt, true, o, nullptr); pl.plan(w, o, cmd);
    if (setup_ != FmeaSetup::kEmpty) { w.stepAgents(dt); }
    w.stepEgoAndPhysics(dt, cmd, {}); EXPECT_FALSE(w.lastContact().occurred);
  }
};

TEST_F(AllResFmeaTest, FmeaSoilingRedundancyAndMrmAcrossAllFourResolutions) {
  for (const char* m : kModes) {
    PO out{}; PC cmd{}; float clnV = 0.f, clnF = 0.f;
    { FmeaRig cln(s_ctx, "v15", m, FmeaSetup::kNom, false, false);
      for (int s = 0; s < kClnF; ++s) cln.step(s, out, cmd);
      clnV = out.soilCameras[0].meanEpistemicVar;
      clnF = out.forwardEpistemicSigma;
    }
    for (const char* v : {"v14", "v15"}) {
      FmeaRig s0(s_ctx, v, m, FmeaSetup::kNom, true, false);
      for (int s = 0; s < kSoilF; ++s) s0.step(s, out, cmd);
      EXPECT_GT(out.soilCameras[0].meanSoilProb, kMinSp) << v << m;
      EXPECT_LT(out.soilCameras[0].tokenWeight, kMaxSw) << v << m;
      EXPECT_GT(out.soilCameras[0].meanEpistemicVar, clnV) << v << m;
      EXPECT_FALSE(out.detections3D.empty()); EXPECT_FALSE(out.tracks.empty());
    }
    FmeaRig sAll(s_ctx, "v15", m, FmeaSetup::kNom, true, true);
    bool seenMrm = false; float maxFwd = 0.f;
    for (int s = 0; s < kMrmF; ++s) {
      sAll.step(s, out, cmd);
      maxFwd = std::max(maxFwd, out.forwardEpistemicSigma);
      if (cmd.behaviorState == "SENSOR_SOILED_MRM") seenMrm = true;
    }
    EXPECT_GT(maxFwd, clnF) << m; EXPECT_TRUE(seenMrm) << m;
  }
}

TEST_F(AllResFmeaTest, FmeaEmptyRoadZeroHallucinationsAndBrakingNoSplitAllRes) {
  for (const char* m : kModes) {
    for (const char* v : {"v14", "v15"}) {
      FmeaRig emp(s_ctx, v, m, FmeaSetup::kEmpty, false, false);
      PO out{}; PC cmd{};
      for (int s = 0; s < kEmpF; ++s) {
        emp.step(s, out, cmd); EXPECT_FALSE(cmd.aebActive) << v << " " << m;
        EXPECT_EQ(out.eval.ghostVehDetections, 0)
          << v << " " << m << " s=" << s;
        EXPECT_EQ(out.eval.ghostPedDetections, 0)
          << v << " " << m << " s=" << s;
        for (const auto& vx : out.activeVoxels)
          EXPECT_TRUE(vx.sem != SC::Vehicle && vx.sem != SC::Pedestrian)
            << v << " " << m << " s=" << s;
        for (const auto& g : out.v15Gaussians) {
          const int sem = static_cast<int>(std::lround(g.scaleSem.w));
          EXPECT_TRUE(sem != static_cast<int>(SC::Vehicle) &&
                      sem != static_cast<int>(SC::Pedestrian))
            << v << " " << m << " s=" << s;
        }
      }
    }
    const bool nat = (std::strcmp(m, "cybercab") == 0);
    const char* vs[] = {"v15", nat ? "v15" : "v14"};
    for (int vi = 0; vi < (nat ? 1 : 2); ++vi) {
      if (vi == 1 && std::strcmp(m, "half") == 0) continue;
      FmeaRig brk(s_ctx, vs[vi], m, FmeaSetup::kBrake, false, false);
      PO out{}; PC cmd{}; bool pOk = false;
      for (int s = 0; s < (nat ? kBrkNat : kBrkStd); ++s) {
        brk.step(s, out, cmd); if (brk.w.ego().pitch > kMinP) pOk = true;
        EXPECT_EQ(out.eval.misclassifiedTrucks, 0);
        const auto& tr = brk.w.vehicles().at(0); int nr = 0;
        for (const auto& d : out.detections3D)
          if (d.semantic != SC::SignalSign &&
              std::hypot(d.worldPos.x - tr.pos.x,
                         d.worldPos.z - tr.pos.z) < kNearM) ++nr;
        EXPECT_LE(nr, 1) << vs[vi] << " " << m << " s=" << s;
      }
      EXPECT_TRUE(pOk) << vs[vi] << " " << m;
    }
  }
}
}  // namespace
}  // namespace cybercab
