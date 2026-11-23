#include "MetricsEvalHelpers.hpp"
#include "perception/ByteTrack.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace cybercab {
namespace {
using AllResFmeaEnvTest = testutil::SharedGpuMetricsTest;
using testutil::fmtProp; using SC = SemanticClass;
using PO = PerceptionFrameOutput; using PC = PlannerCommand;
using CS = v15::CameraSelfCalib;
constexpr int kCams = 8, kNgStd = 28, kNgNat = 16, kHitN = 6;
constexpr float kLatM = -1.9f, kRollK = -1.3f, kPitchK = 1.5f;
constexpr float kU0 = 0.05f, kDu = 0.025f, kTolH = 0.35f, kTolHi = 0.25f;
constexpr float kLeak = 0.01f, kX0 = 22.f, kVx = 8.f, kSc = 0.85f, kPad = 0.2f;
const Vec3 kVehHe{0.96f, 0.72f, 2.18f};
constexpr const char* kModes[] = {
  "half", "full", "cybercab_binned", "cybercab"};

struct EnvRig {
  VulkanContext& ctx; SimConfig cfg; WorldEngine w; CameraRigRenderer r;
  PerceptionPipeline p; TrajectoryPlanner pl;
  EnvRig(VulkanContext& c, const char* v, const char* m, bool night)
    : ctx(c), cfg(mk(v, m, night)), w(cfg), p(cfg), pl(cfg) {
    if (night) w.triggerJaywalkerCutIn();
    r.init(ctx, cfg, w); p.init(ctx, r);
  }
  ~EnvRig() { p.destroy(); r.destroy(); }
  static SimConfig mk(const char* v, const char* m, bool night) {
    SimConfig c =
      SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    c.pipelineVersion = v; c.initialResMode = m;
    if (night) { c.lightingMode = "night_glare"; c.photonRccbMosaic = true; }
    return c;
  }
  void step(int s, PO& o, PC& cmd) {
    const float dt = 1.f / cfg.controlHz;
    r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, nullptr);
    p.runFrame(ctx, r, w, dt, true, o, nullptr); pl.plan(w, o, cmd);
    w.stepAgents(dt); w.stepEgoAndPhysics(dt, cmd, {});
    EXPECT_FALSE(w.lastContact().occurred);
  }
};

TEST_F(AllResFmeaEnvTest,
       FmeaNightGlareSelfCalibResizeSyncAndDropoutAllRes) {
  for (const char* m : kModes) {
    const bool nat = (std::strcmp(m, "cybercab") == 0);
    EnvRig ng(s_ctx, "v15", m, true);
    PO out{}; PC cmd{};
    for (int s = 0; s < (nat ? kNgNat : kNgStd); ++s) {
      ng.step(s, out, cmd);
      if (s == 0) {
        const auto& buf = ng.r.cameraLinearSSBO();
        ASSERT_GE(buf.size(), sizeof(Vec4));
        Vec4 firstPx{}, lastPx{};
        buf.download(&firstPx, sizeof(Vec4), 0u);
        buf.download(&lastPx, sizeof(Vec4), buf.size() - sizeof(Vec4));
        EXPECT_TRUE(std::isfinite(firstPx.x) && std::isfinite(lastPx.z));
      }
      EXPECT_FALSE(out.detections3D.empty());
      EXPECT_FALSE(out.activeVoxels.empty());
      EXPECT_FALSE(out.v15Gaussians.empty());
    }
    EnvRig cal(s_ctx, "v15", m, false);
    cal.cfg.cameraKnockRpyDeg = {kRollK, kPitchK, 0.f};
    const auto& lp = cal.w.centerline();
    const float dt = 1.f / cal.cfg.controlHz;
    for (int f = 0; f < CS::kWindowFrames; ++f) {
      const float u = std::fmod(kU0 + kDu * static_cast<float>(f), 1.f);
      EgoState& e = cal.w.mutableEgo(); const Vec3 tn = lp.getTangentAt(u);
      e.pos = lp.getPointAt(u) + Vec3{tn.z, 0.f, -tn.x} * kLatM;
      e.yaw = std::atan2(tn.x, tn.z); e.pitch = e.roll = 0.f;
      cal.r.renderAllCameras(cal.w, cal.cfg, static_cast<float>(f) * dt,
                             nullptr);
      cal.p.runFrame(s_ctx, cal.r, cal.w, dt, true, out, nullptr);
    }
    const float tol = (std::strcmp(m, "half") == 0) ? kTolH : kTolHi;
    RecordProperty(std::string(m) + "_calib_pitch_deg",
                   fmtProp(out.v15.calibPitchDeg));
    RecordProperty(std::string(m) + "_calib_roll_deg",
                   fmtProp(out.v15.calibRollDeg));
    EXPECT_NEAR(out.v15.calibPitchDeg, kPitchK, tol) << m;
    EXPECT_NEAR(out.v15.calibRollDeg, kRollK, tol) << m;
  }
  for (const char* v : {"v14", "v15"}) {
    EnvRig rz(s_ctx, v, "half", false);
    PO out{}; PC cmd{};
    std::vector<Vec4> s4(static_cast<size_t>(
      kCams * rz.cfg.featWidth * rz.cfg.featHeight * 4));
    rz.step(0, out, cmd);
    for (const char* m : {"full", "cybercab_binned", "cybercab", "half"}) {
      rz.cfg.initialResMode = m; const auto t = rz.cfg.resolveInitialRes();
      rz.r.resize(t.width, t.height); rz.p.onCameraResized(rz.r);
      EXPECT_EQ(rz.r.camWidth(), t.width);
      EXPECT_EQ(rz.r.camHeight(), t.height);
      rz.step(1, out, cmd);
      rz.p.backboneRtdetr().featS4SSBO().downloadVector(s4);
      float mot = 0.f;
      for (size_t i = 2u; i < s4.size(); i += 4u) mot = std::max(mot, s4[i].z);
      EXPECT_LT(mot, kLeak) << v << m; rz.step(2, out, cmd);
    }
  }
  const SimConfig cfg = SimConfig::defaultConfig(); ByteTracker bt(cfg);
  TrajectoryPlanner pl(cfg); WorldEngine w(cfg); EgoState ego = w.ego();
  PO out{}; PC cmd{}; const float dt = 1.f / cfg.controlHz;
  for (int s = 0; s < kHitN; ++s) {
    Detection3D d{}; d.semantic = SC::Vehicle; d.score = kSc;
    d.halfExtents = kVehHe; const float x = kX0 + kVx * s * dt;
    d.worldPos = {x, kVehHe.y, 0.f}; d.egoPos = {x, 0.f, kVehHe.y};
    d.rangeM = x; bt.update(dt, ego, {d});
  }
  bt.collectActiveTracks(out.tracks); ASSERT_EQ(out.tracks.size(), 1u);
  float pA = 0.f, pS = out.tracks[0].posSigmaM;
  for (float t = 0.f; t <= cfg.btMaxTimeLostSec + kPad; t += dt) {
    bt.update(dt, ego, {}); bt.collectActiveTracks(out.tracks);
    pl.plan(w, out, cmd); EXPECT_TRUE(std::isfinite(cmd.targetSpeed));
    if (!out.tracks.empty()) {
      EXPECT_GT(out.tracks[0].timeSinceUpdateSec, pA);
      EXPECT_GE(out.tracks[0].posSigmaM, pS);
      pA = out.tracks[0].timeSinceUpdateSec; pS = out.tracks[0].posSigmaM;
    }
  }
  EXPECT_TRUE(out.tracks.empty());
}
}  // namespace
}  // namespace cybercab
