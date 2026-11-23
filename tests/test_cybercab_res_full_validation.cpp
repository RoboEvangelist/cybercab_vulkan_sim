#include "MetricsEvalHelpers.hpp"
#include "app/V15HudOverlay.hpp"
#include "core/GpuProfiler.hpp"
#include "core/TraceLogger.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/v15/PatchTokenizer.hpp"
#include "perception/v15/VirtualCameraRectifier.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <json/json.h>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace cybercab {
namespace {
using CybercabResFullValidationTest = testutil::SharedGpuMetricsTest;
using testutil::fmtProp;
using Clk = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>; using SC = SemanticClass;
constexpr int kProfSlots = 64, kWarm = 4, kLatF = 6, kBinF = 36, kNatF = 10;
constexpr int kMinObs = 100, kMinHits = 4, kMinCor = 2, kMain = 0, kNar = 1;
constexpr int kWide = 2; constexpr float kNearM = 20.0f, kSlackM = 0.5f;
constexpr float kCalTol = 0.25f, kPitch = 1.5f, kRoll = -1.3f, kLaneM = -1.9f;
constexpr float kU0 = 0.05f, kDu = 0.025f, kVoxTol = 0.05f, kPadM = 1.80f;
constexpr float kEvalR = 42.0f, kBndR = 48.0f, kPrG = 5.0f, kPlkTol = 1e-4f;
constexpr double kV14BinMs = 20.0, kV14NatMs = 45.0, kV15GpuMs = 27.8;
constexpr double kV15WalMs = 35.0, kMinRec = 0.50, kMinPrec = 0.65;
constexpr double kMinAp = 0.40, kMaxRmse = 2.8, kMinVoxP = 0.50;
constexpr float kTeslaHz = 36.0f, kDetR = 44.0f;

struct SimRig {
  VulkanContext& ctx; SimConfig cfg; WorldEngine w; CameraRigRenderer r;
  PerceptionPipeline p; TrajectoryPlanner pl;
  SimRig(VulkanContext& c, const char* v, const char* m)
    : ctx(c), cfg(load(v, m)), w(cfg), p(cfg), pl(cfg) {
    r.init(ctx, cfg, w); p.init(ctx, r); }
  ~SimRig() { p.destroy(); r.destroy(); }
  static SimConfig load(const char* v, const char* m) {
    SimConfig c =
      SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    c.pipelineVersion = v; c.initialResMode = m; return c;
  }
  void sense(int s, GpuProfiler* pr, PerceptionFrameOutput& o,
             PlannerCommand& cmd) {
    const float dt = 1.0f / cfg.controlHz;
    r.renderAllCameras(w, cfg, static_cast<float>(s) * dt, pr);
    p.runFrame(ctx, r, w, dt, true, o, pr); pl.plan(w, o, cmd);
  }
  void adv(const PlannerCommand& cmd) {
    const float dt = 1.0f / cfg.controlHz; w.stepAgents(dt);
    w.stepEgoAndPhysics(dt, cmd, {}); EXPECT_FALSE(w.lastContact().occurred);
  }
  void bench(GpuProfiler* pr, PerceptionFrameOutput& o, PlannerCommand& cmd,
             double& meanW, double& meanG, bool coldChk) {
    double wS = 0.0, gS = 0.0;
    for (int s = 0; s < kWarm + kLatF; ++s) {
      const auto t0 = Clk::now(); sense(s, pr, o, cmd); adv(cmd);
      if (coldChk && s == 0) {
        EXPECT_FALSE(o.activeVoxels.empty());
        EXPECT_GT(o.bevSummary.observedCells, kMinObs);
        EXPECT_EQ(o.gpuStageMs.count("1c_v15_freq_split"), 0u);
      } else if (s >= kWarm) {
        wS += Ms(Clk::now() - t0).count(); gS += o.eval.totalPipelineMs;
      }
    }
    meanW = wS / kLatF; meanG = gS / kLatF;
  }
};

TEST_F(CybercabResFullValidationTest, V14AndV15AllModulesAndCalibPresets) {
  for (const char* m : {"cybercab_binned", "cybercab"}) {
    SimRig sr(s_ctx, "v14", m);
    PerceptionFrameOutput out{}; PlannerCommand cmd{}; double mW = 0, mG = 0;
    EXPECT_FALSE(sr.r.linearHdrEnabled()); EXPECT_EQ(sr.p.v15(), nullptr);
    EXPECT_EQ(sr.r.cameraLinearSSBO().size(), 16u);
    sr.bench(nullptr, out, cmd, mW, mG, true);
    RecordProperty(std::string("v14_") + m + "_wall_ms", fmtProp(mW));
    EXPECT_LT(mW, std::string(m) == "cybercab_binned" ? kV14BinMs : kV14NatMs);
    EXPECT_FALSE(out.detections3D.empty()); EXPECT_FALSE(out.tracks.empty());
    EXPECT_FALSE(out.worldGhosts.empty());
  }
  SimRig sr(s_ctx, "v15", "cybercab_binned");
  GpuProfiler prof; prof.init(s_ctx, kProfSlots);
  PerceptionFrameOutput out{}; PlannerCommand cmd{}; double mW = 0, mG = 0;
  sr.bench(&prof, out, cmd, mW, mG, false);
  RecordProperty("v15_binned_gpu_ms", fmtProp(mG));
  RecordProperty("v15_binned_wall_ms", fmtProp(mW));
  EXPECT_LT(mG, kV15GpuMs); EXPECT_LT(mW, kV15WalMs);
  EXPECT_GT(out.bevSummary.observedCells, kMinObs);
  EXPECT_FALSE(out.worldGhosts.empty()); EXPECT_FALSE(out.v15Gaussians.empty());
  EXPECT_EQ(out.v15RealityRgba.size(), v15_hud::kRealityExpectedBytes);
  auto* v15p = sr.p.v15(); ASSERT_NE(v15p, nullptr);
  using R = v15::VirtualCameraRectifier; using T = v15::PatchTokenizer;
  std::vector<Vec4> vPx(R::kNumViews * R::kVirtW * R::kVirtH);
  v15p->rectifier().virtLinearSSBO().downloadVector(vPx);
  EXPECT_TRUE(std::isfinite(vPx[0].x) && std::isfinite(vPx.back().z));
  const auto& tok = v15p->patchTokenizer();
  EXPECT_EQ(tok.validSlots(), static_cast<uint32_t>(T::kHistoryFrames));
  constexpr size_t nF = T::kTokensPerFrame;
  std::vector<v15::PatchTokenGpu> ring(T::kHistoryFrames * nF);
  tok.ringSSBO().downloadVector(ring);
  const size_t lat = tok.latestWrittenSlot();
  const auto& tk = ring[lat * nF];
  const Vec3 dV{tk.pluckerD.x, tk.pluckerD.y, tk.pluckerD.z};
  const Vec3 mV{tk.pluckerM.x, tk.pluckerM.y, tk.pluckerM.z};
  EXPECT_NEAR(dV.norm(), 1.0f, kPlkTol);
  EXPECT_LT(std::abs(dV.dot(mV)), kPlkTol);
  EXPECT_GT(ring[((lat + 1u) % T::kHistoryFrames) * nF].pluckerM.w, 0.0f);
  const auto trP = std::filesystem::temp_directory_path() / "v15_bin.jsonl";
  { TraceLogger tr; ASSERT_TRUE(tr.open(trP.string()));
    tr.logFrame(0, 0.0f, sr.cfg, sr.w.ego(), cmd, out, {}); }
  std::ifstream ifs(trP); Json::CharReaderBuilder rb; Json::Value jv;
  std::string err; EXPECT_TRUE(Json::parseFromStream(rb, ifs, &jv, &err));
  EXPECT_TRUE(jv["v15"]["active"].asBool()); std::filesystem::remove(trP);
  sr.cfg.cameraKnockRpyDeg = {kRoll, kPitch, 0.0f}; sr.p.reset();
  const auto& lp = sr.w.centerline();
  const float dt = 1.0f / sr.cfg.controlHz;
  for (int f = 0; f < v15::CameraSelfCalib::kWindowFrames; ++f) {
    const float u = std::fmod(kU0 + kDu * static_cast<float>(f), 1.0f);
    EgoState& e = sr.w.mutableEgo(); const Vec3 tn = lp.getTangentAt(u);
    e.pos = lp.getPointAt(u) + Vec3{tn.z, 0.0f, -tn.x} * kLaneM;
    e.yaw = std::atan2(tn.x, tn.z); e.pitch = e.roll = 0.0f;
    sr.r.renderAllCameras(sr.w, sr.cfg, static_cast<float>(f) * dt, &prof);
    sr.p.runFrame(s_ctx, sr.r, sr.w, dt, true, out, &prof);
  }
  RecordProperty("binned_calib_pitch_deg", fmtProp(out.v15.calibPitchDeg));
  RecordProperty("binned_calib_roll_deg", fmtProp(out.v15.calibRollDeg));
  EXPECT_NEAR(out.v15.calibPitchDeg, kPitch, kCalTol);
  EXPECT_NEAR(out.v15.calibRollDeg, kRoll, kCalTol); prof.destroy();
}

void runResPrEval(VulkanContext& ctx, const char* mode, int evalF) {
  SimRig sr(ctx, "v15", mode); const SimConfig& cfg = sr.cfg;
  EXPECT_FLOAT_EQ(cfg.controlHz, kTeslaHz);
  EXPECT_GT(cfg.cameras[kNar].maxRangeM, cfg.cameras[kMain].maxRangeM);
  EXPECT_GT(cfg.cameras[kMain].maxRangeM, cfg.cameras[kWide].maxRangeM);
  PerceptionFrameOutput out{}; PlannerCommand cmd{}; testutil::PrAccum vehAcc{};
  std::vector<OBB> gt, reqV, bndV;
  double rec = 0.0, prec = 0.0, rmse = 0.0, ap = 0.0, f1 = 0.0;
  float maxErr = 0.0f, maxAge = 0.0f, maxRng = 0.0f; size_t dTot = 0, dOk = 0;
  int nEv = 0, misT = 0, nearD = 0, midFarD = 0, maxH = 0, maxCd = 0, maxCt = 0;
  for (int s = 0; s < kWarm + evalF; ++s) {
    sr.sense(s, nullptr, out, cmd);
    if (s < kWarm) { sr.adv(cmd); continue; }
    rec += out.eval.detRecall; prec += out.eval.detPrecision;
    rmse += out.eval.trackPosRmseM; misT += out.eval.misclassifiedTrucks;
    for (const auto& d : out.detections3D) {
      const size_t ci = static_cast<size_t>(d.camIndex);
      ASSERT_TRUE(d.camIndex >= 0 && ci < cfg.cameras.size());
      const float lm = std::min(cfg.rtdetrMaxRangeM, cfg.cameras[ci].maxRangeM);
      EXPECT_GT(d.rangeM, 0.0f); EXPECT_LE(d.rangeM, lm + kSlackM);
      maxRng = std::max(maxRng, d.rangeM);
      if (d.rangeM < kNearM) ++nearD; else ++midFarD;
      maxCd = std::max(maxCd, d.corroboratingCamCount);
    }
    for (const auto& t : out.tracks) {
      maxH = std::max(maxH, t.hits); EXPECT_TRUE(std::isfinite(t.posSigmaM));
      maxAge = std::max(maxAge, t.timeSinceUpdateSec);
      maxCt = std::max(maxCt, t.corroboratingCamCount); EXPECT_GT(t.score, 0.f);
    }
    sr.w.collectDynamicGroundTruthOBBs(gt); reqV.clear(); bndV.clear();
    const Mat4 wIso = BEVFormer::worldFromIsoEgo(sr.w.ego());
    for (const auto& v : out.activeVoxels) {
      const Vec3 ex = wIso.transformPoint({v.egoX, v.egoY, v.wy});
      maxErr = std::max(maxErr, std::hypot(v.wx - ex.x, v.wz - ex.z));
      if (v.sem != SC::Vehicle && v.sem != SC::Pedestrian) continue;
      ++dTot; for (const auto& b : gt)
        if (testutil::isPointInOBB(b, v.wx, v.wz, kPadM)) { ++dOk; break; }
    }
    const EgoState& e = sr.w.ego();
    for (const auto& g : gt) {
      if (g.semantic != SC::Vehicle) continue;
      const float d = std::hypot(g.cx - e.pos.x, g.cz - e.pos.z);
      if (d <= kBndR) (d <= kEvalR ? reqV : bndV).push_back(g);
    }
    testutil::accumulatePrCls(SC::Vehicle, out.detections3D, reqV, bndV, kPrG,
      kDetR, vehAcc); ++nEv; sr.adv(cmd);
  }
  testutil::summarizePrCurve(vehAcc, ap, f1);
  const double voxP = dTot > 0u ? static_cast<double>(dOk) / dTot : 0.0;
  const std::string tag = std::string(mode) + "_";
  auto rp = [&](const char* k, const std::string& v) {
    testing::Test::RecordProperty(tag + k, v); };
  rp("recall", fmtProp(rec / nEv)); rp("prec", fmtProp(prec / nEv));
  rp("ap", fmtProp(ap)); rp("rmse", fmtProp(rmse / nEv));
  rp("max_range_m", fmtProp(maxRng)); rp("max_corrob", std::to_string(maxCd));
  EXPECT_GT(rec / nEv, kMinRec); EXPECT_GT(prec / nEv, kMinPrec);
  EXPECT_GT(ap, kMinAp); EXPECT_LT(rmse / nEv, kMaxRmse); EXPECT_EQ(misT, 0);
  EXPECT_LT(maxErr, kVoxTol); EXPECT_GT(voxP, kMinVoxP); EXPECT_GT(nearD, 0);
  EXPECT_GT(midFarD, 0); EXPECT_GE(maxH, kMinHits); EXPECT_GE(maxCd, kMinCor);
  EXPECT_LE(maxAge, cfg.btMaxTimeLostSec); EXPECT_GE(maxCt, kMinCor);
}

TEST_F(CybercabResFullValidationTest, CybercabResPrVoxelAgeAndCorroboration) {
  runResPrEval(s_ctx, "cybercab_binned", kBinF);
  runResPrEval(s_ctx, "cybercab", kNatF);
}
}  // namespace
}  // namespace cybercab
