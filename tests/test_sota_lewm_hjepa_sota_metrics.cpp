#include "MetricsEvalHelpers.hpp"
#include "app/V15HudOverlay.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/sota/LeWmLatentModel.hpp"
#include "perception/sota/LeWmTypes.hpp"
#include "planning/TrajectoryPlanner.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace cybercab::sota {
namespace {

using LeWmHJepaSotaMetricsTest = testutil::SharedGpuMetricsTest;

constexpr float kEvalRangeM = 42.0f;
constexpr float kBoundRangeM = 48.0f;
constexpr float kPrGateM = 5.0f;
constexpr float kOnRoadLimitM = 3.65f;

float cosSim16(const LatentVec16& a, const LatentVec16& b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < kLeWmLatentDim; ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  const double den = std::sqrt(na) * std::sqrt(nb);
  return (den > 1e-8) ? static_cast<float>(dot / den) : 1.0f;
}

float mse16(const LatentVec16& a, const LatentVec16& b) {
  double s = 0.0;
  for (size_t i = 0; i < kLeWmLatentDim; ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    s += d * d;
  }
  return static_cast<float>(s / static_cast<double>(kLeWmLatentDim));
}

struct ModeEvalCfg {
  const char* label;
  const char* spatial;
  const char* wm;
};

struct ScenarioPdms {
  double nc = 1.0, dac = 1.0, ep = 1.0, ttc = 1.0, comfort = 1.0, pdms = 1.0;
  double distM = 0.0, meanJerk = 0.0;
  double pred1Mse = 0.0, pred1Cos = 1.0, roll4Mse = 0.0, roll4Cos = 1.0;
  double alignRate = 1.0, peakSurprise = 0.0, steadySurprise = 0.0;
};

ScenarioPdms runScenarioEval(VulkanContext& ctx, const ModeEvalCfg& mc,
                             const std::string& pipeVer, const std::string& scen,
                             int steps, double refNomDistM,
                             double* outRecall = nullptr, double* outPrec = nullptr,
                             double* outVehAp = nullptr, double* outRmse = nullptr,
                             int* outGhosts = nullptr) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = pipeVer;
  cfg.spatialRepMode = mc.spatial;
  cfg.worldModelMode = mc.wm;
  if (scen == "night_glare_jaywalk") {
    cfg.lightingMode = "night_glare";
    cfg.photonRccbMosaic = true;
  }
  WorldEngine w(cfg);
  if (scen == "jaywalk" || scen == "night_glare_jaywalk") w.triggerJaywalkerCutIn();
  else if (scen == "occluded") w.triggerOccludedDartOut();
  else if (scen == "hardbrake") w.triggerLeadHardBrake();

  CameraRigRenderer rig; rig.init(ctx, cfg, w);
  PerceptionPipeline pipe(cfg); pipe.init(ctx, rig);
  TrajectoryPlanner planner(cfg);
  PerceptionFrameOutput out{};
  PlannerCommand cmd{};
  const float dt = 1.0f / cfg.controlHz;

  testutil::PrAccum vehAcc{};
  std::vector<OBB> gtAll, reqV, boundV;
  std::vector<LatentVec16> obsZ;
  std::vector<LeWmAction> obsAct;
  double recSum = 0.0, precSum = 0.0, rmseSum = 0.0;
  int evalN = 0, ghostSum = 0, totalTtcSafe = 0, ttcFrames = 0;
  int alignHits = 0, alignTot = 0;
  double jerkSum = 0.0, maxAcc = 0.0, maxLatAbs = 0.0;
  double distM = 0.0, p1MseSum = 0.0, p1CosSum = 0.0, r4MseSum = 0.0, r4CosSum = 0.0;
  int p1Cnt = 0, r4Cnt = 0;
  bool col = false;
  Vec3 prevPos = w.ego().pos;
  float prevAcc = w.ego().accel;
  ScenarioPdms res{};

  for (int s = 0; s < steps; ++s) {
    rig.renderAllCameras(w, cfg, static_cast<float>(s) * dt);
    pipe.runFrame(ctx, rig, w, dt, true, out);
    planner.plan(w, out, cmd);
    const EgoState& egoBefore = w.ego();
    maxLatAbs = std::max(maxLatAbs, static_cast<double>(std::abs(egoBefore.currentLateralOffset)));

    if (pipe.worldModelSota()) {
      const LatentVec16 zCurr = pipe.worldModelSota()->lastLatent();
      if (s >= 2) res.peakSurprise = std::max(res.peakSurprise, static_cast<double>(out.worldModelSota.predSurprise));
      if (s >= 4) res.steadySurprise += static_cast<double>(out.worldModelSota.predSurprise);
      const float curLat = egoBefore.currentLateralOffset;
      const float recLatV = out.worldModelSota.recommendedLatM - curLat;
      const LeWmAction actCurr{clampf(egoBefore.accel, -8.5f, 4.0f), clampf(recLatV * 0.5f, -1.5f, 1.5f)};
      if (!obsZ.empty()) {
        LatentVec16 zPred1{};
        if (pipe.worldModelSota()->predictNext(obsZ.back(), obsAct.back(), dt, zPred1)) {
          p1MseSum += mse16(zPred1, zCurr);
          p1CosSum += cosSim16(zPred1, zCurr);
          ++p1Cnt;
        }
      }
      if (obsZ.size() >= 4u) {
        LatentVec16 zRoll = obsZ[obsZ.size() - 4u];
        bool ok = true;
        for (size_t k = obsZ.size() - 4u; k < obsZ.size(); ++k) {
          LatentVec16 nxt{};
          if (!pipe.worldModelSota()->predictNext(zRoll, obsAct[k], dt, nxt)) { ok = false; break; }
          zRoll = nxt;
        }
        if (ok) {
          r4MseSum += mse16(zRoll, zCurr);
          r4CosSum += cosSim16(zRoll, zCurr);
          ++r4Cnt;
        }
      }
      obsZ.push_back(zCurr);
      obsAct.push_back(actCurr);

      if (out.worldModelSota.hierarchical && !cmd.optimalTrajectory.empty()) {
        const auto sub = static_cast<MacroSubgoal>(out.worldModelSota.macroSubgoalCode);
        const Vec3 endPt = cmd.optimalTrajectory.back();
        const float endU = w.findClosestLoopU(endPt);
        const Vec3 endTan = w.centerline().getTangentAt(endU);
        const float endLat = (endPt - w.centerline().getPointAt(endU)).dot({endTan.z, 0.0f, -endTan.x});
        const bool optLeft = (endLat > 0.35f) || (cmd.behaviorState == "LANE_CHANGE_OVERTAKE");
        const bool okAlign = (sub == MacroSubgoal::OvertakeLeft) ? optLeft : !optLeft;
        if (okAlign) ++alignHits;
        ++alignTot;
      }
    }

    if (outRecall && s >= 4) {
      recSum += out.eval.detRecall;
      precSum += out.eval.detPrecision;
      rmseSum += out.eval.trackPosRmseM;
      ghostSum += out.eval.ghostVehDetections + out.eval.ghostPedDetections;
      w.collectDynamicGroundTruthOBBs(gtAll);
      reqV.clear(); boundV.clear();
      for (const auto& g : gtAll) {
        if (g.semantic != SemanticClass::Vehicle) continue;
        const float d = std::hypot(g.cx - egoBefore.pos.x, g.cz - egoBefore.pos.z);
        if (d <= kEvalRangeM) reqV.push_back(g);
        else if (d <= kBoundRangeM) boundV.push_back(g);
      }
      testutil::accumulatePrCls(SemanticClass::Vehicle, out.detections3D,
                                reqV, boundV, kPrGateM, cfg.rtdetrMaxRangeM, vehAcc);
      ++evalN;
      if (s == 10 && cfg.isV15()) {
        EXPECT_EQ(out.v15RealityRgba.size(), v15_hud::kRealityExpectedBytes);
        uint8_t minR = 255u, maxR = 0u;
        for (size_t k = 0; k < out.v15RealityRgba.size(); k += 4u) {
          minR = std::min(minR, out.v15RealityRgba[k]);
          maxR = std::max(maxR, out.v15RealityRgba[k]);
        }
        EXPECT_GT(static_cast<int>(maxR) - static_cast<int>(minR), 10);
      }
    }

    w.collectDynamicGroundTruthOBBs(gtAll);
    float minTtc = 99.0f;
    const float sE = std::sin(egoBefore.yaw), cE = std::cos(egoBefore.yaw);
    for (const auto& g : gtAll) {
      const float dx = g.cx - egoBefore.pos.x, dz = g.cz - egoBefore.pos.z;
      const float sFwd = sE * dx + cE * dz - (egoBefore.hz + egoBefore.centerOffsetForward + g.hz);
      const float dLat = std::abs(cE * dx - sE * dz);
      const float relSpd = egoBefore.speed - (sE * g.vx + cE * g.vz);
      if (sFwd > 0.0f && dLat < (egoBefore.hx + g.hx + 0.4f) && relSpd > 0.5f) {
        minTtc = std::min(minTtc, sFwd / relSpd);
      }
    }
    ++ttcFrames;
    if (minTtc > 1.0f) ++totalTtcSafe;

    w.stepAgents(dt);
    w.stepEgoAndPhysics(dt, cmd, {});
    if (w.lastContact().occurred) col = true;
    const EgoState& e = w.ego();
    distM += std::hypot(e.pos.x - prevPos.x, e.pos.z - prevPos.z);
    prevPos = e.pos;
    maxAcc = std::max(maxAcc, static_cast<double>(std::abs(e.accel)));
    if (s > 0) jerkSum += std::abs(e.accel - prevAcc) / dt;
    prevAcc = e.accel;
  }

  if (outRecall && evalN > 0) {
    double ap = 0.0, f1 = 0.0;
    testutil::summarizePrCurve(vehAcc, ap, f1);
    *outRecall = recSum / evalN;
    *outPrec = precSum / evalN;
    *outVehAp = ap;
    *outRmse = rmseSum / evalN;
    *outGhosts = ghostSum;
  }
  res.distM = distM;
  res.meanJerk = (steps > 1) ? (jerkSum / (steps - 1)) : 0.0;
  res.nc = col ? 0.0 : 1.0;
  res.dac = (maxLatAbs <= kOnRoadLimitM) ? 1.0 : 0.0;
  res.ep = (refNomDistM > 0.0) ? std::min(1.0, distM / (0.85 * refNomDistM)) : 1.0;
  res.ttc = (ttcFrames > 0) ? (static_cast<double>(totalTtcSafe) / ttcFrames) : 1.0;
  res.comfort = (maxAcc <= 10.5)
    ? std::clamp(1.0 - std::max(0.0, res.meanJerk - 14.0) / 35.0, 0.50, 1.0)
    : 0.25;
  res.pdms = res.nc * res.dac * (5.0 * res.ep + 5.0 * res.ttc + 2.0 * res.comfort) / 12.0;
  res.pred1Mse = (p1Cnt > 0) ? (p1MseSum / p1Cnt) : 0.0;
  res.pred1Cos = (p1Cnt > 0) ? (p1CosSum / p1Cnt) : 1.0;
  res.roll4Mse = (r4Cnt > 0) ? (r4MseSum / r4Cnt) : 0.0;
  res.roll4Cos = (r4Cnt > 0) ? (r4CosSum / r4Cnt) : 1.0;
  res.alignRate = (alignTot > 0) ? (static_cast<double>(alignHits) / alignTot) : 1.0;
  res.steadySurprise = (steps > 4) ? (res.steadySurprise / (steps - 4)) : 0.0;
  pipe.destroy();
  rig.destroy();
  return res;
}

TEST_F(LeWmHJepaSotaMetricsTest, SotaPredictionAccuracyPdmScoreAndPerceptionParity) {
  const ModeEvalCfg cfgs[] = {
    {"legacy", "legacy", "legacy"},
    {"lewm", "legacy", "lewm"},
    {"hjepa", "legacy", "hjepa"},
    {"gf3d+hjepa", "gaussianformer3d", "hjepa"},
  };
  double baseNomDist = 0.0, legRec = 0.0, legPrec = 0.0, legAp = 0.0;
  int legGhosts = 0;
  for (const auto& mc : cfgs) {
    double rec = 0.0, prec = 0.0, ap = 0.0, rmse = 0.0;
    int ghosts = 0;
    const auto nom = runScenarioEval(
      s_ctx, mc, "v15", "nominal", 90, 0.0, &rec, &prec, &ap, &rmse, &ghosts);
    if (std::string(mc.label) == "legacy") {
      baseNomDist = nom.distM;
      legRec = rec;
      legPrec = prec;
      legAp = ap;
      legGhosts = ghosts;
    }
    const auto hb = runScenarioEval(s_ctx, mc, "v15", "hardbrake", 60, 0.0);
    const auto occ = runScenarioEval(s_ctx, mc, "v15", "occluded", 60, 0.0);
    const auto jw = runScenarioEval(s_ctx, mc, "v15", "jaywalk", 60, 0.0);
    const auto ng = runScenarioEval(s_ctx, mc, "v15", "night_glare_jaywalk", 60, 0.0);
    const auto v14n = runScenarioEval(s_ctx, mc, "v14", "nominal", 90, 0.0);

    const double meanEp = std::min(1.0, nom.distM / std::max(1.0, 0.90 * baseNomDist));
    const double meanTtc = 0.2 * (nom.ttc + hb.ttc + occ.ttc + jw.ttc + ng.ttc);
    const double meanComf = 0.2 * (nom.comfort + hb.comfort + occ.comfort + jw.comfort + ng.comfort);
    const double allNc = nom.nc * hb.nc * occ.nc * jw.nc * ng.nc * v14n.nc;
    const double allDac = nom.dac * hb.dac * occ.dac * jw.dac * ng.dac * v14n.dac;
    const double pdms = allNc * allDac * (5.0 * meanEp + 5.0 * meanTtc + 2.0 * meanComf) / 12.0;

    std::printf("[SOTA_PERC] %-10s | Recall=%.4f Prec=%.4f VehAP=%.4f RMSE=%.4fm Ghosts=%d\n",
                mc.label, rec, prec, ap, rmse, ghosts);
    std::printf("[SOTA_PRED] %-10s | 1stepMSE=%.4f 1stepCos=%.4f roll4MSE=%.4f roll4Cos=%.4f align=%.3f hbPeakSurp=%.3f nomMeanSurp=%.3f\n",
                mc.label, nom.pred1Mse, nom.pred1Cos, nom.roll4Mse, nom.roll4Cos,
                nom.alignRate, hb.peakSurprise, nom.steadySurprise);
    std::printf("[SOTA_PDMS] %-10s | NC=%.2f DAC=%.2f EP=%.3f TTC=%.3f Comf=%.3f PDMS=%.4f | v14Dist=%.1fm\n",
                mc.label, allNc, allDac, meanEp, meanTtc, meanComf, pdms, v14n.distM);

    EXPECT_GE(rec, legRec - 0.03) << mc.label;
    EXPECT_GE(prec, legPrec - 0.03) << mc.label;
    EXPECT_GE(ap, legAp - 0.04) << mc.label;
    EXPECT_LE(rmse, 2.50) << mc.label;
    EXPECT_LE(ghosts, legGhosts + 8) << mc.label;
    EXPECT_DOUBLE_EQ(allNc, 1.0) << mc.label;
    EXPECT_DOUBLE_EQ(allDac, 1.0) << mc.label;
    EXPECT_GE(pdms, 0.85) << mc.label;
    if (std::string(mc.wm) != "legacy") {
      EXPECT_GT(nom.pred1Cos, 0.90) << mc.label;
      EXPECT_GT(nom.roll4Cos, 0.75) << mc.label;
      EXPECT_LT(nom.pred1Mse, 0.15) << mc.label;
      EXPECT_LT(nom.roll4Mse, 0.35) << mc.label;
      EXPECT_GE(hb.peakSurprise + 1e-3, nom.steadySurprise) << mc.label;
      if (std::string(mc.wm) == "hjepa") { EXPECT_GE(nom.alignRate, 0.70) << mc.label; }
    }
  }
}

}  // namespace
}  // namespace cybercab::sota
