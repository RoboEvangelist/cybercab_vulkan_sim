#include "MetricsEvalHelpers.hpp"
#include "perception/BEVFormer.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/v15/GaussianWorldModel.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "planning/v15/StaticVoxelGuard.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace cybercab::sota {
namespace {

using GaussianFormer3DAbMetricsTest = testutil::SharedGpuMetricsTest;

constexpr int kPrSteps = 40;
constexpr int kVoxSteps = 16;
constexpr int kWarmSteps = 4;
constexpr int kObsSteps = 130;
constexpr float kOffMap = -350.0f;
constexpr float kEvalRangeM = 42.0f;
constexpr float kBoundRangeM = 48.0f;
constexpr float kPrGateM = 5.0f;
constexpr float kObbPadM = 1.80f;
constexpr float kVoxLocEps = 0.05f;
constexpr float kOccPadZ = 0.5f;
constexpr float kFreeSpaceClearRadM = 5.0f;
constexpr float kGaussHitThresh = 0.35f;
constexpr float kObsPadM = 2.2f;
constexpr float kObsPosX = -14.0f;
constexpr float kObsLatZ = 1.9f;
constexpr float kEgoStartX = -36.0f;
constexpr float kInitSpd = 8.5f;
constexpr float kMinVoxConf = 0.50f;
constexpr float kStopGapPadM = 0.5f;
constexpr float kStopMaxSpdMs = 1.2f;
constexpr float kPassMinLatM = 0.5f;
constexpr float kPassBehindObsM = 2.0f;
constexpr float kMinSafeClearM = 0.05f;

struct AbPercMetrics {
    double meanRecall = 0.0;
    double meanPrec = 0.0;
    double vehAp = 0.0;
    double vehF1 = 0.0;
    double meanRmseM = 0.0;
    double ghostPerFrame = 0.0;
    double meanVoxels = 0.0;
    double meanGaussians = 0.0;
    double maxLocErrM = 0.0;
    double dynVoxPrec = 0.0;
    double gtObbVoxCov = 0.0;
    double gaussGtMeanOcc = 0.0;
    double gaussGtHitRate = 0.0;
    double gaussFreeMeanOcc = 0.0;
    int misTrucks = 0;
    int sumTruckVeh = 0;
    int sumTruckPed = 0;
    int sumSignVox = 0;
};

struct AbObsMetrics {
    int collisions = 0;
    int obsVox = 0;
    float minFsErrM = 99.0f;
    float minClearM = 99.0f;
    float finalX = 0.0f;
    float finalLat = 0.0f;
    float finalSpd = 0.0f;
    bool avoided = false;
};

SimConfig makeAbCfg(const std::string& spatialMode, const std::string& res = "half") {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    cfg.spatialRepMode = spatialMode;
    cfg.initialResMode = res;
    const CamResPreset p = cfg.resolveInitialRes();
    cfg.camWidth = p.width;
    cfg.camHeight = p.height;
    return cfg;
}

void clearUsers(WorldEngine& w) {
    for (auto& vh : w.mutableVehicles()) {
        vh.pos = {kOffMap, 0.0f, kOffMap};
        vh.speed = vh.vx = vh.vz = 0.0f;
    }
    for (auto& p : w.mutablePedestrians()) {
        p.x = p.ax = p.bx = p.targetX = kOffMap;
        p.z = p.az = p.bz = p.targetZ = kOffMap;
    }
}

AbPercMetrics measurePercAndVoxels(VulkanContext& ctx, const std::string& spatialMode) {
    const SimConfig cfg = makeAbCfg(spatialMode);
    const float dt = 1.0f / cfg.controlHz;
    AbPercMetrics m{};

    {
        WorldEngine w(cfg);
        CameraRigRenderer rig; rig.init(ctx, cfg, w);
        PerceptionPipeline pipe(cfg); pipe.init(ctx, rig);
        TrajectoryPlanner planner(cfg);
        PerceptionFrameOutput out{}; PlannerCommand cmd{};
        testutil::PrAccum vehAcc{};
        std::vector<OBB> gtAll, reqVeh, boundVeh;
        double recSum = 0.0, precSum = 0.0, rmseSum = 0.0;
        double ghostSum = 0.0, voxSum = 0.0, gaussSum = 0.0;
        int nEval = 0;

        for (int s = 0; s < kPrSteps; ++s) {
            rig.renderAllCameras(w, cfg, static_cast<float>(s) * dt);
            pipe.runFrame(ctx, rig, w, dt, true, out);
            planner.plan(w, out, cmd);
            if (s >= kWarmSteps) {
                recSum += out.eval.detRecall;
                precSum += out.eval.detPrecision;
                rmseSum += out.eval.trackPosRmseM;
                ghostSum += out.eval.ghostVehDetections + out.eval.ghostPedDetections;
                voxSum += static_cast<double>(out.activeVoxels.size());
                gaussSum += static_cast<double>(out.v15Gaussians.size());
                m.misTrucks += out.eval.misclassifiedTrucks;
                w.collectDynamicGroundTruthOBBs(gtAll);
                reqVeh.clear(); boundVeh.clear();
                const EgoState& e = w.ego();
                for (const auto& g : gtAll) {
                    if (g.semantic != SemanticClass::Vehicle) continue;
                    const float d = std::hypot(g.cx - e.pos.x, g.cz - e.pos.z);
                    if (d <= kEvalRangeM) reqVeh.push_back(g);
                    else if (d <= kBoundRangeM) boundVeh.push_back(g);
                }
                testutil::accumulatePrCls(SemanticClass::Vehicle, out.detections3D,
                                          reqVeh, boundVeh, kPrGateM, cfg.rtdetrMaxRangeM, vehAcc);
                ++nEval;
            }
            w.stepAgents(dt);
            w.stepEgoAndPhysics(dt, cmd, {});
        }
        m.meanRecall = recSum / nEval;
        m.meanPrec = precSum / nEval;
        m.meanRmseM = rmseSum / nEval;
        m.ghostPerFrame = ghostSum / nEval;
        m.meanVoxels = voxSum / nEval;
        m.meanGaussians = gaussSum / nEval;
        testutil::summarizePrCurve(vehAcc, m.vehAp, m.vehF1);
        pipe.destroy(); rig.destroy();
    }

    {
        WorldEngine w(cfg);
        CameraRigRenderer rig; rig.init(ctx, cfg, w);
        PerceptionPipeline pipe(cfg); pipe.init(ctx, rig);
        TrajectoryPlanner planner(cfg);
        PerceptionFrameOutput out{}; PlannerCommand cmd{};
        std::vector<OBB> gt;
        size_t dynTot = 0u, dynHit = 0u;
        size_t gtCovTot = 0u, gtCovHit = 0u;
        double gGtOccSum = 0.0, gFreeOccSum = 0.0;
        size_t gGtCnt = 0u, gGtHit = 0u, gFreeCnt = 0u;

        for (int s = 0; s < kVoxSteps; ++s) {
            rig.renderAllCameras(w, cfg, static_cast<float>(s) * dt);
            pipe.runFrame(ctx, rig, w, dt, true, out);
            planner.plan(w, out, cmd);
            const Mat4 wFromIso = BEVFormer::worldFromIsoEgo(w.ego());
            w.collectDynamicGroundTruthOBBs(gt);

            for (const auto& v : out.activeVoxels) {
                const Vec3 exp = wFromIso.transformPoint({v.egoX, v.egoY, v.wy});
                m.maxLocErrM = std::max(m.maxLocErrM, static_cast<double>(std::hypot(v.wx - exp.x, v.wz - exp.z)));
                EXPECT_GE(v.egoX, cfg.occMinX - kVoxLocEps);
                EXPECT_LE(v.egoX, cfg.occMaxX + kVoxLocEps);
                EXPECT_GE(v.egoY, cfg.occMinY - kVoxLocEps);
                EXPECT_LE(v.egoY, cfg.occMaxY + kVoxLocEps);
                EXPECT_GE(v.wy, 0.0f);
                EXPECT_LE(v.wy, cfg.occMaxZ + kOccPadZ);
                if (v.sem == SemanticClass::Vehicle || v.sem == SemanticClass::Pedestrian) {
                    ++dynTot;
                    for (const auto& b : gt) {
                        if (testutil::isPointInOBB(b, v.wx, v.wz, kObbPadM)) { ++dynHit; break; }
                    }
                }
            }
            if (s >= kWarmSteps) {
                m.sumTruckVeh += out.eval.truckVehVoxels;
                m.sumTruckPed += out.eval.truckPedVoxels;
                m.sumSignVox += out.eval.signPoleVoxels;
                const EgoState& e = w.ego();
                for (const auto& b : gt) {
                    if (std::hypot(b.cx - e.pos.x, b.cz - e.pos.z) > kEvalRangeM) continue;
                    ++gtCovTot;
                    bool voxHit = false;
                    for (const auto& v : out.activeVoxels) {
                        if (v.sem == b.semantic && testutil::isPointInOBB(b, v.wx, v.wz, kObbPadM)) {
                            voxHit = true; break;
                        }
                    }
                    if (voxHit) ++gtCovHit;
                    const float p = v15::GaussianWorldModel::queryOccupancyProb(
                        out.v15Gaussians, {b.cx, b.cy, b.cz}, 0.0f, true);
                    gGtOccSum += p;
                    if (p >= kGaussHitThresh) ++gGtHit;
                    ++gGtCnt;
                }
                const float fwdDists[] = {6.0f, 10.0f, 14.0f};
                for (float dFwd : fwdDists) {
                    const Vec3 probe{e.pos.x + std::sin(e.yaw) * dFwd, 0.7f, e.pos.z + std::cos(e.yaw) * dFwd};
                    bool nearGt = false;
                    for (const auto& b : gt) {
                        if (testutil::isPointInOBB(b, probe.x, probe.z, kFreeSpaceClearRadM)) { nearGt = true; break; }
                    }
                    if (!nearGt) {
                        gFreeOccSum += v15::GaussianWorldModel::queryOccupancyProb(
                            out.v15Gaussians, probe, 0.0f, true);
                        ++gFreeCnt;
                    }
                }
            }
            w.stepAgents(dt);
            w.stepEgoAndPhysics(dt, cmd, {});
        }
        m.dynVoxPrec = dynTot > 0u ? static_cast<double>(dynHit) / dynTot : 0.0;
        m.gtObbVoxCov = gtCovTot > 0u ? static_cast<double>(gtCovHit) / gtCovTot : 0.0;
        m.gaussGtMeanOcc = gGtCnt > 0u ? gGtOccSum / gGtCnt : 0.0;
        m.gaussGtHitRate = gGtCnt > 0u ? static_cast<double>(gGtHit) / gGtCnt : 0.0;
        m.gaussFreeMeanOcc = gFreeCnt > 0u ? gFreeOccSum / gFreeCnt : 0.0;
        pipe.destroy(); rig.destroy();
    }
    return m;
}

AbObsMetrics runOpenSetObs(VulkanContext& ctx, const std::string& spatialMode, const UnknownObstacleSpec& obs) {
    const SimConfig cfg = makeAbCfg(spatialMode);
    WorldEngine w(cfg);
    clearUsers(w);
    w.mutableEgo().pos = {kEgoStartX, 0.0f, WorldEngine::kLoopHalfZ + kObsLatZ};
    w.mutableEgo().yaw = kPi * 0.5f;
    w.mutableEgo().speed = kInitSpd;
    w.addUnknownRoadObstacle(obs);

    CameraRigRenderer rig; rig.init(ctx, cfg, w);
    PerceptionPipeline pipe(cfg); pipe.init(ctx, rig);
    TrajectoryPlanner planner(cfg);
    PerceptionFrameOutput out{}; PlannerCommand cmd{};
    const float dt = 1.0f / cfg.controlHz;
    const OBB obsObb{obs.id, obs.cx, obs.cy, obs.cz, obs.hx, obs.hy, obs.hz, obs.yaw, 0.0f, 0.0f, SemanticClass::Static, ""};
    const float obsFrontX = obs.cx - obs.hx;
    AbObsMetrics res{};

    for (int s = 0; s < kObsSteps; ++s) {
        rig.renderAllCameras(w, cfg, static_cast<float>(s) * dt);
        pipe.runFrame(ctx, rig, w, dt, true, out);
        planner.plan(w, out, cmd);
        if (w.lastContact().occurred) ++res.collisions;
        const OBB egoObb = w.getEgoWorldOBB();
        const float gx = std::max(0.0f, std::abs(egoObb.cx - obs.cx) - (egoObb.hx + obs.hx));
        const float gz = std::max(0.0f, std::abs(egoObb.cz - obs.cz) - (egoObb.hz + obs.hz));
        res.minClearM = std::min(res.minClearM, std::hypot(gx, gz));
        if (s >= 2) {
            for (const auto& vx : out.activeVoxels) {
                if (vx.confidence < kMinVoxConf || !testutil::isPointInOBB(obsObb, vx.wx, vx.wz, kObsPadM) ||
                    vx.wy < 0.30f || vx.wy > 1.95f) continue;
                ++res.obsVox;
                if (std::abs(v15::voxelRoadLateralM(w, vx)) <= 3.2f) {
                    res.minFsErrM = std::min(res.minFsErrM, std::abs(vx.wx - obsFrontX));
                }
            }
        }
        w.stepAgents(dt);
        w.stepEgoAndPhysics(dt, cmd, {});
    }
    const EgoState& ef = w.ego();
    res.finalX = ef.pos.x;
    res.finalLat = ef.currentLateralOffset;
    res.finalSpd = ef.speed;
    const bool stopped = (ef.pos.x + ef.hz < obsFrontX + kStopGapPadM) && (ef.speed < kStopMaxSpdMs);
    const bool passed = (ef.currentLateralOffset > kPassMinLatM) && (ef.pos.x > obs.cx - kPassBehindObsM);
    res.avoided = stopped || passed;
    pipe.destroy(); rig.destroy();
    return res;
}

TEST_F(GaussianFormer3DAbMetricsTest, ABPerceptionVoxelGaussianAndOpenSetQuality) {
    const AbPercMetrics leg = measurePercAndVoxels(s_ctx, "legacy");
    const AbPercMetrics gf = measurePercAndVoxels(s_ctx, "gaussianformer3d");

    const float obsZ = WorldEngine::kLoopHalfZ + kObsLatZ;
    const UnknownObstacleSpec cargo{"CARGO", kObsPosX, 0.60f, obsZ, 0.85f, 0.60f, 0.85f, 0.0f, {0.84f, 0.48f, 0.12f}, true};
    const UnknownObstacleSpec barrier{"BARRIER", kObsPosX, 0.55f, obsZ, 1.10f, 0.55f, 0.55f, 0.0f, {0.56f, 0.60f, 0.65f}, true};
    const AbObsMetrics legCargo = runOpenSetObs(s_ctx, "legacy", cargo);
    const AbObsMetrics gfCargo = runOpenSetObs(s_ctx, "gaussianformer3d", cargo);
    const AbObsMetrics legBar = runOpenSetObs(s_ctx, "legacy", barrier);
    const AbObsMetrics gfBar = runOpenSetObs(s_ctx, "gaussianformer3d", barrier);

    std::printf("\n[AB_METRICS] Recall: leg=%.4f gf3d=%.4f | Prec: leg=%.4f gf3d=%.4f | VehAP: leg=%.4f gf3d=%.4f | RMSE: leg=%.4f gf3d=%.4f\n",
                leg.meanRecall, gf.meanRecall, leg.meanPrec, gf.meanPrec, leg.vehAp, gf.vehAp, leg.meanRmseM, gf.meanRmseM);
    std::printf("[AB_METRICS] DynVoxPrec: leg=%.4f gf3d=%.4f | GtVoxCov: leg=%.4f gf3d=%.4f | Vox/f: leg=%.1f gf3d=%.1f | Gauss/f: leg=%.1f gf3d=%.1f\n",
                leg.dynVoxPrec, gf.dynVoxPrec, leg.gtObbVoxCov, gf.gtObbVoxCov, leg.meanVoxels, gf.meanVoxels, leg.meanGaussians, gf.meanGaussians);
    std::printf("[AB_METRICS] GaussGtOcc: leg=%.4f gf3d=%.4f | GaussGtHit: leg=%.4f gf3d=%.4f | GaussFreeOcc: leg=%.4f gf3d=%.4f\n",
                leg.gaussGtMeanOcc, gf.gaussGtMeanOcc, leg.gaussGtHitRate, gf.gaussGtHitRate, leg.gaussFreeMeanOcc, gf.gaussFreeMeanOcc);
    std::printf("[AB_METRICS] TruckVeh: leg=%d gf3d=%d | SignVox: leg=%d gf3d=%d | Cargo(vox=%d/%d clear=%.2f/%.2f) | Barrier(vox=%d/%d clear=%.2f/%.2f legEnd=(%.1f,%.2f,%.2f) gfEnd=(%.1f,%.2f,%.2f))\n",
                leg.sumTruckVeh, gf.sumTruckVeh, leg.sumSignVox, gf.sumSignVox,
                legCargo.obsVox, gfCargo.obsVox, legCargo.minClearM, gfCargo.minClearM,
                legBar.obsVox, gfBar.obsVox, legBar.minClearM, gfBar.minClearM,
                legBar.finalX, legBar.finalLat, legBar.finalSpd,
                gfBar.finalX, gfBar.finalLat, gfBar.finalSpd);

    RecordProperty("leg_dyn_vox_prec", testutil::fmtProp(leg.dynVoxPrec));
    RecordProperty("gf3d_dyn_vox_prec", testutil::fmtProp(gf.dynVoxPrec));
    RecordProperty("leg_gt_vox_cov", testutil::fmtProp(leg.gtObbVoxCov));
    RecordProperty("gf3d_gt_vox_cov", testutil::fmtProp(gf.gtObbVoxCov));
    RecordProperty("leg_gauss_gt_occ", testutil::fmtProp(leg.gaussGtMeanOcc));
    RecordProperty("gf3d_gauss_gt_occ", testutil::fmtProp(gf.gaussGtMeanOcc));

    EXPECT_LT(leg.maxLocErrM, kVoxLocEps);
    EXPECT_LT(gf.maxLocErrM, kVoxLocEps);
    EXPECT_EQ(leg.misTrucks, 0);
    EXPECT_EQ(gf.misTrucks, 0);
    EXPECT_EQ(leg.sumTruckPed, 0);
    EXPECT_EQ(gf.sumTruckPed, 0);
    EXPECT_GT(leg.sumTruckVeh, 0);
    EXPECT_GT(gf.sumTruckVeh, 0);
    EXPECT_GT(leg.sumSignVox, 0);
    EXPECT_GT(gf.sumSignVox, 0);
    EXPECT_EQ(legCargo.collisions, 0);
    EXPECT_EQ(gfCargo.collisions, 0);
    EXPECT_EQ(legBar.collisions, 0);
    EXPECT_EQ(gfBar.collisions, 0);
    EXPECT_TRUE(legCargo.avoided);
    EXPECT_TRUE(gfCargo.avoided);
    EXPECT_TRUE(legBar.avoided);
    EXPECT_TRUE(gfBar.avoided);
    EXPECT_GT(gfCargo.obsVox, 0);
    EXPECT_GT(gfBar.obsVox, 0);
    EXPECT_GT(gfCargo.minClearM, kMinSafeClearM);
    EXPECT_GT(gfBar.minClearM, kMinSafeClearM);
}

} // namespace
} // namespace cybercab::sota
