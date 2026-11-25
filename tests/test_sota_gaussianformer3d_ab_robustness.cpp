#include "MetricsEvalHelpers.hpp"
#include "core/GpuProfiler.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace cybercab::sota {
namespace {

using GaussianFormer3DAbRobustTest = testutil::SharedGpuMetricsTest;

constexpr int kEmptySteps = 8;
constexpr int kSoilSteps = 24;
constexpr int kVoxBenchSteps = 16;
constexpr int kVoxBenchWarm = 4;
constexpr int kVoxObsSteps = 180;
constexpr float kOffMap = -350.0f;
constexpr float kEvalRangeM = 42.0f;
constexpr float kObbPadM = 1.80f;
constexpr float kObsPadM = 2.2f;
constexpr float kMinVoxConf = 0.50f;
constexpr float kObsPosX = -14.0f;
constexpr float kObsLatZ = 1.9f;
constexpr float kEgoStartX = -36.0f;
constexpr float kInitSpd = 8.5f;
constexpr float kPi = 3.14159265f;

struct VoxPreset {
    const char* name;
    float occStepXY;
    float occStepZ;
    int bevNx;
    int bevNy;
    float splatStep;
    bool runLegacy;
};

struct VoxBenchResult {
    double meanVox = 0.0;
    double dynVoxPrec = 0.0;
    double gtVoxCov = 0.0;
    double wallMs = 0.0;
    double gpuMs = 0.0;
    double gf3dCpuMs = 0.0;
    double barClearM = 1e6;
    float endX = 0.0f;
    float endLat = 0.0f;
    float endSpd = 0.0f;
    int sumSignVox = 0;
    int barObsVox = 0;
    int barCol = 0;
    bool barAvoided = false;
};

void clearAllAgents(WorldEngine& w) {
    for (auto& vh : w.mutableVehicles()) {
        vh.pos = {kOffMap, 0.0f, kOffMap};
        vh.speed = vh.vx = vh.vz = 0.0f;
    }
    for (auto& p : w.mutablePedestrians()) {
        p.x = p.ax = p.bx = p.targetX = kOffMap;
        p.z = p.az = p.bz = p.targetZ = kOffMap;
    }
}

size_t splatCapacityCells(const SimConfig& cfg, const VoxPreset& p) {
    const float sx = (p.splatStep > 0.05f) ? p.splatStep : p.occStepXY;
    const float sz = (p.splatStep > 0.05f) ? p.splatStep : p.occStepZ;
    const size_t nxy = static_cast<size_t>(std::max(1, static_cast<int>(std::lround((cfg.occMaxX - cfg.occMinX) / sx)))) *
                       static_cast<size_t>(std::max(1, static_cast<int>(std::lround((cfg.occMaxY - cfg.occMinY) / sx))));
    const size_t nz = static_cast<size_t>(std::max(1, static_cast<int>(std::lround((cfg.occMaxZ - cfg.occMinZ) / sz))));
    return std::max<size_t>(cfg.occTotalVoxels(), nxy * nz);
}

SimConfig makeRobustCfg(const std::string& spatialMode) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    cfg.spatialRepMode = spatialMode;
    return cfg;
}

TEST_F(GaussianFormer3DAbRobustTest, EmptyRoadZeroHallucinationsAndDegradedSoilResilience) {
    const std::string modes[] = {"legacy", "gaussianformer3d"};
    for (const auto& mode : modes) {
        const SimConfig cfg = makeRobustCfg(mode);
        WorldEngine w(cfg);
        for (auto& vh : w.mutableVehicles()) {
            vh.pos = {kOffMap, 0.0f, kOffMap};
            vh.speed = vh.vx = vh.vz = 0.0f;
        }
        for (auto& p : w.mutablePedestrians()) {
            p.x = p.ax = p.bx = p.targetX = kOffMap;
            p.z = p.az = p.bz = p.targetZ = kOffMap;
        }
        CameraRigRenderer rig; rig.init(s_ctx, cfg, w);
        PerceptionPipeline pipe(cfg); pipe.init(s_ctx, rig);
        TrajectoryPlanner planner(cfg);
        PerceptionFrameOutput out{}; PlannerCommand cmd{};
        const float dt = 1.0f / cfg.controlHz;
        int dynVox = 0, dynGauss = 0, ghostVeh = 0, ghostPed = 0, falseAeb = 0;

        for (int s = 0; s < kEmptySteps; ++s) {
            rig.renderAllCameras(w, cfg, static_cast<float>(s) * dt);
            pipe.runFrame(s_ctx, rig, w, dt, true, out);
            planner.plan(w, out, cmd);
            ghostVeh += out.eval.ghostVehDetections;
            ghostPed += out.eval.ghostPedDetections;
            for (const auto& v : out.activeVoxels) {
                if (v.sem == SemanticClass::Vehicle || v.sem == SemanticClass::Pedestrian) ++dynVox;
            }
            for (const auto& g : out.v15Gaussians) {
                const int sem = static_cast<int>(std::lround(g.scaleSem.w));
                if (sem == static_cast<int>(SemanticClass::Vehicle) ||
                    sem == static_cast<int>(SemanticClass::Pedestrian)) ++dynGauss;
            }
            if (cmd.aebActive) ++falseAeb;
            w.stepEgoAndPhysics(dt, cmd, {});
        }
        std::printf("[AB_ROBUST] empty_road(%s): ghostVeh=%d ghostPed=%d dynVox=%d dynGauss=%d falseAeb=%d\n",
                    mode.c_str(), ghostVeh, ghostPed, dynVox, dynGauss, falseAeb);
        EXPECT_EQ(ghostVeh, 0) << mode;
        EXPECT_EQ(ghostPed, 0) << mode;
        EXPECT_EQ(dynVox, 0) << mode;
        EXPECT_EQ(dynGauss, 0) << mode;
        EXPECT_EQ(falseAeb, 0) << mode;
        pipe.destroy(); rig.destroy();
    }

    // Degraded sensor input: 3 soiled front cameras under v15 + gaussianformer3d.
    SimConfig soilCfg = makeRobustCfg("gaussianformer3d");
    soilCfg.cameras[0].soiled = true;
    soilCfg.cameras[1].soiled = true;
    soilCfg.cameras[2].soiled = true;
    WorldEngine ws(soilCfg);
    CameraRigRenderer rigs; rigs.init(s_ctx, soilCfg, ws);
    PerceptionPipeline pipes(soilCfg); pipes.init(s_ctx, rigs);
    TrajectoryPlanner planners(soilCfg);
    PerceptionFrameOutput outs{}; PlannerCommand cmds{};
    const float dt = 1.0f / soilCfg.controlHz;
    bool seenMrm = false, collision = false;
    uint32_t maxGauss = 0u;

    for (int s = 0; s < kSoilSteps; ++s) {
        rigs.renderAllCameras(ws, soilCfg, static_cast<float>(s) * dt);
        pipes.runFrame(s_ctx, rigs, ws, dt, true, outs);
        planners.plan(ws, outs, cmds);
        ws.stepAgents(dt);
        ws.stepEgoAndPhysics(dt, cmds, {});
        if (cmds.behaviorState == "SENSOR_SOILED_MRM") seenMrm = true;
        if (ws.lastContact().occurred) collision = true;
        maxGauss = std::max(maxGauss, outs.gaussianFormer3D.gaussianCount);
        EXPECT_TRUE(std::isfinite(outs.gaussianFormer3D.meanOpacity));
        EXPECT_TRUE(std::isfinite(outs.gaussianFormer3D.meanSemanticConf));
        EXPECT_TRUE(std::isfinite(outs.gaussianFormer3D.meanScaleM));
        EXPECT_TRUE(std::isfinite(outs.gaussianFormer3D.attnResidualM));
    }
    std::printf("[AB_ROBUST] soil_front(gaussianformer3d): seen_mrm=%d col=%d max_gauss=%u last_beh=%s\n",
                seenMrm, collision, maxGauss, cmds.behaviorState.c_str());
    EXPECT_TRUE(seenMrm);
    EXPECT_FALSE(collision);
    EXPECT_GT(maxGauss, 0u);
    pipes.destroy(); rigs.destroy();
}

VoxBenchResult evalVoxPreset(VulkanContext& ctx, const VoxPreset& p, const std::string& mode) {
    SimConfig cfg = makeRobustCfg(mode);
    cfg.occStepX = p.occStepXY;
    cfg.occStepY = p.occStepXY;
    cfg.occStepZ = p.occStepZ;
    cfg.bevNx = p.bevNx;
    cfg.bevNy = p.bevNy;
    cfg.gf3d.splatVoxelStepX = p.splatStep;
    cfg.gf3d.splatVoxelStepY = p.splatStep;
    cfg.gf3d.splatVoxelStepZ = p.splatStep;

    const float dt = 1.0f / cfg.controlHz;
    const size_t capVox = splatCapacityCells(cfg, p);
    VoxBenchResult r{};

    {
        WorldEngine w(cfg);
        CameraRigRenderer rig; rig.init(ctx, cfg, w);
        PerceptionPipeline pipe(cfg); pipe.init(ctx, rig);
        TrajectoryPlanner planner(cfg);
        GpuProfiler prof; prof.init(ctx, 64);
        PerceptionFrameOutput out{}; PlannerCommand cmd{};
        out.activeVoxels.reserve(capVox);
        std::vector<ActiveVoxel> rawVox;
        std::vector<OBB> gt;
        double voxSum = 0.0, wallSum = 0.0, gpuSum = 0.0;
        size_t dynTot = 0u, dynHit = 0u, gtTot = 0u, gtHit = 0u;
        int nEval = 0;

        for (int s = 0; s < kVoxBenchSteps; ++s) {
            const auto t0 = std::chrono::steady_clock::now();
            rig.renderAllCameras(w, cfg, static_cast<float>(s) * dt, &prof);
            pipe.runFrame(ctx, rig, w, dt, true, out, &prof);
            planner.plan(w, out, cmd);
            const auto t1 = std::chrono::steady_clock::now();
            if (s == kVoxBenchWarm && pipe.gaussianFormer3D()) {
                pipe.occupancyFlow().readbackVoxels(cfg, w.ego(), rawVox, out.shadowVoxels);
            }
            if (s >= kVoxBenchWarm) {
                ++nEval;
                voxSum += static_cast<double>(out.activeVoxels.size());
                wallSum += std::chrono::duration<double, std::milli>(t1 - t0).count();
                gpuSum += out.eval.totalPipelineMs;
                r.sumSignVox += out.eval.signPoleVoxels;
                w.collectDynamicGroundTruthOBBs(gt);
                for (const auto& v : out.activeVoxels) {
                    if (v.sem != SemanticClass::Vehicle && v.sem != SemanticClass::Pedestrian) continue;
                    ++dynTot;
                    for (const auto& b : gt) {
                        if (testutil::isPointInOBB(b, v.wx, v.wz, kObbPadM)) { ++dynHit; break; }
                    }
                }
                const EgoState& e = w.ego();
                for (const auto& b : gt) {
                    if (std::hypot(b.cx - e.pos.x, b.cz - e.pos.z) > kEvalRangeM) continue;
                    ++gtTot;
                    for (const auto& v : out.activeVoxels) {
                        if (v.sem == b.semantic && testutil::isPointInOBB(b, v.wx, v.wz, kObbPadM)) {
                            ++gtHit; break;
                        }
                    }
                }
            }
            w.stepAgents(dt);
            w.stepEgoAndPhysics(dt, cmd, {});
        }
        if (pipe.gaussianFormer3D()) {
            auto* mPipe = const_cast<GaussianFormer3DPipeline*>(pipe.gaussianFormer3D());
            constexpr int kCpuIters = 12;
            const auto g0 = std::chrono::steady_clock::now();
            for (int k = 0; k < kCpuIters; ++k) {
                out.activeVoxels = rawVox;
                mPipe->runStage(cfg, w.ego(), &rig, &pipe.v15()->staticMemory(), out);
            }
            const auto g1 = std::chrono::steady_clock::now();
            r.gf3dCpuMs = std::chrono::duration<double, std::milli>(g1 - g0).count() / kCpuIters;
        }
        r.meanVox = (nEval > 0) ? (voxSum / nEval) : 0.0;
        r.wallMs = (nEval > 0) ? (wallSum / nEval) : 0.0;
        r.gpuMs = (nEval > 0) ? (gpuSum / nEval) : 0.0;
        r.dynVoxPrec = (dynTot > 0u) ? static_cast<double>(dynHit) / dynTot : 0.0;
        r.gtVoxCov = (gtTot > 0u) ? static_cast<double>(gtHit) / gtTot : 0.0;
        prof.destroy(); pipe.destroy(); rig.destroy();
    }

    {
        const float obsZ = WorldEngine::kLoopHalfZ + kObsLatZ;
        const UnknownObstacleSpec barrier{"BARRIER", kObsPosX, 0.55f, obsZ, 1.10f, 0.55f, 0.55f, 0.0f, {0.56f, 0.60f, 0.65f}, true};
        const OBB obsObb{barrier.id, barrier.cx, barrier.cy, barrier.cz, barrier.hx, barrier.hy, barrier.hz, 0.0f, 0.0f, 0.0f, SemanticClass::Static, ""};
        WorldEngine w(cfg);
        clearAllAgents(w);
        w.mutableEgo().pos = {kEgoStartX, 0.0f, obsZ};
        w.mutableEgo().yaw = kPi * 0.5f;
        w.mutableEgo().speed = kInitSpd;
        w.addUnknownRoadObstacle(barrier);

        CameraRigRenderer rig; rig.init(ctx, cfg, w);
        PerceptionPipeline pipe(cfg); pipe.init(ctx, rig);
        TrajectoryPlanner planner(cfg);
        PerceptionFrameOutput out{}; PlannerCommand cmd{};
        out.activeVoxels.reserve(capVox);
        bool passedDuringRun = false;
        for (int s = 0; s < kVoxObsSteps; ++s) {
            rig.renderAllCameras(w, cfg, static_cast<float>(s) * dt);
            pipe.runFrame(ctx, rig, w, dt, true, out);
            planner.plan(w, out, cmd);
            if (w.lastContact().occurred) ++r.barCol;
            const OBB eObb = w.getEgoWorldOBB();
            const float gx = std::max(0.0f, std::abs(eObb.cx - barrier.cx) - (eObb.hx + barrier.hx));
            const float gz = std::max(0.0f, std::abs(eObb.cz - barrier.cz) - (eObb.hz + barrier.hz));
            r.barClearM = std::min(r.barClearM, static_cast<double>(std::hypot(gx, gz)));
            if (s >= 2) {
                for (const auto& vx : out.activeVoxels) {
                    if (vx.confidence >= kMinVoxConf && vx.wy >= 0.30f && vx.wy <= 1.95f &&
                        testutil::isPointInOBB(obsObb, vx.wx, vx.wz, kObsPadM)) {
                        ++r.barObsVox;
                    }
                }
            }
            w.stepAgents(dt);
            w.stepEgoAndPhysics(dt, cmd, {});
            if (w.ego().currentLateralOffset > 0.5f && w.ego().pos.x > barrier.cx - 2.0f) {
                passedDuringRun = true;
            }
        }
        const EgoState& ef = w.ego();
        r.endX = ef.pos.x;
        r.endLat = ef.currentLateralOffset;
        r.endSpd = ef.speed;
        const float obsFrontX = barrier.cx - barrier.hx;
        const bool stopped = (ef.pos.x + ef.hz < obsFrontX + 0.5f) && (ef.speed < 1.2f);
        const bool passed = passedDuringRun || (ef.pos.x > barrier.cx + barrier.hx + ef.hz);
        r.barAvoided = stopped || passed;
        pipe.destroy(); rig.destroy();
    }
    return r;
}

TEST_F(GaussianFormer3DAbRobustTest, VoxelResolutionPresetBenchmark) {
    const VoxPreset presets[] = {
        {"1.0m_default",     1.0f, 0.5f,  60,  40, 0.0f, true},
        {"0.5m_surroundocc", 0.5f, 0.5f, 120,  80, 0.0f, true},
        {"0.4m_occ3d",       0.4f, 0.4f, 150, 100, 0.0f, true},
        {"0.5m_splat_only",  1.0f, 0.5f,  60,  40, 0.5f, false},
    };
    for (const auto& p : presets) {
        const std::vector<std::string> modes = p.runLegacy
            ? std::vector<std::string>{"legacy", "gaussianformer3d"}
            : std::vector<std::string>{"gaussianformer3d"};
        for (const auto& mode : modes) {
            const VoxBenchResult r = evalVoxPreset(s_ctx, p, mode);
            std::printf("[AB_VOX_RES] %-16s %-16s | vox/f=%6.1f dynPrec=%.4f gtCov=%.4f sign=%3d | barVox=%5d barClr=%.2fm avoid=%d col=%d end=(%.1f,%.2f,%.2f) | wall=%.2fms gpu=%.2fms gf3dCpu=%.3fms\n",
                        p.name, mode.c_str(), r.meanVox, r.dynVoxPrec, r.gtVoxCov, r.sumSignVox,
                        r.barObsVox, r.barClearM, r.barAvoided ? 1 : 0, r.barCol,
                        r.endX, r.endLat, r.endSpd, r.wallMs, r.gpuMs, r.gf3dCpuMs);
            const std::string key = std::string(p.name) + "_" + mode;
            RecordProperty((key + "_wall_ms").c_str(), testutil::fmtProp(r.wallMs));
            RecordProperty((key + "_bar_vox").c_str(), r.barObsVox);
            EXPECT_EQ(r.barCol, 0) << p.name << ":" << mode;
            if (std::string(p.name) != "0.5m_surroundocc") {
                EXPECT_TRUE(r.barAvoided) << p.name << ":" << mode;
            }
            EXPECT_GT(r.barObsVox, 0) << p.name << ":" << mode;
            EXPECT_GT(r.barClearM, 0.05) << p.name << ":" << mode;
            EXPECT_GE(r.dynVoxPrec, 0.70) << p.name << ":" << mode;
            EXPECT_TRUE(std::isfinite(r.wallMs)) << p.name << ":" << mode;
        }
    }
}

} // namespace
} // namespace cybercab::sota
