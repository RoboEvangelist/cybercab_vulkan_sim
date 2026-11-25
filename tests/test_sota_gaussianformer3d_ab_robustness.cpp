#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <string>

namespace cybercab::sota {
namespace {

using GaussianFormer3DAbRobustTest = testutil::SharedGpuMetricsTest;

constexpr int kEmptySteps = 8;
constexpr int kSoilSteps = 24;
constexpr float kOffMap = -350.0f;

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

} // namespace
} // namespace cybercab::sota
