#include "MetricsEvalHelpers.hpp"
#include "app/V15HudOverlay.hpp"
#include "core/TraceLogger.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"

#include <gtest/gtest.h>
#include <json/json.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

namespace cybercab::sota {
namespace {

using GaussianFormer3DGpuTest = testutil::SharedGpuMetricsTest;

constexpr uint8_t kMinNonZeroRgb = 4u;
constexpr float kMinMeanSemConf = 0.40f;

SimConfig makeConfig(const std::string& pipelineVer,
                     const std::string& spatialMode,
                     const std::string& resMode = "half") {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = pipelineVer;
    cfg.spatialRepMode = spatialMode;
    cfg.initialResMode = resMode;
    const CamResPreset p = cfg.resolveInitialRes();
    cfg.camWidth = p.width;
    cfg.camHeight = p.height;
    return cfg;
}

TEST_F(GaussianFormer3DGpuTest, YamlDefaultLegacyRemainsInvariant) {
    const SimConfig yamlCfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    EXPECT_EQ(yamlCfg.spatialRepMode, "legacy");
    EXPECT_FALSE(yamlCfg.isGaussianFormer3D());

    WorldEngine world(yamlCfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, yamlCfg, world);
    PerceptionPipeline perc(yamlCfg);
    perc.init(s_ctx, rig);

    PerceptionFrameOutput out{};
    const float dt = 1.0f / yamlCfg.controlHz;
    rig.renderAllCameras(world, yamlCfg, 0.0f);
    perc.runFrame(s_ctx, rig, world, dt, yamlCfg.occEnableShadows, out);

    EXPECT_FALSE(out.gaussianFormer3D.active);
    EXPECT_EQ(perc.gaussianFormer3D(), nullptr);

    const std::string jsonStr = TraceLogger::formatFrameJson(
        0, 0.0f, yamlCfg, world.ego(), PlannerCommand{}, out, ContactEvent{});
    Json::CharReaderBuilder rb;
    Json::Value parsed;
    std::string errs;
    std::istringstream iss(jsonStr);
    ASSERT_TRUE(Json::parseFromStream(rb, iss, &parsed, &errs)) << errs;
    EXPECT_FALSE(parsed.isMember("gaussianformer3d"));

    perc.destroy();
    rig.destroy();
}

TEST_F(GaussianFormer3DGpuTest, V15GaussianFormer3DRendersEwaHudAndTraceAcrossResolutions) {
    const std::string resModes[] = {"half", "full"};
    for (const auto& res : resModes) {
        const SimConfig cfg = makeConfig("v15", "gaussianformer3d", res);
        WorldEngine world(cfg);
        CameraRigRenderer rig;
        rig.init(s_ctx, cfg, world);
        PerceptionPipeline perc(cfg);
        perc.init(s_ctx, rig);

        PerceptionFrameOutput out{};
        const float dt = 1.0f / cfg.controlHz;
        for (int s = 0; s < 3; ++s) {
            rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt);
            perc.runFrame(s_ctx, rig, world, dt, cfg.occEnableShadows, out);
            world.stepAgents(dt);
        }

        EXPECT_TRUE(out.gaussianFormer3D.active) << res;
        EXPECT_TRUE(out.gaussianFormer3D.usedDeformableAttn) << res;
        EXPECT_GT(out.gaussianFormer3D.gaussianCount, 0u) << res;
        EXPECT_GT(out.gaussianFormer3D.splattedVoxelCount, 0u) << res;
        EXPECT_GT(out.gaussianFormer3D.camCorroboratedCount, 0u) << res;
        EXPECT_GE(out.gaussianFormer3D.meanSemanticConf, kMinMeanSemConf) << res;

        EXPECT_TRUE(out.v15.active) << res;
        EXPECT_EQ(out.v15Gaussians.size(), out.gaussianFormer3D.gaussianCount) << res;
        EXPECT_EQ(out.v15RealityRgba.size(), v15_hud::kRealityExpectedBytes) << res;

        size_t nonZeroRgb = 0u;
        for (size_t i = 0; i + v15_hud::kRealityRgbaChannels <= out.v15RealityRgba.size();
             i += v15_hud::kRealityRgbaChannels) {
            if (out.v15RealityRgba[i] > kMinNonZeroRgb ||
                out.v15RealityRgba[i + 1u] > kMinNonZeroRgb ||
                out.v15RealityRgba[i + 2u] > kMinNonZeroRgb) {
                ++nonZeroRgb;
            }
        }
        EXPECT_GT(nonZeroRgb, 0u) << res;

        const auto hud = v15_hud::formatV15HudTelemetry(world.ego(), out, PlannerCommand{});
        EXPECT_NE(std::string(hud.header.data()).find("GF3D:"), std::string::npos) << res;

        const std::string jsonStr = TraceLogger::formatFrameJson(
            2, 2.0f * dt, cfg, world.ego(), PlannerCommand{}, out, ContactEvent{});
        Json::CharReaderBuilder rb;
        Json::Value parsed;
        std::string errs;
        std::istringstream iss(jsonStr);
        ASSERT_TRUE(Json::parseFromStream(rb, iss, &parsed, &errs)) << errs;
        ASSERT_TRUE(parsed.isMember("gaussianformer3d"));
        ASSERT_TRUE(parsed.isMember("v15"));
        EXPECT_EQ(parsed["gaussianformer3d"]["gaussian_count"].asUInt(),
                  out.gaussianFormer3D.gaussianCount);

        perc.destroy();
        rig.destroy();
    }
}

TEST_F(GaussianFormer3DGpuTest, ClosedLoopV14AndV15ZeroCollisionsAndSteadyState) {
    struct RunCase {
        std::string pipeline;
        bool jaywalk;
        int steps;
    };
    const RunCase cases[] = {
        {"v14", false, 36},
        {"v15", false, 36},
        {"v15", true, 48},
    };

    for (const auto& rc : cases) {
        const SimConfig cfg = makeConfig(rc.pipeline, "gaussianformer3d", "half");
        WorldEngine world(cfg);
        if (rc.jaywalk) world.triggerJaywalkerCutIn();

        CameraRigRenderer rig;
        rig.init(s_ctx, cfg, world);
        PerceptionPipeline perc(cfg);
        perc.init(s_ctx, rig);
        TrajectoryPlanner planner(cfg);

        PerceptionFrameOutput out{};
        out.activeVoxels.reserve(static_cast<size_t>(cfg.occTotalVoxels()));
        PlannerCommand cmd{};
        const float dt = 1.0f / cfg.controlHz;
        const Vec3 startPos = world.ego().pos;
        bool collision = false;
        uint32_t maxGauss = 0u;
        uint32_t maxCorrob = 0u;

        for (int s = 0; s < rc.steps; ++s) {
            rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt);
            perc.runFrame(s_ctx, rig, world, dt, cfg.occEnableShadows, out);
            planner.plan(world, out, cmd);
            world.stepAgents(dt);
            world.stepEgoAndPhysics(dt, cmd, {});
            if (world.lastContact().occurred) collision = true;
            maxGauss = std::max(maxGauss, out.gaussianFormer3D.gaussianCount);
            maxCorrob = std::max(maxCorrob, out.gaussianFormer3D.camCorroboratedCount);
            EXPECT_TRUE(out.gaussianFormer3D.active);
        }

        const float distM = (world.ego().pos - startPos).norm();
        EXPECT_FALSE(collision) << rc.pipeline << " jaywalk=" << rc.jaywalk;
        EXPECT_GT(maxGauss, 0u) << rc.pipeline;
        EXPECT_GT(maxCorrob, 0u) << rc.pipeline;
        if (!rc.jaywalk) {
            EXPECT_GT(distM, 3.0f) << rc.pipeline;
        }
        RecordProperty(rc.pipeline + (rc.jaywalk ? "_jaywalk_max_gauss" : "_nom_max_gauss"),
                       std::to_string(maxGauss));

        perc.destroy();
        rig.destroy();
    }
}

} // namespace
} // namespace cybercab::sota
