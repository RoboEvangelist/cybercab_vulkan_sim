#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

namespace cybercab {
namespace {

using V15EndToEndClosedLoopTest = testutil::SharedGpuMetricsTest;

struct LoopStats {
    bool collision = false;
    bool seenSirenYield = false;
    bool seenSoiledMrm = false;
    float maxAbsLat = 0.0f;
    float maxCruiseDevM = 0.0f;
    int dividerFrames = 0;
    double meanRecall = 0.0;
    double meanPrec = 0.0;
    double meanGpuMs = 0.0;
    double meanWallMs = 0.0;
};

LoopStats runClosedLoop(VulkanContext& ctx, SimConfig cfg, const std::string& mode, int steps) {
    WorldEngine world(cfg);
    if (mode == "jaywalk") world.triggerJaywalkerCutIn();
    else if (mode == "occluded") world.triggerOccludedDartOut();
    else if (mode == "hardbrake") world.triggerLeadHardBrake();
    else if (mode == "siren") world.triggerEmergencySiren();
    else if (mode == "soil_partial") {
        cfg.cameras[0].soiled = cfg.cameras[2].soiled = cfg.cameras[3].soiled = true;
    } else if (mode == "soil_all_front") {
        cfg.cameras[0].soiled = cfg.cameras[1].soiled = cfg.cameras[2].soiled = true;
    }

    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perc(cfg);
    perc.init(ctx, rig);
    TrajectoryPlanner planner(cfg);
    GpuProfiler profiler;
    profiler.init(ctx, 64);

    LoopStats st{};
    PerceptionFrameOutput out;
    PlannerCommand cmd;
    const float dt = 1.0f / cfg.controlHz;
    double rSum = 0.0, pSum = 0.0, gSum = 0.0, wSum = 0.0;
    int mCnt = 0;

    for (int s = 0; s < steps; ++s) {
        const auto t0 = std::chrono::steady_clock::now();
        rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt, &profiler);
        perc.runFrame(ctx, rig, world, dt, cfg.occEnableShadows, out, &profiler);
        planner.plan(world, out, cmd);
        world.stepAgents(dt);
        world.stepEgoAndPhysics(dt, cmd, {});
        const auto t1 = std::chrono::steady_clock::now();

        if (world.lastContact().occurred) st.collision = true;
        if (cmd.behaviorState == "YIELDING_SIREN") st.seenSirenYield = true;
        if (cmd.behaviorState == "SENSOR_SOILED_MRM") st.seenSoiledMrm = true;
        st.maxAbsLat = std::max(st.maxAbsLat, std::abs(world.ego().currentLateralOffset));
        if (std::abs(world.ego().currentLateralOffset) < 1.40f) ++st.dividerFrames;
        if (cmd.behaviorState == "CRUISING" && cmd.gear == 1) {
            st.maxCruiseDevM = std::max(st.maxCruiseDevM, std::abs(std::abs(world.ego().currentLateralOffset) - 1.95f));
        }
        if (s >= 4) {
            rSum += out.eval.detRecall;
            pSum += out.eval.detPrecision;
            gSum += out.eval.totalPipelineMs;
            wSum += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ++mCnt;
        }
    }
    if (mCnt > 0) {
        st.meanRecall = rSum / mCnt;
        st.meanPrec = pSum / mCnt;
        st.meanGpuMs = gSum / mCnt;
        st.meanWallMs = wSum / mCnt;
    }
    profiler.destroy();
    perc.destroy();
    rig.destroy();
    return st;
}

TEST_F(V15EndToEndClosedLoopTest, NominalMiddleOfLanePerceptionAndFullResBudget) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    const LoopStats nom = runClosedLoop(s_ctx, cfg, "nominal", 90);
    EXPECT_FALSE(nom.collision);
    EXPECT_LE(nom.maxAbsLat, 2.45f);
    EXPECT_LE(nom.dividerFrames, 30);
    EXPECT_LT(nom.maxCruiseDevM, 0.35f);
    EXPECT_GT(nom.meanRecall, 0.50);
    EXPECT_GT(nom.meanPrec, 0.50);
    EXPECT_LT(nom.meanGpuMs, 27.8);
    EXPECT_LT(nom.meanWallMs, 27.8);

    // Full-res (640x480) latency check across all v15 stages.
    cfg.initialResMode = "full";
    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    PerceptionPipeline perc(cfg);
    perc.init(s_ctx, rig);
    GpuProfiler profiler;
    profiler.init(s_ctx, 64);
    PerceptionFrameOutput out;
    const float dt = 1.0f / cfg.controlHz;
    double fullGpu = 0.0;
    for (int f = 0; f < 8; ++f) {
        rig.renderAllCameras(world, cfg, dt * f, &profiler);
        perc.runFrame(s_ctx, rig, world, dt, true, out, &profiler);
        world.stepAgents(dt);
        if (f >= 3) fullGpu += out.eval.totalPipelineMs;
    }
    EXPECT_EQ(out.gpuStageMs.count("1c_v15_freq_split"), 1u);
    EXPECT_EQ(out.gpuStageMs.count("5_v15_static_mem"), 1u);
    EXPECT_EQ(out.gpuStageMs.count("6_v15_gaussian_splat"), 1u);
    EXPECT_LT(fullGpu / 5.0, 27.8);

    RecordProperty("v15_nom_recall", testutil::fmtProp(nom.meanRecall));
    RecordProperty("v15_nom_precision", testutil::fmtProp(nom.meanPrec));
    RecordProperty("v15_nom_cruise_dev_m", testutil::fmtProp(nom.maxCruiseDevM));
    RecordProperty("v15_half_gpu_ms", testutil::fmtProp(nom.meanGpuMs));
    RecordProperty("v15_full_gpu_ms", testutil::fmtProp(fullGpu / 5.0));
    profiler.destroy();
    perc.destroy();
    rig.destroy();
}

TEST_F(V15EndToEndClosedLoopTest, HazardAndMultimodalScenariosHaveZeroCollisions) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    for (const char* mode : {"jaywalk", "occluded", "hardbrake", "siren", "soil_partial", "soil_all_front"}) {
        const LoopStats st = runClosedLoop(s_ctx, cfg, mode, 45);
        EXPECT_FALSE(st.collision) << "Collision in v15 scenario: " << mode;
        if (std::string(mode) == "siren") { EXPECT_TRUE(st.seenSirenYield); }
        if (std::string(mode) == "soil_all_front") { EXPECT_TRUE(st.seenSoiledMrm); }
    }
}

} // namespace
} // namespace cybercab
