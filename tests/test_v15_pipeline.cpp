#include "MetricsEvalHelpers.hpp"
#include "core/GpuProfiler.hpp"
#include "perception/PerceptionPipeline.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cybercab {
namespace {

using V15PipelineTest = testutil::SharedGpuMetricsTest;

TEST_F(V15PipelineTest, V14NeverAllocatesOrRunsV15FrontEnd) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    ASSERT_FALSE(cfg.isV15());

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(s_ctx, rig);
    GpuProfiler profiler;
    profiler.init(s_ctx, 64);

    EXPECT_EQ(perception.v15(), nullptr);

    PerceptionFrameOutput out;
    const float dt = 1.0f / cfg.controlHz;
    for (int i = 0; i < 2; ++i) {
        rig.renderAllCameras(world, cfg, static_cast<float>(i) * dt, &profiler);
        perception.runFrame(s_ctx, rig, world, dt, true, out, &profiler);
        world.stepAgents(dt);
    }

    EXPECT_EQ(out.gpuStageMs.count("1c_v15_freq_split"), 0u);
    EXPECT_FALSE(out.activeVoxels.empty());

    rig.resize(cfg.fullCamWidth, cfg.fullCamHeight);
    perception.onCameraResized(rig);
    perception.reset();
    perception.destroy();
    EXPECT_EQ(perception.v15(), nullptr);

    profiler.destroy();
    rig.destroy();
}

TEST_F(V15PipelineTest, V15RunsFreqSplitInBatch1RespectsResizeAndBudget) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    ASSERT_TRUE(cfg.isV15());

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(s_ctx, rig);
    GpuProfiler profiler;
    profiler.init(s_ctx, 64);

    ASSERT_NE(perception.v15(), nullptr);
    EXPECT_TRUE(perception.v15()->isInitialized());
    EXPECT_EQ(perception.v15()->frequencySplit().width(), cfg.camWidth);
    EXPECT_EQ(perception.v15()->frequencySplit().height(), cfg.camHeight);
    EXPECT_FLOAT_EQ(perception.v15()->frequencySplit().params().sigmaPx, 4.0f);

    const float dt = 1.0f / cfg.controlHz;
    PerceptionFrameOutput out;
    double freqMsSum = 0.0, gpuSum = 0.0, wallSum = 0.0;
    constexpr int kWarm = 4;
    constexpr int kMeasure = 10;
    for (int s = 0; s < kWarm + kMeasure; ++s) {
        const auto t0 = std::chrono::steady_clock::now();
        rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt, &profiler);
        perception.runFrame(s_ctx, rig, world, dt, true, out, &profiler);
        world.stepAgents(dt);
        const auto t1 = std::chrono::steady_clock::now();
        ASSERT_EQ(out.gpuStageMs.count("1c_v15_freq_split"), 1u);
        if (s >= kWarm) {
            freqMsSum += out.gpuStageMs.at("1c_v15_freq_split");
            for (const auto& kv : out.gpuStageMs) gpuSum += kv.second;
            wallSum += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
    }
    const double freqMs = freqMsSum / kMeasure;
    const double wallMs = wallSum / kMeasure;
    RecordProperty("v15_freq_split_ms", testutil::fmtProp(freqMs));
    RecordProperty("v15_gpu_sum_ms", testutil::fmtProp(gpuSum / kMeasure));
    RecordProperty("v15_wall_ms", testutil::fmtProp(wallMs));
    EXPECT_LT(freqMs, 2.0);
    EXPECT_LT(wallMs, 27.8); // within 36 Hz v15 frame budget

    std::vector<float> bands(static_cast<size_t>(8 * cfg.camWidth * cfg.camHeight * 4));
    perception.v15()->frequencySplit().bandsSSBO().downloadVector(bands);
    double highSum = 0.0, gradAbsSum = 0.0;
    for (size_t i = 0; i < bands.size(); i += 4) {
        ASSERT_TRUE(std::isfinite(bands[i]) && std::isfinite(bands[i + 1]) &&
                    std::isfinite(bands[i + 2]) && std::isfinite(bands[i + 3]));
        highSum += bands[i + 1];
        gradAbsSum += std::abs(bands[i + 2]) + std::abs(bands[i + 3]);
    }
    const size_t pxCount = bands.size() / 4;
    EXPECT_LT(std::abs(highSum / static_cast<double>(pxCount)), 0.05);
    EXPECT_GT(gradAbsSum / static_cast<double>(pxCount), 1e-3);

    rig.resize(cfg.fullCamWidth, cfg.fullCamHeight);
    perception.onCameraResized(rig);
    perception.reset();
    EXPECT_EQ(perception.v15()->frequencySplit().width(), cfg.fullCamWidth);
    EXPECT_EQ(perception.v15()->frequencySplit().height(), cfg.fullCamHeight);
    EXPECT_FLOAT_EQ(perception.v15()->frequencySplit().params().sigmaPx, 8.0f);
    EXPECT_EQ(perception.v15()->frequencySplit().bandsSSBO().size(),
              static_cast<VkDeviceSize>(8 * cfg.fullCamWidth * cfg.fullCamHeight * 4 * sizeof(float)));

    rig.renderAllCameras(world, cfg, 0.5f, &profiler);
    perception.runFrame(s_ctx, rig, world, dt, true, out, &profiler);
    EXPECT_EQ(out.gpuStageMs.count("1c_v15_freq_split"), 1u);

    perception.destroy();
    EXPECT_EQ(perception.v15(), nullptr);
    profiler.destroy();
    rig.destroy();
}

TEST_F(V15PipelineTest, GuardsRejectNonV15OrNonHdrRigAndUninitializedDispatchIsSafe) {
    SimConfig v14Cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    WorldEngine world(v14Cfg);
    CameraRigRenderer v14Rig;
    v14Rig.init(s_ctx, v14Cfg, world);

    v15::V15Pipeline pipe;
    EXPECT_FALSE(pipe.isInitialized());
    VkCommandBuffer cmd = s_ctx.beginSingleTimeCommands();
    pipe.recordFrontEnd(cmd, nullptr); // safe no-op
    pipe.onCameraResized(v14Rig);      // safe no-op
    s_ctx.endSingleTimeCommands(cmd);

    EXPECT_THROW(pipe.init(s_ctx, v14Cfg, v14Rig), std::runtime_error);
    SimConfig v15Cfg = v14Cfg;
    v15Cfg.pipelineVersion = "v15";
    EXPECT_THROW(pipe.init(s_ctx, v15Cfg, v14Rig), std::runtime_error);

    pipe.destroy();
    v14Rig.destroy();
}

} // namespace
} // namespace cybercab
