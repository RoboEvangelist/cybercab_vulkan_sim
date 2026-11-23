#include "config/SimConfig.hpp"
#include "core/GpuProfiler.hpp"
#include "core/Math.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanImage.hpp"
#include "core/VulkanPipeline.hpp"
#include "planning/RssCorridor.hpp"
#include "planning/SimulationSync.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "world/WorldEngine.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

namespace cybercab {
namespace {

TEST(CoreCoverageTest, GpuProfilerRecordsStagesAndHandlesOverflowCleanly) {
    VulkanContext ctx;
    ctx.init(true, nullptr);

    VulkanBuffer src;
    VulkanBuffer dst;
    constexpr VkDeviceSize kBytes = 4096;
    src.create(ctx, kBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    dst.create(ctx, kBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::vector<uint32_t> pattern(1024);
    std::iota(pattern.begin(), pattern.end(), 100u);
    src.uploadVector(pattern);

    // 4 queries accommodate 2 stages cleanly; 3rd stage exceeds capacity and must be safely ignored.
    GpuProfiler profiler;
    profiler.init(ctx, 4);
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    profiler.reset(cmd);
    profiler.beginStage(cmd, "stage_copy_a");
    VkBufferCopy cpy{0, 0, kBytes};
    vkCmdCopyBuffer(cmd, src.handle(), dst.handle(), 1, &cpy);
    profiler.endStage(cmd, "stage_copy_a");

    profiler.beginStage(cmd, "stage_copy_b");
    vkCmdCopyBuffer(cmd, dst.handle(), src.handle(), 1, &cpy);
    profiler.endStage(cmd, "stage_copy_b");

    profiler.beginStage(cmd, "stage_overflow_ignored");
    profiler.endStage(cmd, "stage_overflow_ignored");
    ctx.endSingleTimeCommands(cmd);

    const auto timings = profiler.resolveTimingsMs();
    EXPECT_TRUE(timings.count("stage_copy_a") == 1u);
    EXPECT_TRUE(timings.count("stage_copy_b") == 1u);
    EXPECT_EQ(timings.count("stage_overflow_ignored"), 0u);
    EXPECT_TRUE(std::isfinite(timings.at("stage_copy_a")));
    EXPECT_GE(timings.at("stage_copy_a"), 0.0);
    EXPECT_GE(timings.at("stage_copy_b"), 0.0);

    profiler.destroy();
    profiler.destroy(); // Idempotent double-destroy
    dst.destroy();
    src.destroy();
    EXPECT_EQ(ctx.validationIssueCount(), 0u);
    ctx.cleanup();
}

TEST(CoreCoverageTest, VulkanBufferDeviceLocalStagingPreferredCachedOffsetAndMove) {
    VulkanContext ctx;
    ctx.init(true, nullptr);

    std::vector<uint32_t> data(512);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint32_t>(i * 17u + 3u);

    // 1. DEVICE_LOCAL staging upload & download round-trip
    VulkanBuffer devBuf;
    devBuf.create(ctx, data.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    devBuf.uploadVector(data);
    std::vector<uint32_t> readBack(data.size(), 0u);
    devBuf.downloadVector(readBack);
    EXPECT_EQ(readBack, data);

    // 2. Move construction & move assignment transfer ownership cleanly
    VulkanBuffer movedCtor(std::move(devBuf));
    EXPECT_TRUE(devBuf.handle() == VK_NULL_HANDLE);
    EXPECT_EQ(devBuf.size(), 0u);
    std::vector<uint32_t> movedData(data.size(), 0u);
    movedCtor.downloadVector(movedData);
    EXPECT_EQ(movedData, data);

    VulkanBuffer movedAssign;
    movedAssign = std::move(movedCtor);
    EXPECT_TRUE(movedCtor.handle() == VK_NULL_HANDLE);
    movedData.assign(data.size(), 0u);
    movedAssign.downloadVector(movedData);
    EXPECT_EQ(movedData, data);
    movedAssign.destroy();

    // 3. Preferred HOST_CACHED with fallback & non-zero offset upload/flush/invalidate/download
    VulkanBuffer cachedBuf;
    cachedBuf.createPreferred(ctx, 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    cachedBuf.map();
    ASSERT_NE(cachedBuf.mappedData(), nullptr);
    cachedBuf.upload(data.data(), 256, 256);
    cachedBuf.flushCache();
    cachedBuf.invalidateCache();
    std::vector<uint32_t> sliceOut(64, 0u);
    cachedBuf.download(sliceOut.data(), 256, 256);
    for (size_t i = 0; i < sliceOut.size(); ++i) {
        EXPECT_EQ(sliceOut[i], data[i]);
    }
    cachedBuf.destroy();

    EXPECT_EQ(ctx.validationIssueCount(), 0u);
    ctx.cleanup();
}

TEST(CoreCoverageTest, VulkanImageRoundTripTransitionsMoveAndPipelineErrorPaths) {
    VulkanContext ctx;
    ctx.init(true, nullptr);

    constexpr uint32_t W = 32, H = 24;
    VulkanImage img;
    img.create(ctx, W, H, VK_FORMAT_R8G8B8A8_UNORM,
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
               VK_IMAGE_ASPECT_COLOR_BIT, true, VK_FILTER_NEAREST);
    EXPECT_EQ(img.layout(), VK_IMAGE_LAYOUT_UNDEFINED);
    EXPECT_TRUE(img.view() != VK_NULL_HANDLE);
    EXPECT_TRUE(img.sampler() != VK_NULL_HANDLE);

    std::vector<uint8_t> pixels(W * H * 4u);
    for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<uint8_t>((i * 31u + 7u) & 0xFFu);
    img.uploadRGBA8(pixels.data(), W, H);
    EXPECT_EQ(img.layout(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    VulkanImage moved(std::move(img));
    EXPECT_TRUE(img.handle() == VK_NULL_HANDLE);
    const auto got = moved.downloadRGBA8();
    EXPECT_EQ(got, pixels);

    moved.transitionLayoutImmediate(VK_IMAGE_LAYOUT_GENERAL);
    EXPECT_EQ(moved.layout(), VK_IMAGE_LAYOUT_GENERAL);
    moved.destroy();

    // Missing / malformed SPIR-V error paths + resolveAssetPath checks
    EXPECT_THROW(readSpirvFile("/tmp/cybercab_nonexistent_shader_88888.spv"), std::runtime_error);
    const auto badSpv = std::filesystem::temp_directory_path() / "cybercab_bad_spv_unit.spv";
    {
        std::ofstream ofs(badSpv, std::ios::binary);
        ofs << "123"; // 3 bytes (not multiple of uint32_t / invalid)
    }
    EXPECT_THROW(readSpirvFile(badSpv.string()), std::runtime_error);
    std::filesystem::remove(badSpv);

    EXPECT_TRUE(std::filesystem::exists(resolveAssetPath("config/cybercab_sim.yaml")));
    EXPECT_FALSE(resolveAssetPath("config/no_such_asset_xyz.yaml").empty());

    EXPECT_EQ(ctx.validationIssueCount(), 0u);
    ctx.cleanup();
}

TEST(CoreCoverageTest, SimulationSyncClockClampsDtRolloverAndResets) {
    SimulationSyncClock clock;
    constexpr float kControlDt = 1.0f / 36.0f;
    constexpr float kMaxDt = kControlDt * 2.0f;

    int percCalls = 0;
    int ticks = 0;
    // Huge dt spike (0.50s) must clamp to kMaxDt (exactly 2 ticks)
    const int ran = clock.stepFrame(0.50f, kControlDt, kMaxDt,
                                    [&](float, float pDt) {
                                        ++percCalls;
                                        EXPECT_NEAR(pDt, kControlDt, 1e-4f);
                                    },
                                    [&](int, float, float tDt) {
                                        ++ticks;
                                        EXPECT_NEAR(tDt, kControlDt, 1e-6f);
                                    });
    EXPECT_EQ(ran, 2);
    EXPECT_EQ(ticks, 2);
    EXPECT_EQ(percCalls, 1);
    EXPECT_NEAR(clock.perceptionDt(kControlDt), kControlDt * 2.0f, 1e-4f);

    clock.reset();
    EXPECT_FLOAT_EQ(clock.simTimeSec, 0.0f);
    EXPECT_FLOAT_EQ(clock.accumSec, 0.0f);
    EXPECT_FLOAT_EQ(clock.perceptionDt(kControlDt), kControlDt);
}

TEST(WorldCoverageTest, ManualInputDriveModeSignalCycleHardBrakeAndGtOBBs) {
    SimConfig cfg = SimConfig::defaultConfig();
    cfg.driveMode = "manual";
    WorldEngine world(cfg);

    PlannerCommand dummyCmd;
    const float dt = 1.0f / cfg.controlHz;
    world.mutableEgo().speed = 0.0f;
    const Vec3 p0 = world.ego().pos;

    ManualInput fwdLeft{true, false, true, false};
    for (int i = 0; i < 20; ++i) {
        world.stepAgents(dt);
        world.stepEgoAndPhysics(dt, dummyCmd, fwdLeft);
    }
    EXPECT_GT(world.ego().speed, 1.5f);
    EXPECT_GT(world.ego().steerAngle, 0.05f);
    EXPECT_GT((world.ego().pos - p0).norm(), 0.4f);

    ManualInput brakeRight{false, true, false, true};
    const float speedPeak = world.ego().speed;
    for (int i = 0; i < 15; ++i) {
        world.stepAgents(dt);
        world.stepEgoAndPhysics(dt, dummyCmd, brakeRight);
    }
    EXPECT_LT(world.ego().speed, speedPeak);
    EXPECT_LT(world.ego().steerAngle, 0.05f);

    // Signal cycle & forced red override
    world.reset();
    world.setForcedRedSignal(true);
    world.stepAgents(dt);
    EXPECT_EQ(world.signalPhase(), SignalPhase::Red);
    world.setForcedRedSignal(false);

    bool sawYellow = false, sawGreen = false;
    for (int i = 0; i < 900; ++i) {
        world.stepAgents(dt);
        if (world.signalPhase() == SignalPhase::Yellow) sawYellow = true;
        if (world.signalPhase() == SignalPhase::Green) sawGreen = true;
    }
    EXPECT_TRUE(sawYellow);
    EXPECT_TRUE(sawGreen);

    // triggerLeadHardBrake drops lead vehicles()[1] speed; allGroundTruthOBBs covers static + dynamic
    world.reset();
    world.triggerLeadHardBrake();
    const float leadSpdBefore = world.vehicles().at(1).speed;
    for (int i = 0; i < 25; ++i) world.stepAgents(dt);
    EXPECT_LT(world.vehicles().at(1).speed, leadSpdBefore - 1.5f);

    const auto allObbs = world.allGroundTruthOBBs();
    const auto dynObbs = world.dynamicGroundTruthOBBs();
    EXPECT_GT(allObbs.size(), dynObbs.size());
    bool hasSignalMast = false, hasStopSign = false;
    for (const auto& o : allObbs) {
        if (o.id == "SIGNAL_MAST") hasSignalMast = true;
        if (o.id == "SIGN_STOP") hasStopSign = true;
    }
    EXPECT_TRUE(hasSignalMast);
    EXPECT_TRUE(hasStopSign);
}

TEST(PlanningCoverageTest, RssCorridorEdgeCasesAndPlannerMrmRedLightStopSignStates) {
    const SimConfig cfg = SimConfig::defaultConfig();

    // 1. projectOntoPolyline & resamplePolyline degenerate handling + signed extensions
    float sOut = 0.0f, latOut = 0.0f, tx = 0.0f, tz = 0.0f;
    rss::projectOntoPolyline({}, {}, 1.0f, 2.0f, sOut, latOut, tx, tz);
    EXPECT_FLOAT_EQ(sOut, 1e6f);

    std::vector<Vec3> linePts{{0.0f, 0.08f, 0.0f}, {0.0f, 0.08f, 10.0f}, {5.0f, 0.08f, 15.0f}};
    std::vector<float> cumS{0.0f, 10.0f, 10.0f + std::hypot(5.0f, 5.0f)};
    rss::projectOntoPolyline(linePts, cumS, 2.0f, -3.0f, sOut, latOut, tx, tz);
    EXPECT_NEAR(sOut, -3.0f, 1e-4f);
    EXPECT_NEAR(latOut, std::hypot(2.0f, -3.0f), 1e-4f);

    rss::projectOntoPolyline(linePts, cumS, 7.0f, 17.0f, sOut, latOut, tx, tz);
    EXPECT_GT(sOut, cumS.back());
    EXPECT_TRUE(std::isfinite(latOut));

    std::vector<Vec3> resampled;
    rss::resamplePolyline(linePts, cumS, 0.1f, 5, resampled);
    ASSERT_EQ(resampled.size(), 5u);
    EXPECT_NEAR(resampled.front().z, 0.0f, 1e-4f);
    EXPECT_NEAR(resampled.back().z, 0.5f, 1e-3f); // clamped minimum targetLen = 0.5f

    // 2. Manual steering bicycle-arc corridor
    WorldEngine world(cfg);
    world.mutableEgo().steerAngle = 0.25f;
    std::vector<Vec3> arcPts;
    std::vector<float> arcCumS;
    rss::buildRssCorridorPoints(world.ego(), world.centerline(), world.centerline().getLength(),
                                world.findClosestLoopU(world.ego().pos), 20.0f, -1.9f, -1.9f,
                                cfg.wheelbaseM, true, 16.0f, arcPts, arcCumS);
    ASSERT_GE(arcPts.size(), 10u);
    // Turning left from heading +X moves trajectory into smaller Z (-Z = Ego ISO +Y left)
    EXPECT_LT(arcPts.back().z, arcPts.front().z - 1.0f);

    // 3. TrajectoryPlanner SENSOR_SOILED_MRM (all 3 front cams soiled)
    TrajectoryPlanner planner(cfg);
    PerceptionFrameOutput soiledPerc{};
    for (int c = 0; c < 3; ++c) {
        soiledPerc.soilCameras[c].meanSoilProb = 0.90f;
        soiledPerc.soilCameras[c].tokenWeight = 0.15f;
    }
    world.reset();
    world.mutableEgo().speed = 6.0f;
    PlannerCommand mrmCmd;
    planner.plan(world, soiledPerc, mrmCmd);
    EXPECT_EQ(mrmCmd.behaviorState, "SENSOR_SOILED_MRM");
    EXPECT_FLOAT_EQ(mrmCmd.targetSpeed, 0.0f);
    EXPECT_TRUE(mrmCmd.aebActive);

    // 4. TrajectoryPlanner RED_LIGHT_HOLD
    world.reset();
    world.setForcedRedSignal(true);
    world.mutableEgo().pos = {WorldEngine::kStopLineX - 8.0f, 0.0f, 45.9f};
    world.mutableEgo().yaw = kPi * 0.5f;
    world.mutableEgo().speed = 7.0f;
    PlannerCommand redCmd;
    planner.plan(world, PerceptionFrameOutput{}, redCmd);
    EXPECT_EQ(redCmd.behaviorState, "RED_LIGHT_HOLD");
    EXPECT_LT(redCmd.targetSpeed, 6.0f);

    // 5. TrajectoryPlanner STOP_SIGN_HOLD timer accumulation & cooldown release
    world.reset();
    world.mutableEgo().pos = {18.5f, 0.0f, -45.9f};
    world.mutableEgo().yaw = -kPi * 0.5f;
    world.mutableEgo().speed = 0.15f;
    world.mutableEgo().stopSignHoldTimer = 0.0f;
    world.mutableEgo().passedStopSignCooldown = 0.0f;
    PlannerCommand stopCmd;
    bool heldZero = false;
    bool releasedCruise = false;
    for (int i = 0; i < 65; ++i) {
        planner.plan(world, PerceptionFrameOutput{}, stopCmd);
        if (stopCmd.behaviorState == "STOP_SIGN_HOLD" && stopCmd.targetSpeed == 0.0f) heldZero = true;
        if (world.ego().passedStopSignCooldown > 0.0f && stopCmd.behaviorState == "CRUISING") {
            releasedCruise = true;
            break;
        }
    }
    EXPECT_TRUE(heldZero);
    EXPECT_TRUE(releasedCruise);
}

} // namespace
} // namespace cybercab
